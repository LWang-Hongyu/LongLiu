/*
 * ============================================================================
 * Experiment C — CPU Epoch Emulator (ibv verbs RDMA write, multi-QP DSCP)
 * ============================================================================
 *
 * ARCHITECTURE CHANGE (2026-07-29):
 *   Previous version tried ibv_modify_qp(IBV_QP_AV) on a live QP to update
 *   DSCP. mlx5_0 rejects this with EINVAL (rc=22) regardless of mask
 *   combinations — same finding that drove NCCL LongLiu8 to the multi-
 *   communicator design (multi_comm_slo/DESIGN.md). This emulator now uses
 *   the same approach: pre-create NUM_QPS QPs at setup time, each pinned to
 *   a different DSCP at RTR transition. Runtime "DSCP change" = switch
 *   active_qp_idx — no ibv_modify_qp call, O(1) switching.
 *
 * One process per job per node. Server (N2/226) + Client (N1/10.1) pair.
 *
 * Behavior loop (client side):
 *   1. sleep(T_comp + jitter)            — compute phase
 *   2. Read DSCP from /tmp/expC_dscp_<job_id>
 *   3. If DSCP changed → switch active_qp_idx (no IB call)
 *   4. RDMA write D_j bytes via qp[active_qp_idx]
 *   5. Poll completion
 *   6. Log epoch stats to /tmp/expC_stats_<job_id>.csv
 *
 * Usage:
 *   Server: epoch_emulator --server --port <port> --job-id <id> \
 *           --num-epochs <n> --iters-per-epoch <n>
 *   Client: epoch_emulator --client --host <ip> --port <port> --job-id <id> \
 *           --data-size <bytes> --sleep-us <us> --num-epochs <n> \
 *           --iters-per-epoch <n> [--jitter-pct <p>]
 *
 * Compile:
 *   gcc -O2 -Wall -o epoch_emulator epoch_emulator.c -libverbs -lpthread
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <signal.h>
#include <getopt.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>

#define MAX_DATA_SIZE (256 * 1024 * 1024)  /* 256 MB — v2 S2 premium needs ~115MB/iter */
#define CTRL_FILE_FMT "/tmp/expC_dscp_%d"
#define STATS_FILE_FMT "/tmp/expC_stats_%d.csv"
#define STATS_HEADER "epoch,iter,comm_us,data_bytes,dscp,bw_gbps,sleep_us\n"

/* Multi-QP DSCP table — mirrors SLOScheduler.PRIORITY_TO_DSCP (paper §5.3).
 * Order: index 0..3, used to select active QP at runtime.
 *   P6 → DSCP=8   (highest)
 *   P4 → DSCP=0
 *   P2 → DSCP=24
 *   P1 → DSCP=32  (lowest)
 */
#define NUM_QPS 4
static const uint8_t DSCP_TABLE[NUM_QPS] = {8, 0, 24, 32};
static const char *PRIO_TABLE[NUM_QPS] = {"P6", "P4", "P2", "P1"};

static int dscp_to_idx(uint8_t dscp) {
    for (int i = 0; i < NUM_QPS; i++) {
        if (DSCP_TABLE[i] == dscp) return i;
    }
    /* Unknown DSCP → default to P4 (idx=1) */
    return 1;
}

static volatile int g_running = 1;

static void sighandler(int sig) {
    (void)sig;
    g_running = 0;
}

/* ---------------------------------------------------------------------------
 * Connection helper: exchange QP info via TCP socket (NUM_QPS QPs at once)
 * ------------------------------------------------------------------------- */
struct qp_info {
    uint32_t qpn;
    uint32_t rkey;
    uint64_t vaddr;
    uint16_t lid;
    union ibv_gid gid;
};

/* Send/recv the full qp_info array (NUM_QPS entries) */
static int exchange_qp_info_all(int sock, struct qp_info *local, struct qp_info *remote,
                                int is_server) {
    size_t bytes = NUM_QPS * sizeof(struct qp_info);
    ssize_t n;
    if (is_server) {
        n = write(sock, local, bytes);
        if (n != (ssize_t)bytes) return -1;
        n = read(sock, remote, bytes);
        if (n != (ssize_t)bytes) return -1;
    } else {
        n = read(sock, remote, bytes);
        if (n != (ssize_t)bytes) return -1;
        n = write(sock, local, bytes);
        if (n != (ssize_t)bytes) return -1;
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * IB resource setup — one context, one PD, one MR, one CQ, NUM_QPS QPs
 * ------------------------------------------------------------------------- */
struct ib_res {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp[NUM_QPS];   /* one QP per DSCP */
    void *buf;
    size_t buf_size;
    struct ibv_port_attr port_attr;
    int port_num;
    union ibv_gid gid;
    int gid_index;
};

static int ib_setup(struct ib_res *res, const char *dev_name, int port_num) {
    memset(res, 0, sizeof(*res));
    res->port_num = port_num;

    /* Open device */
    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    if (!dev_list) {
        fprintf(stderr, "[ERR] ibv_get_device_list failed\n");
        return -1;
    }
    struct ibv_device *dev = NULL;
    if (dev_name) {
        for (int i = 0; dev_list[i]; i++) {
            if (strcmp(ibv_get_device_name(dev_list[i]), dev_name) == 0) {
                dev = dev_list[i];
                break;
            }
        }
    } else {
        dev = dev_list[0];
    }
    if (!dev) {
        fprintf(stderr, "[ERR] device %s not found\n", dev_name ?: "(default)");
        ibv_free_device_list(dev_list);
        return -1;
    }
    res->ctx = ibv_open_device(dev);
    ibv_free_device_list(dev_list);
    if (!res->ctx) {
        fprintf(stderr, "[ERR] ibv_open_device failed\n");
        return -1;
    }

    if (ibv_query_port(res->ctx, port_num, &res->port_attr)) {
        fprintf(stderr, "[ERR] ibv_query_port failed\n");
        return -1;
    }

    res->gid_index = 3;  /* NCCL_IB_GID_INDEX=3 */
    if (ibv_query_gid(res->ctx, port_num, res->gid_index, &res->gid)) {
        fprintf(stderr, "[ERR] ibv_query_gid(idx=%d) failed\n", res->gid_index);
        return -1;
    }

    res->pd = ibv_alloc_pd(res->ctx);
    if (!res->pd) {
        fprintf(stderr, "[ERR] ibv_alloc_pd failed\n");
        return -1;
    }

    res->buf_size = MAX_DATA_SIZE;
    if (posix_memalign(&res->buf, 4096, res->buf_size)) {
        fprintf(stderr, "[ERR] posix_memalign failed\n");
        return -1;
    }
    memset(res->buf, 0, res->buf_size);
    res->mr = ibv_reg_mr(res->pd, res->buf, res->buf_size,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
    if (!res->mr) {
        fprintf(stderr, "[ERR] ibv_reg_mr failed: %s\n", strerror(errno));
        return -1;
    }

    /* Shared CQ for all QPs (low traffic, 64 entries is plenty) */
    res->cq = ibv_create_cq(res->ctx, 64, NULL, NULL, 0);
    if (!res->cq) {
        fprintf(stderr, "[ERR] ibv_create_cq failed\n");
        return -1;
    }

    /* Create NUM_QPS QPs */
    for (int i = 0; i < NUM_QPS; i++) {
        struct ibv_qp_init_attr qp_init = {
            .send_cq = res->cq,
            .recv_cq = res->cq,
            .qp_type = IBV_QPT_RC,
            .cap = { .max_send_wr = 16, .max_recv_wr = 16,
                     .max_send_sge = 1, .max_recv_sge = 1,
                     .max_inline_data = 0 },
        };
        res->qp[i] = ibv_create_qp(res->pd, &qp_init);
        if (!res->qp[i]) {
            fprintf(stderr, "[ERR] ibv_create_qp[%d] failed: %s\n", i, strerror(errno));
            return -1;
        }
    }

    return 0;
}

/* Modify QP to INIT */
static int qp_to_init(struct ibv_qp *qp, int port_num) {
    struct ibv_qp_attr attr = {
        .qp_state = IBV_QPS_INIT,
        .port_num = port_num,
        .pkey_index = 0,
        .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC,
    };
    return ibv_modify_qp(qp, &attr,
                         IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX |
                         IBV_QP_ACCESS_FLAGS);
}

/* Modify QP to RTR — sets DSCP via traffic_class */
static int qp_to_rtr(struct ibv_qp *qp, struct ibv_port_attr *port_attr,
                     int port_num, int gid_index, union ibv_gid *local_gid,
                     struct qp_info *peer, uint8_t dscp) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = port_attr->active_mtu;
    attr.dest_qp_num = peer->qpn;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.dlid = peer->lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = port_num;
    attr.ah_attr.grh.dgid = peer->gid;
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.grh.hop_limit = 64;
    /* traffic_class = DSCP << 2 (TOS format) */
    attr.ah_attr.grh.traffic_class = dscp << 2;

    return ibv_modify_qp(qp, &attr,
                         IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                         IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                         IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

/* Modify QP to RTS */
static int qp_to_rts(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = {
        .qp_state = IBV_QPS_RTS,
        .sq_psn = 0,
        .timeout = 14,
        .retry_cnt = 7,
        .rnr_retry = 7,
        .max_rd_atomic = 1,
    };
    return ibv_modify_qp(qp, &attr,
                         IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                         IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                         IBV_QP_MAX_QP_RD_ATOMIC);
}

/* ---------------------------------------------------------------------------
 * RDMA write + poll completion (uses specified QP)
 * ------------------------------------------------------------------------- */
static int rdma_write_poll(struct ibv_qp *qp, struct ibv_cq *cq,
                           struct ibv_mr *mr, void *buf,
                           struct qp_info *peer, size_t size) {
    struct ibv_sge sge = {
        .addr = (uintptr_t)buf,
        .length = (uint32_t)size,
        .lkey = mr->lkey,
    };
    struct ibv_send_wr wr = {
        .wr_id = 0xdead,
        .sg_list = &sge,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE,
        .send_flags = IBV_SEND_SIGNALED,
        .wr.rdma.remote_addr = peer->vaddr,
        .wr.rdma.rkey = peer->rkey,
    };
    struct ibv_send_wr *bad_wr = NULL;
    if (ibv_post_send(qp, &wr, &bad_wr)) {
        fprintf(stderr, "[ERR] ibv_post_send failed: %s\n", strerror(errno));
        return -1;
    }
    struct ibv_wc wc;
    int ne;
    do {
        ne = ibv_poll_cq(cq, 1, &wc);
    } while (ne == 0 && g_running);
    if (ne < 0) {
        fprintf(stderr, "[ERR] ibv_poll_cq failed\n");
        return -1;
    }
    if (wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "[ERR] WC status %d: %s\n", wc.status,
                ibv_wc_status_str(wc.status));
        return -1;
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * DSCP control file reader
 * ------------------------------------------------------------------------- */
static int read_dscp_file(int job_id, uint8_t *dscp_out) {
    char path[64];
    snprintf(path, sizeof(path), CTRL_FILE_FMT, job_id);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int val = -1;
    if (fscanf(f, "%d", &val) == 1 && val >= 0 && val <= 63) {
        *dscp_out = (uint8_t)val;
        fclose(f);
        return 0;
    }
    fclose(f);
    return -1;
}

/* ---------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------- */
int main(int argc, char **argv) {
    int is_server = 0;
    const char *host = NULL;
    int port = 0;
    int job_id = -1;
    int data_size = 1024 * 1024;
    int sleep_us = 30000;
    int num_epochs = 25;
    int iters_per_epoch = 20;
    int jitter_pct = 0;
    const char *dev_name = NULL;

    static struct option long_opts[] = {
        {"server",         no_argument,       0, 'S'},
        {"client",         no_argument,       0, 'C'},
        {"host",           required_argument, 0, 'h'},
        {"port",           required_argument, 0, 'p'},
        {"job-id",         required_argument, 0, 'j'},
        {"data-size",      required_argument, 0, 'd'},
        {"sleep-us",       required_argument, 0, 's'},
        {"num-epochs",     required_argument, 0, 'e'},
        {"iters-per-epoch",required_argument, 0, 'i'},
        {"jitter-pct",     required_argument, 0, 'J'},
        {"device",         required_argument, 0, 'D'},
        {0, 0, 0, 0},
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "SCh:p:j:d:s:e:i:J:D:", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'S': is_server = 1; break;
            case 'C': is_server = 0; break;
            case 'h': host = optarg; break;
            case 'p': port = atoi(optarg); break;
            case 'j': job_id = atoi(optarg); break;
            case 'd': data_size = atoi(optarg); break;
            case 's': sleep_us = atoi(optarg); break;
            case 'e': num_epochs = atoi(optarg); break;
            case 'i': iters_per_epoch = atoi(optarg); break;
            case 'J': jitter_pct = atoi(optarg); break;
            case 'D': dev_name = optarg; break;
            default:
                fprintf(stderr, "Unknown option\n");
                return 1;
        }
    }
    if (!port || job_id < 0) {
        fprintf(stderr, "Usage: %s --port <p> --job-id <id> [--server|--client --host <ip>]\n", argv[0]);
        return 1;
    }
    if (!is_server && !host) {
        fprintf(stderr, "Client mode requires --host\n");
        return 1;
    }
    if (data_size > MAX_DATA_SIZE) {
        fprintf(stderr, "data_size %d > max %d\n", data_size, MAX_DATA_SIZE);
        return 1;
    }

    signal(SIGINT, sighandler);
    signal(SIGTERM, sighandler);
    srand48(time(NULL) ^ (job_id << 8));

    /* ---- TCP socket for QP info exchange ---- */
    int sock;
    if (is_server) {
        int lfd = socket(AF_INET, SOCK_STREAM, 0);
        int yes = 1;
        setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        struct sockaddr_in addr = { .sin_family = AF_INET, .sin_port = htons(port),
                                     .sin_addr.s_addr = INADDR_ANY };
        if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            fprintf(stderr, "[ERR] bind port %d: %s\n", port, strerror(errno));
            return 1;
        }
        listen(lfd, 1);
        fprintf(stderr, "[server] waiting on port %d (job %d)...\n", port, job_id);
        sock = accept(lfd, NULL, NULL);
        close(lfd);
        if (sock < 0) { fprintf(stderr, "[ERR] accept: %s\n", strerror(errno)); return 1; }
    } else {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in addr = { .sin_family = AF_INET, .sin_port = htons(port) };
        inet_pton(AF_INET, host, &addr.sin_addr);
        int retries = 30;
        while (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0 && retries-- > 0) {
            usleep(500000);
        }
        if (retries < 0) {
            fprintf(stderr, "[ERR] connect %s:%d: %s\n", host, port, strerror(errno));
            return 1;
        }
    }
    fprintf(stderr, "[%s] TCP connected (job %d)\n", is_server ? "server" : "client", job_id);

    /* ---- IB setup (creates NUM_QPS QPs) ---- */
    struct ib_res res;
    if (ib_setup(&res, dev_name, 1) < 0) {
        fprintf(stderr, "[ERR] ib_setup failed\n");
        return 1;
    }
    fprintf(stderr, "[%s] IB ready: dev=%s, gid_idx=%d, %d QPs created\n",
            is_server ? "server" : "client", res.ctx->device->name,
            res.gid_index, NUM_QPS);

    /* ---- Bring all QPs to INIT ---- */
    for (int i = 0; i < NUM_QPS; i++) {
        if (qp_to_init(res.qp[i], res.port_num)) {
            fprintf(stderr, "[ERR] qp_to_init[%d]\n", i);
            return 1;
        }
    }

    /* ---- Exchange NUM_QPS QP info entries ---- */
    struct qp_info local_arr[NUM_QPS], remote_arr[NUM_QPS];
    for (int i = 0; i < NUM_QPS; i++) {
        local_arr[i].qpn = res.qp[i]->qp_num;
        local_arr[i].rkey = res.mr->rkey;       /* same MR shared by all QPs */
        local_arr[i].vaddr = (uint64_t)res.buf;
        local_arr[i].lid = res.port_attr.lid;
        local_arr[i].gid = res.gid;
    }
    if (exchange_qp_info_all(sock, local_arr, remote_arr, is_server)) {
        fprintf(stderr, "[ERR] exchange_qp_info_all\n");
        return 1;
    }
    close(sock);

    /* ---- RTR + RTS for each QP, each with its pinned DSCP ---- */
    for (int i = 0; i < NUM_QPS; i++) {
        uint8_t dscp = DSCP_TABLE[i];
        if (qp_to_rtr(res.qp[i], &res.port_attr, res.port_num, res.gid_index,
                      &res.gid, &remote_arr[i], dscp)) {
            fprintf(stderr, "[ERR] qp_to_rtr[%d] (DSCP=%d)\n", i, dscp);
            return 1;
        }
        if (qp_to_rts(res.qp[i])) {
            fprintf(stderr, "[ERR] qp_to_rts[%d]\n", i);
            return 1;
        }
    }
    fprintf(stderr, "[%s] All %d QPs ready (RTR+RTS), DSCPs: [",
            is_server ? "server" : "client", NUM_QPS);
    for (int i = 0; i < NUM_QPS; i++) {
        fprintf(stderr, "%s=%d%s", PRIO_TABLE[i], DSCP_TABLE[i],
                i < NUM_QPS - 1 ? "," : "]\n");
    }

    /* ---- Stats file ---- */
    char stats_path[64];
    snprintf(stats_path, sizeof(stats_path), STATS_FILE_FMT, job_id);
    FILE *stats = fopen(stats_path, "w");
    if (!stats) { fprintf(stderr, "[ERR] fopen %s\n", stats_path); return 1; }
    fprintf(stats, STATS_HEADER);
    fflush(stats);

    /* ---- Main loop ----
     * Active QP index is selected from current_dscp. Initial DSCP = 0 (P4, idx=1).
     * Daemon writes /tmp/expC_dscp_<job_id>; client re-reads every iter.
     * On DSCP change: just switch active_qp_idx — no ibv_modify_qp call.
     */
    uint8_t current_dscp = 0;       /* P4 default — daemon will overwrite */
    int active_idx = dscp_to_idx(current_dscp);

    int total_iters = num_epochs * iters_per_epoch;
    int epoch = 0, iter_in_epoch = 0;
    double epoch_comm_us = 0.0;

    if (is_server) {
        /* Server: RDMA write (without immediate) doesn't generate receive
         * completions. Keep QPs alive for expected duration, then exit. */
        int est_comm_us = (data_size * 8) / (50 * 1000);  /* ~50G line rate */
        int per_iter_us = sleep_us + est_comm_us + 500;
        int duration_s = total_iters * per_iter_us / 1000000 + 10;
        fprintf(stderr, "[server] keeping %d QPs alive for ~%ds (%d iters)...\n",
                NUM_QPS, duration_s, total_iters);
        for (int s = 0; s < duration_s && g_running; s++) {
            sleep(1);
        }
        fprintf(stderr, "[server] job %d done (exiting after %ds)\n", job_id, duration_s);
    } else {
        /* Client: sleep → (check DSCP file) → RDMA write via qp[active_idx] → log */
        for (int it = 0; it < total_iters && g_running; it++) {
            int actual_sleep = sleep_us;
            if (jitter_pct > 0) {
                double jitter = (drand48() - 0.5) * 2.0 * jitter_pct / 100.0 * sleep_us;
                actual_sleep = sleep_us + (int)jitter;
                if (actual_sleep < 0) actual_sleep = 0;
            }
            usleep(actual_sleep);

            /* Check DSCP control file for updates */
            uint8_t new_dscp;
            int rd = read_dscp_file(job_id, &new_dscp);
            if (rd == 0 && new_dscp != current_dscp) {
                int new_idx = dscp_to_idx(new_dscp);
                fprintf(stderr, "[client] job %d DSCP %d→%d (%s→%s) "
                                "(epoch %d, iter %d) — switching QP idx %d→%d\n",
                        job_id, current_dscp, new_dscp,
                        PRIO_TABLE[active_idx], PRIO_TABLE[new_idx],
                        epoch, iter_in_epoch, active_idx, new_idx);
                current_dscp = new_dscp;
                active_idx = new_idx;
            }

            /* RDMA write via active QP + measure time */
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            if (rdma_write_poll(res.qp[active_idx], res.cq, res.mr, res.buf,
                                &remote_arr[active_idx], data_size) < 0) {
                fprintf(stderr, "[client] rdma_write_poll failed at epoch %d iter %d\n",
                        epoch, iter_in_epoch);
                break;
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double comm_us = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
            epoch_comm_us += comm_us;

            double bw_gbps = (data_size * 8) / (comm_us / 1e6) / 1e9;
            fprintf(stats, "%d,%d,%.1f,%d,%d,%.2f,%d\n",
                    epoch, iter_in_epoch, comm_us, data_size, current_dscp, bw_gbps, actual_sleep);

            iter_in_epoch++;
            if (iter_in_epoch >= iters_per_epoch) {
                double avg_comm_us = epoch_comm_us / iters_per_epoch;
                fprintf(stderr, "[client] job %d epoch %d done: avg_comm=%.1fus DSCP=%d (%s)\n",
                        job_id, epoch, avg_comm_us, current_dscp, PRIO_TABLE[active_idx]);
                epoch++;
                iter_in_epoch = 0;
                epoch_comm_us = 0.0;
                fflush(stats);
            }
        }
    }

    fprintf(stderr, "[%s] job %d done: %d epochs\n",
            is_server ? "server" : "client", job_id, epoch);
    fclose(stats);

    /* Cleanup */
    for (int i = 0; i < NUM_QPS; i++) {
        if (res.qp[i]) ibv_destroy_qp(res.qp[i]);
    }
    if (res.cq) ibv_destroy_cq(res.cq);
    if (res.mr) ibv_dereg_mr(res.mr);
    if (res.pd) ibv_dealloc_pd(res.pd);
    if (res.ctx) ibv_close_device(res.ctx);
    free(res.buf);
    return 0;
}
