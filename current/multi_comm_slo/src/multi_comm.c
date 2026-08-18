#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <cuda_runtime.h>
#include "nccl.h"
#include "multi_comm.h"

typedef struct {
    ncclComm_t comms[NUM_PRIORITIES][MAX_DEVICES];
    int num_devices;
    int current_priority;
    int rank;
    int size;
} MultiCommHandle;

static MultiCommHandle g_handle;

/* 按需创建：MULTI_COMM_PRIOS 环境变量（逗号分隔，如 "3,6"）指定要创建的优先级子集。
 * 默认创建全部 P0-P6。用于减少连续创建多个 NCCL communicator 时的偶发死锁概率，
 * 实验只需部分优先级时（如 test1 只用 P3/P6）显著降低挂死风险。 */
static int g_active_prios[NUM_PRIORITIES];
static int g_num_active_prios = 0;

/* Hardware DSCP mapping (aligned with testbed's measured DSCP→TC map).
 *
 * IMPORTANT: This testbed's NIC maps DSCP→TC NON-monotonically w.r.t. the
 * class-selector names. Measured with `mlnx_qos -i mlx5_0 --trust dscp` and
 * probe experiments on 10.1 (see results/testbed/HANDOFF_physical_evidence.md
 * and QUOTA_EXPERIMENT_RESULTS.md V6-P4 section):
 *
 *   tc:0(prio1,dscp8-15) > tc:1(prio0,dscp0-7) > tc:2(prio2,dscp16-23) >
 *   tc:3(prio3,dscp24-31) > tc:4(prio4,dscp32-39) > tc:5(prio5,dscp40-47) >
 *   tc:6(prio6,dscp48-55) > tc:7(prio7,dscp56-63)
 *
 * So the effective priority is NOT "higher DSCP class = higher priority".
 * Under strict priority (SP) scheduling, the ordering is by TC:
 *   tc:0 > tc:1 > tc:2 > tc:3 > tc:4 > tc:5 > tc:6 > tc:7.
 * The naive monotonic mapping (P6→DSCP=56→tc:7) put P6 in the LOWEST queue,
 * which is why P6 never preempted P3 (P3→DSCP=32→tc:4) in test1.
 *
 * CRITICAL: NCCL `config.trafficClass` is the 8-bit IP ToS byte, NOT the 6-bit
 * DSCP value. DSCP occupies the upper 6 bits of ToS, so ToS = DSCP << 2.
 *
 * Corrected mapping (software priority → DSCP → TC, SP order):
 *   Priority    DSCP  ToS(=DSCP<<2)   TC    SP order
 *     P6          8       32          tc:0   highest
 *     P4          0        0          tc:1   second
 *     P3         16       64          tc:2   third (also used by P5)
 *     P5         16       64          tc:2   third
 *     P2         24       96          tc:3   fourth
 *     P1         32      128          tc:4   fifth
 *     P0         40      160          tc:5   sixth
 */
static const int prio_dscp[NUM_PRIORITIES] = {160, 128, 96, 64, 0, 64, 32};
static const int prio_dscp_val[NUM_PRIORITIES] = {40, 32, 24, 16, 0, 16, 8};

/* 
 * TCP-based NCCL Unique ID exchange.
 * Rank 0 acts as server, rank 1+ as clients.
 * Exchanges NUM_PRIORITIES unique IDs.
 */
static int exchange_ids_via_tcp(int rank, const char* master_addr, int port,
                                 ncclUniqueId* ids, int nids) {
    if (rank == 0) {
        /* Server: generate IDs and send to all other ranks */
        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) {
            fprintf(stderr, "[MultiComm] Socket creation failed\n");
            return -1;
        }
        
        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);
        
        if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            fprintf(stderr, "[MultiComm] Bind failed on port %d\n", port);
            close(server_fd);
            return -1;
        }
        
        if (listen(server_fd, 4) < 0) {
            fprintf(stderr, "[MultiComm] Listen failed\n");
            close(server_fd);
            return -1;
        }
        
        printf("[MultiComm] Rank 0: waiting for TCP connections on port %d...\n", port);
        
        /* Generate IDs first */
        for (int p = 0; p < nids; p++) {
            ncclGetUniqueId(&ids[p]);
        }
        
        /* Accept connection from each other rank */
        for (int r = 0; r < 1; r++) { /* only rank 1 for now */
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) {
                fprintf(stderr, "[MultiComm] Accept failed\n");
                close(server_fd);
                return -1;
            }
            
            /* Send all IDs */
            int total = sizeof(ncclUniqueId) * nids;
            int sent = 0;
            while (sent < total) {
                int n = write(client_fd, (char*)ids + sent, total - sent);
                if (n <= 0) break;
                sent += n;
            }
            
            if (sent != total) {
                fprintf(stderr, "[MultiComm] Failed to send all IDs\n");
                close(client_fd);
                close(server_fd);
                return -1;
            }
            
            printf("[MultiComm] Rank 0: sent %d IDs to rank %d\n", nids, r + 1);
            
            /* Barrier: wait for client's "ready" signal before proceeding */
            char barrier_byte = 0;
            if (read(client_fd, &barrier_byte, 1) != 1) {
                fprintf(stderr, "[MultiComm] Barrier sync failed on server\n");
                close(client_fd);
                close(server_fd);
                return -1;
            }
            
            close(client_fd);
        }
        
        close(server_fd);
        
    } else {
        /* Client: connect to rank 0 and receive IDs */
        struct hostent* host = gethostbyname(master_addr);
        if (!host) {
            fprintf(stderr, "[MultiComm] Failed to resolve master address: %s\n", master_addr);
            return -1;
        }
        
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        memcpy(&addr.sin_addr.s_addr, host->h_addr_list[0], host->h_length);
        
        /* Retry connection with timeout */
        int client_fd = -1;
        struct timespec start, now;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        while (1) {
            client_fd = socket(AF_INET, SOCK_STREAM, 0);
            if (client_fd < 0) {
                fprintf(stderr, "[MultiComm] Socket creation failed\n");
                return -1;
            }
            
            if (connect(client_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                break; /* Connected! */
            }
            
            close(client_fd);
            client_fd = -1;
            
            clock_gettime(CLOCK_MONOTONIC, &now);
            if ((now.tv_sec - start.tv_sec) > 30) {
                fprintf(stderr, "[MultiComm] Connection timeout to %s:%d\n", master_addr, port);
                return -1;
            }
            
            usleep(100000); /* 100ms retry */
        }
        
        printf("[MultiComm] Rank %d: connected to %s:%d\n", rank, master_addr, port);
        
        /* Receive all IDs */
        int total = sizeof(ncclUniqueId) * nids;
        int received = 0;
        while (received < total) {
            int n = read(client_fd, (char*)ids + received, total - received);
            if (n <= 0) break;
            received += n;
        }
        
        if (received != total) {
            fprintf(stderr, "[MultiComm] Failed to receive all IDs\n");
            close(client_fd);
            return -1;
        }
        
        printf("[MultiComm] Rank %d: received %d IDs\n", rank, nids);
        
        /* Barrier: send "ready" signal to server */
        char barrier_byte = 'R';
        if (write(client_fd, &barrier_byte, 1) != 1) {
            fprintf(stderr, "[MultiComm] Barrier sync failed on client\n");
            close(client_fd);
            return -1;
        }
        
        close(client_fd);
    }
    
    return 0;
}

/* 
 * Initialize multi-communicator setup.
 * Uses TCP to exchange NCCL unique IDs across nodes.
 * Each priority communicator sets config.trafficClass = prio_dscp[p]
 * (ToS byte; see prio_dscp table above).
 */
int multi_comm_init(int rank, int world_size, const char* device_list,
                    const char* master_addr, int port) {
    memset(&g_handle, 0, sizeof(g_handle));
    g_handle.rank = rank;
    g_handle.size = world_size;
    g_handle.current_priority = NUM_PRIORITIES - 1; /* Start at highest priority P6 */
    
    /* Parse MULTI_COMM_PRIOS (comma-separated subset of 0-6); default: all */
    const char* prios_env = getenv("MULTI_COMM_PRIOS");
    if (prios_env && prios_env[0]) {
        char buf[64];
        strncpy(buf, prios_env, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        char* tok = strtok(buf, ",");
        while (tok && g_num_active_prios < NUM_PRIORITIES) {
            g_active_prios[g_num_active_prios++] = atoi(tok);
            tok = strtok(NULL, ",");
        }
    } else {
        for (int p = 0; p < NUM_PRIORITIES; p++) {
            g_active_prios[g_num_active_prios++] = p;
        }
    }
    printf("[MultiComm] Rank %d: active priorities =", rank);
    for (int i = 0; i < g_num_active_prios; i++) {
        printf(" %d", g_active_prios[i]);
    }
    printf(" (%d total)\n", g_num_active_prios);
    
    /* Parse device list (comma-separated GPU indices) */
    char dev_copy[256];
    strncpy(dev_copy, device_list, sizeof(dev_copy) - 1);
    dev_copy[sizeof(dev_copy) - 1] = '\0';
    
    int devices[MAX_DEVICES];
    int ndev = 0;
    char* token = strtok(dev_copy, ",");
    while (token && ndev < MAX_DEVICES) {
        devices[ndev++] = atoi(token);
        token = strtok(NULL, ",");
    }
    g_handle.num_devices = ndev;
    
    printf("[MultiComm] Rank %d: initializing %d priorities across %d devices\n",
           rank, NUM_PRIORITIES, ndev);
    
    /* Exchange NCCL unique IDs via TCP */
    ncclUniqueId ids[NUM_PRIORITIES];
    if (exchange_ids_via_tcp(rank, master_addr, port, ids, g_num_active_prios) != 0) {
        fprintf(stderr, "[MultiComm] ID exchange failed\n");
        return -1;
    }
    
    /* Create communicators for each priority and device.
     * NCCL comm creation on this testbed is intermittently flaky (hard
     * "remote process exited" errors or transient hangs). Mitigations:
     *   1) small delay between priorities to reduce burst IB/proxy contention;
     *   2) retry per-priority creation up to COMM_CREATE_RETRIES on hard error.
     */
    const int COMM_CREATE_RETRIES = 3;
    for (int ai = 0; ai < g_num_active_prios; ai++) {
        int p = g_active_prios[ai];
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.trafficClass = prio_dscp[p];
        
        int created = 0;
        for (int attempt = 1; attempt <= COMM_CREATE_RETRIES; attempt++) {
            int ok = 1;
            for (int d = 0; d < ndev; d++) {
                cudaError_t ce = cudaSetDevice(devices[d]);
                if (ce != cudaSuccess) {
                    fprintf(stderr, "[MultiComm] Rank %d: cudaSetDevice(%d) failed: %s\n",
                            rank, devices[d], cudaGetErrorString(ce));
                    return -1;
                }
                
                ncclComm_t comm;
                ncclResult_t ret = ncclCommInitRankConfig(
                    &comm, world_size, ids[ai], rank, &config);
                if (ret != ncclSuccess) {
                    fprintf(stderr, "[MultiComm] Rank %d: failed to create comm for P%d device %d "
                            "(attempt %d/%d): %s\n",
                            rank, p, devices[d], attempt, COMM_CREATE_RETRIES,
                            ncclGetErrorString(ret));
                    ok = 0;
                    /* Destroy any comms created for this priority on prior devices */
                    for (int dd = 0; dd < d; dd++) {
                        if (g_handle.comms[p][dd]) {
                            ncclCommDestroy(g_handle.comms[p][dd]);
                            g_handle.comms[p][dd] = NULL;
                        }
                    }
                    break;
                }
                g_handle.comms[p][d] = comm;
            }
            if (ok) {
                created = 1;
                break;
            }
            fprintf(stderr, "[MultiComm] Rank %d: retrying P%d creation (%d/%d)\n",
                    rank, p, attempt, COMM_CREATE_RETRIES);
            usleep(2000000); /* 2s between retries */
        }
        if (!created) {
            fprintf(stderr, "[MultiComm] Rank %d: giving up on P%d after %d attempts\n",
                    rank, p, COMM_CREATE_RETRIES);
            return -1;
        }
        
        printf("[MultiComm] Rank %d: created communicator for priority P%d (DSCP=%d, ToS=%d)\n",
               rank, p, prio_dscp_val[p], prio_dscp[p]);
        
        /* Debounce between priorities to reduce burst resource contention */
        usleep(300000);
    }
    
    printf("[MultiComm] Rank %d: initialization complete, starting at P%d (DSCP=%d)\n",
           rank, g_handle.current_priority, prio_dscp_val[g_handle.current_priority]);
    
    return 0;
}

/* Switch to a different priority level */
int multi_comm_set_priority(int priority) {
    if (priority < 0 || priority >= NUM_PRIORITIES) {
        fprintf(stderr, "[MultiComm] Invalid priority %d (must be 0-%d)\n", priority, NUM_PRIORITIES - 1);
        return -1;
    }
    
    if (priority == g_handle.current_priority) return 0;
    
    printf("[MultiComm] Rank %d: switching priority P%d -> P%d (DSCP %d -> %d)\n",
           g_handle.rank, g_handle.current_priority, priority,
           prio_dscp_val[g_handle.current_priority], prio_dscp_val[priority]);
    
    g_handle.current_priority = priority;
    return 0;
}

/* Get the communicator for the current priority and device index */
ncclComm_t multi_comm_get_current(int device_idx) {
    if (device_idx < 0 || device_idx >= g_handle.num_devices) {
        fprintf(stderr, "[MultiComm] Invalid device index %d\n", device_idx);
        return NULL;
    }
    return g_handle.comms[g_handle.current_priority][device_idx];
}

/* Perform allreduce on current priority communicator */
int multi_comm_allreduce(void* sendbuff, void* recvbuff, size_t count,
                         ncclDataType_t datatype, ncclRedOp_t op, int device_idx) {
    ncclComm_t comm = multi_comm_get_current(device_idx);
    if (!comm) return -1;
    
    ncclResult_t ret = ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, NULL);
    if (ret != ncclSuccess) {
        fprintf(stderr, "[MultiComm] AllReduce failed: %s\n", ncclGetErrorString(ret));
        return -1;
    }
    /* CRITICAL (2026-08-17): ncclAllReduce 是异步的——入队后立即返回，实际传输在
     * 流上异步完成。若不在此同步等待，Python wrapper 的 _window_comm_s 只会累计到
     * 入队开销（实测 ~7ms/次，窗口累计 ~135ms），comm_ratio 恒≈1.0，
     * SLO 紧急信号永远无法触发 P6。
     * 用 cudaStreamSynchronize(NULL)（默认流）而非 cudaDeviceSynchronize()：
     * 双作业共享同一 GPU0，device 级同步会把对方作业无关的 GPU 工作计入通信时延。 */
    cudaError_t ce = cudaStreamSynchronize(NULL);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[MultiComm] AllReduce sync failed: %s\n", cudaGetErrorString(ce));
        return -1;
    }
    return 0;
}

/* Perform allgather on current priority communicator (Experiment 4) */
int multi_comm_allgather(void* sendbuff, void* recvbuff, size_t sendcount,
                         ncclDataType_t datatype, int device_idx) {
    ncclComm_t comm = multi_comm_get_current(device_idx);
    if (!comm) return -1;

    ncclResult_t ret = ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, NULL);
    if (ret != ncclSuccess) {
        fprintf(stderr, "[MultiComm] AllGather failed: %s\n", ncclGetErrorString(ret));
        return -1;
    }
    /* 与 allreduce 相同：ncclAllGather 异步，需同步等待完成，否则计时失真 */
    cudaError_t ce = cudaStreamSynchronize(NULL);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[MultiComm] AllGather sync failed: %s\n", cudaGetErrorString(ce));
        return -1;
    }
    return 0;
}

/* Cleanup */
void multi_comm_destroy(void) {
    for (int p = 0; p < NUM_PRIORITIES; p++) {
        for (int d = 0; d < g_handle.num_devices; d++) {
            if (g_handle.comms[p][d]) {
                ncclCommDestroy(g_handle.comms[p][d]);
            }
        }
    }
    
    printf("[MultiComm] Rank %d: destroyed\n", g_handle.rank);
}
