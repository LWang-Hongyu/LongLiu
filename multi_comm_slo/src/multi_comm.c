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

/* Hardware DSCP mapping (corrected for 10.1 NIC TC config):
 * 10.1 mlnx_qos: tc:0(prio1,dscp8-15) > tc:1(prio0,dscp0-7) > tc:2(prio2,dscp16-23) >
 *                tc:3(prio3,dscp24-31) > tc:4(prio4,dscp32-39) > tc:5(prio5,dscp40-47) >
 *                tc:6(prio6,dscp48-55) > tc:7(prio7,dscp56-63)
 * Higher software priority → higher hardware TC:
 *   P6(highest) → DSCP=8 (prio1→tc:0), P4→DSCP=0 (prio0→tc:1),
 *   P3→DSCP=16 (prio2→tc:2), P2→DSCP=24 (prio3→tc:3),
 *   P1(lowest)→DSCP=32 (prio4→tc:4)
 * Previously: config.trafficClass = p * 8 (wrong — mapped P6 to tc:6, lowest) */
static const int prio_dscp[NUM_PRIORITIES] = {40, 32, 24, 16, 0, 0, 8};

/* 
 * TCP-based NCCL Unique ID exchange.
 * Rank 0 acts as server, rank 1+ as clients.
 * Exchanges NUM_PRIORITIES unique IDs.
 */
static int exchange_ids_via_tcp(int rank, const char* master_addr, int port,
                                 ncclUniqueId* ids) {
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
        for (int p = 0; p < NUM_PRIORITIES; p++) {
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
            int total = sizeof(ncclUniqueId) * NUM_PRIORITIES;
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
            
            printf("[MultiComm] Rank 0: sent %d IDs to rank %d\n", NUM_PRIORITIES, r + 1);
            
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
        int total = sizeof(ncclUniqueId) * NUM_PRIORITIES;
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
        
        printf("[MultiComm] Rank %d: received %d IDs\n", rank, NUM_PRIORITIES);
        
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
 * Each priority communicator sets trafficClass = DSCP = priority * 8.
 */
int multi_comm_init(int rank, int world_size, const char* device_list,
                    const char* master_addr, int port) {
    memset(&g_handle, 0, sizeof(g_handle));
    g_handle.rank = rank;
    g_handle.size = world_size;
    g_handle.current_priority = NUM_PRIORITIES - 1; /* Start at highest priority P6 */
    
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
    if (exchange_ids_via_tcp(rank, master_addr, port, ids) != 0) {
        fprintf(stderr, "[MultiComm] ID exchange failed\n");
        return -1;
    }
    
    /* Create communicators for each priority and device */
    for (int p = 0; p < NUM_PRIORITIES; p++) {
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.trafficClass = prio_dscp[p];
        
        for (int d = 0; d < ndev; d++) {
            cudaError_t ce = cudaSetDevice(devices[d]);
            if (ce != cudaSuccess) {
                fprintf(stderr, "[MultiComm] Rank %d: cudaSetDevice(%d) failed: %s\n",
                        rank, devices[d], cudaGetErrorString(ce));
                return -1;
            }
            
            ncclComm_t comm;
            ncclResult_t ret = ncclCommInitRankConfig(
                &comm, world_size, ids[p], rank, &config);
            if (ret != ncclSuccess) {
                fprintf(stderr, "[MultiComm] Rank %d: failed to create comm for P%d device %d: %s\n",
                        rank, p, devices[d], ncclGetErrorString(ret));
                return -1;
            }
            g_handle.comms[p][d] = comm;
        }
        
        printf("[MultiComm] Rank %d: created communicator for priority P%d (DSCP=%d)\n",
               rank, p, prio_dscp[p]);
    }
    
    printf("[MultiComm] Rank %d: initialization complete, starting at P%d (DSCP=%d)\n",
           rank, g_handle.current_priority, prio_dscp[g_handle.current_priority]);
    
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
           prio_dscp[g_handle.current_priority], prio_dscp[priority]);
    
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
