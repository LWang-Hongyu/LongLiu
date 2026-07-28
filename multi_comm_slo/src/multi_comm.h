#ifndef MULTI_COMM_H
#define MULTI_COMM_H

#include "nccl.h"

#define NUM_PRIORITIES 7
#define MAX_DEVICES 8

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize multi-communicator setup.
 * rank: this process's rank
 * world_size: total number of ranks
 * device_list: comma-separated GPU device indices (e.g. "0" or "0,1")
 * master_addr: IP address of rank 0 for TCP ID exchange
 * port: TCP port for ID exchange
 * Returns 0 on success, -1 on failure.
 */
int multi_comm_init(int rank, int world_size, const char* device_list,
                    const char* master_addr, int port);

/* Switch to a different priority level (0-6).
 * P0 = lowest priority (DSCP=0), P6 = highest priority (DSCP=48).
 * Returns 0 on success, -1 on failure.
 */
int multi_comm_set_priority(int priority);

/* Get the communicator for the current priority and device index. */
ncclComm_t multi_comm_get_current(int device_idx);

/* Perform allreduce on current priority communicator.
 * Returns 0 on success, -1 on failure.
 */
int multi_comm_allreduce(void* sendbuff, void* recvbuff, size_t count,
                         ncclDataType_t datatype, ncclRedOp_t op, int device_idx);

/* Destroy all communicators and clean up. */
void multi_comm_destroy(void);

#ifdef __cplusplus
}
#endif

#endif /* MULTI_COMM_H */
