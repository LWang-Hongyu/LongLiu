/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

//----------longliu8 add----------

#ifndef NCCL_DSCP_ADAPTER_H_
#define NCCL_DSCP_ADAPTER_H_

#include "nccl.h"
#include "comm_stats.h"
#include <stdint.h>
#include <pthread.h>

// Maximum number of epochs to track
#define NCCL_DSCP_MAX_EPOCHS 1000

// Epoch statistics for DSCP calculation
struct ncclEpochStats {
  int epoch;                      // Epoch number
  double startTime;               // Epoch start time (from NCCL monotonic clock)
  double endTime;                 // Epoch end time (from NCCL monotonic clock)
  size_t totalBytes;              // Total communication bytes in this epoch
  double commDuration;            // Total communication duration (seconds)
  double computeDuration;         // Estimated compute duration (seconds)
  int numIterations;              // Number of iterations in this epoch
  int startIteration;             // First iteration index
  int endIteration;               // Last iteration index
  double ui;                      // Urgency Index (calculated for this epoch)
  int dscp;                       // DSCP value used for this epoch
};

// Forward declaration
struct ncclComm;

// DSCP adapter structure
struct ncclDscpAdapter {
  pthread_mutex_t mutex;          // Mutex for thread safety
  double sloThreshold;            // SLO threshold (default 1.2)
  int rank;                       // Current rank
  int enabled;                    // Whether DSCP adaptation is enabled
  struct ncclComm* comm;          // Reference to comm for QP updates
  
  // Epoch statistics
  struct ncclEpochStats epochs[NCCL_DSCP_MAX_EPOCHS];
  int numEpochs;                  // Number of epochs recorded
  double firstEpochStartTime;     // First epoch start time (for ai calculation)
  
  // Ideal bandwidth (calculated from first two epochs)
  double idealBandwidth;          // Ideal bandwidth in Gbps

  // EMA (Exponential Moving Average) bandwidth for T_target online calibration
  double emaBandwidth;            // EMA of measured bandwidth in Gbps
  double emaAlpha;                // Smoothing factor for EMA (0.0-1.0, default 0.3)
  int emaInitialized;             // Whether EMA has been seeded with initial value

  // Epoch trigger flags (set by ncclDscpEpochStart/EpochEnd exported functions)
  int pendingStartEpoch;          // Epoch number to start (-1 = none pending)
  int pendingEndEpoch;            // Epoch number to end (-1 = none pending)
  
  // Priority history for dynamic mapping
  double priorityHistory[NCCL_DSCP_MAX_EPOCHS];
  int numPriorities;              // Number of priorities recorded
  double minPriority;             // Minimum priority value seen
  double maxPriority;             // Maximum priority value seen
  int useDynamicMapping;          // Whether to use dynamic mapping
  
  // Current DSCP value
  int currentDscp;                // Current DSCP value
  
  // DSCP mapping (7 levels, excluding highest priority)
  int dscpMapping[7];             // DSCP values: [0, 18, 28, 26, 36, 34, 38]
};

#ifdef __cplusplus
extern "C" {
#endif

// Initialize DSCP adapter
ncclResult_t ncclDscpAdapterInit(struct ncclDscpAdapter* adapter,
                                  double sloThreshold,
                                  int rank);

// Destroy DSCP adapter
ncclResult_t ncclDscpAdapterDestroy(struct ncclDscpAdapter* adapter);

// Start a new epoch
ncclResult_t ncclDscpAdapterStartEpoch(struct ncclDscpAdapter* adapter,
                                       int epoch,
                                       int startIteration);

// End an epoch and update statistics from comm stats
ncclResult_t ncclDscpAdapterEndEpoch(struct ncclDscpAdapter* adapter,
                                      int epoch,
                                      int endIteration,
                                      struct ncclCommStats* stats);

// Calculate priority (ui) for an epoch
ncclResult_t ncclDscpAdapterCalculatePriority(struct ncclDscpAdapter* adapter,
                                               int epoch,
                                               double* priority);

// Map priority to DSCP value
int ncclDscpAdapterPriorityToDscp(struct ncclDscpAdapter* adapter,
                                    double priority);

// Update DSCP for next epoch based on current epoch's priority
ncclResult_t ncclDscpAdapterUpdateDscpForNextEpoch(struct ncclDscpAdapter* adapter,
                                                     int currentEpoch,
                                                     double* priority,
                                                     int* dscp);

// Check and process epoch trigger flags set by exported functions.
// Called from enqueue.cc in the NCCL op path.
ncclResult_t ncclDscpAdapterCheckEpochTriggers(struct ncclDscpAdapter* adapter,
                                                struct ncclCommStats* stats);

// Update IB QP priority dynamically (for RDMA data channel)
ncclResult_t ncclDscpAdapterUpdateIbQpPriority(struct ncclDscpAdapter* adapter,
                                                 struct ncclComm* comm,
                                                 int dscp);

// Enable/disable DSCP adaptation
ncclResult_t ncclDscpAdapterSetEnabled(struct ncclDscpAdapter* adapter, int enabled);

// Exported C functions for PyTorch to trigger epoch boundaries via ctypes.
// These set pending flags on the active adapter; the actual work is done
// in ncclDscpAdapterCheckEpochTriggers called from the NCCL op path.
ncclResult_t ncclDscpEpochStart(int epoch);
ncclResult_t ncclDscpEpochEnd(int epoch);

#ifdef __cplusplus
}
#endif

#endif

//----------longliu8 add----------
