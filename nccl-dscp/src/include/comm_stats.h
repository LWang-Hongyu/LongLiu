/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

//----------longliu8 add----------

#ifndef NCCL_COMM_STATS_H_
#define NCCL_COMM_STATS_H_

#include "nccl.h"
#include "info.h"
#include <stdint.h>
#include <pthread.h>

// Maximum number of iterations to track
#define NCCL_STATS_MAX_ITERATIONS 10000
// Maximum number of operations per iteration
#define NCCL_STATS_MAX_OPS_PER_ITER 1000

// Communication operation record
struct ncclCommOpRecord {
  ncclFunc_t func;              // Operation type (AllReduce, Broadcast, etc.)
  size_t bytes;                 // Communication bytes
  double startTime;              // Start time (seconds since epoch)
  double endTime;                // End time (seconds since epoch)
  int iteration;                // Iteration number
  int rank;                     // Rank that performed this operation
};

// Iteration statistics
struct ncclIterationStats {
  int iteration;                // Iteration number
  size_t totalBytes;            // Total communication bytes in this iteration
  double startTime;             // First operation start time
  double endTime;               // Last operation end time
  int numOps;                   // Number of operations in this iteration
  struct ncclCommOpRecord ops[NCCL_STATS_MAX_OPS_PER_ITER];
};

// Communication statistics structure
struct ncclCommStats {
  pthread_mutex_t mutex;        // Mutex for thread safety
  int currentIteration;         // Current iteration number
  int numIterations;            // Number of iterations recorded
  int enabled;                  // Whether statistics collection is enabled
  struct ncclIterationStats iterations[NCCL_STATS_MAX_ITERATIONS];
  int currentOpIndex;           // Current operation index in current iteration
};

// Initialize communication statistics
ncclResult_t ncclCommStatsInit(struct ncclCommStats* stats);

// Destroy communication statistics
ncclResult_t ncclCommStatsDestroy(struct ncclCommStats* stats);

// Start recording a communication operation
ncclResult_t ncclCommStatsStartOp(struct ncclCommStats* stats, 
                                   struct ncclInfo* info, 
                                   int iteration);

// End recording a communication operation
ncclResult_t ncclCommStatsEndOp(struct ncclCommStats* stats, 
                                 struct ncclInfo* info, 
                                 int iteration);

// Mark the start of a new iteration
ncclResult_t ncclCommStatsStartIteration(struct ncclCommStats* stats, int iteration);

// Mark the end of an iteration
ncclResult_t ncclCommStatsEndIteration(struct ncclCommStats* stats, int iteration);

// Get statistics for a specific iteration
ncclResult_t ncclCommStatsGetIteration(struct ncclCommStats* stats, 
                                       int iteration,
                                       struct ncclIterationStats* out);

// Get all statistics
ncclResult_t ncclCommStatsGetAll(struct ncclCommStats* stats,
                                  struct ncclIterationStats** out,
                                  int* numIterations);

// Enable/disable statistics collection
ncclResult_t ncclCommStatsSetEnabled(struct ncclCommStats* stats, int enabled);

// Export statistics to file (JSON format)
// comm can be NULL; if provided, UI and DSCP values will be included from dscpAdapter
ncclResult_t ncclCommStatsExportToFile(struct ncclCommStats* stats, const char* filename, struct ncclComm* comm);

// Export statistics for a specific iteration range to file (JSON format)
ncclResult_t ncclCommStatsExportRangeToFile(struct ncclCommStats* stats, 
                                             const char* filename,
                                             int startIteration, 
                                             int endIteration);

// Get current time in seconds (high precision)
static inline double ncclCommStatsGetTime() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

#endif

//----------longliu8 add----------