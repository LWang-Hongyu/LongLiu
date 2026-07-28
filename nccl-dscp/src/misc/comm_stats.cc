/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

//----------longliu8 add----------

#include "comm_stats.h"
#include "comm.h"
#include "dscp_adapter.h"
#include "utils.h"
#include "argcheck.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

ncclResult_t ncclCommStatsInit(struct ncclCommStats* stats) {
  if (stats == NULL) return ncclInvalidArgument;
  
  memset(stats, 0, sizeof(struct ncclCommStats));
  int pthread_ret = pthread_mutex_init(&stats->mutex, NULL);
  if (pthread_ret != 0) {
    return ncclSystemError;
  }
  stats->enabled = 1; // Enabled by default
  stats->currentIteration = -1;
  stats->numIterations = 0;
  stats->currentOpIndex = 0;
  
  return ncclSuccess;
}

ncclResult_t ncclCommStatsDestroy(struct ncclCommStats* stats) {
  if (stats == NULL) return ncclInvalidArgument;
  
  pthread_mutex_destroy(&stats->mutex);
  return ncclSuccess;
}

ncclResult_t ncclCommStatsSetEnabled(struct ncclCommStats* stats, int enabled) {
  if (stats == NULL) return ncclInvalidArgument;
  
  pthread_mutex_lock(&stats->mutex);
  stats->enabled = enabled ? 1 : 0;
  pthread_mutex_unlock(&stats->mutex);
  
  return ncclSuccess;
}

ncclResult_t ncclCommStatsStartIteration(struct ncclCommStats* stats, int iteration) {
  if (stats == NULL) return ncclInvalidArgument;
  if (!stats->enabled) return ncclSuccess;
  
  pthread_mutex_lock(&stats->mutex);
  
  if (iteration < 0 || iteration >= NCCL_STATS_MAX_ITERATIONS) {
    pthread_mutex_unlock(&stats->mutex);
    return ncclInvalidArgument;
  }
  
  stats->currentIteration = iteration;
  if (iteration >= stats->numIterations) {
    stats->numIterations = iteration + 1;
  }
  
  struct ncclIterationStats* iter = &stats->iterations[iteration];
  iter->iteration = iteration;
  iter->totalBytes = 0;
  iter->startTime = 0;
  iter->endTime = 0;
  iter->numOps = 0;
  stats->currentOpIndex = 0;
  
  pthread_mutex_unlock(&stats->mutex);
  
  return ncclSuccess;
}

ncclResult_t ncclCommStatsEndIteration(struct ncclCommStats* stats, int iteration) {
  if (stats == NULL) return ncclInvalidArgument;
  if (!stats->enabled) return ncclSuccess;
  
  pthread_mutex_lock(&stats->mutex);
  
  if (iteration < 0 || iteration >= NCCL_STATS_MAX_ITERATIONS) {
    pthread_mutex_unlock(&stats->mutex);
    return ncclInvalidArgument;
  }
  
  struct ncclIterationStats* iter = &stats->iterations[iteration];
  if (iter->numOps > 0 && iter->endTime == 0) {
    // Find the latest end time from all operations
    double maxEndTime = 0;
    for (int i = 0; i < iter->numOps; i++) {
      if (iter->ops[i].endTime > maxEndTime) {
        maxEndTime = iter->ops[i].endTime;
      }
    }
    iter->endTime = maxEndTime;
  }
  
  pthread_mutex_unlock(&stats->mutex);
  
  return ncclSuccess;
}

ncclResult_t ncclCommStatsStartOp(struct ncclCommStats* stats, 
                                   struct ncclInfo* info, 
                                   int iteration) {
  if (stats == NULL || info == NULL) return ncclInvalidArgument;
  if (!stats->enabled) return ncclSuccess;
  
  pthread_mutex_lock(&stats->mutex);
  
  if (iteration < 0 || iteration >= NCCL_STATS_MAX_ITERATIONS) {
    pthread_mutex_unlock(&stats->mutex);
    return ncclInvalidArgument;
  }
  
  // Automatically update numIterations if this is a new iteration
  // This ensures that operations are counted even if StartIteration was not called
  if (iteration >= stats->numIterations) {
    stats->numIterations = iteration + 1;
  }
  
  struct ncclIterationStats* iter = &stats->iterations[iteration];
  
  // Initialize iteration if this is the first operation
  if (iter->numOps == 0) {
    iter->iteration = iteration;
    iter->totalBytes = 0;
    iter->startTime = 0;
    iter->endTime = 0;
  }
  
  if (iter->numOps >= NCCL_STATS_MAX_OPS_PER_ITER) {
    pthread_mutex_unlock(&stats->mutex);
    return ncclInternalError; // Too many operations
  }
  
  int opIndex = iter->numOps++;
  struct ncclCommOpRecord* op = &iter->ops[opIndex];
  
  op->func = info->coll;
  op->bytes = info->nBytes;
  op->startTime = ncclCommStatsGetTime();
  op->endTime = 0;
  op->iteration = iteration;
  op->rank = info->comm ? info->comm->rank : -1;
  
  // Update iteration start time (first operation)
  if (iter->startTime == 0 || op->startTime < iter->startTime) {
    iter->startTime = op->startTime;
  }
  
  // Update total bytes
  iter->totalBytes += op->bytes;
  
  pthread_mutex_unlock(&stats->mutex);
  
  return ncclSuccess;
}

ncclResult_t ncclCommStatsEndOp(struct ncclCommStats* stats, 
                                 struct ncclInfo* info, 
                                 int iteration) {
  if (stats == NULL || info == NULL) return ncclInvalidArgument;
  if (!stats->enabled) return ncclSuccess;
  
  double endTime = ncclCommStatsGetTime();
  
  pthread_mutex_lock(&stats->mutex);
  
  if (iteration < 0 || iteration >= NCCL_STATS_MAX_ITERATIONS) {
    pthread_mutex_unlock(&stats->mutex);
    return ncclInvalidArgument;
  }
  
  // Ensure numIterations is updated (in case StartOp was called but StartIteration was not)
  if (iteration >= stats->numIterations) {
    stats->numIterations = iteration + 1;
  }
  
  struct ncclIterationStats* iter = &stats->iterations[iteration];
  
  // Find the most recent operation with matching function and bytes
  // This is a simple matching - in practice, you might want more sophisticated matching
  int opIndex = -1;
  for (int i = iter->numOps - 1; i >= 0; i--) {
    if (iter->ops[i].func == info->coll && 
        iter->ops[i].endTime == 0 &&
        iter->ops[i].bytes == info->nBytes) {
      opIndex = i;
      break;
    }
  }
  
  if (opIndex >= 0) {
    struct ncclCommOpRecord* op = &iter->ops[opIndex];
    op->endTime = endTime;
    
    // Update iteration end time
    if (iter->endTime == 0 || endTime > iter->endTime) {
      iter->endTime = endTime;
    }
  }
  
  pthread_mutex_unlock(&stats->mutex);
  
  return ncclSuccess;
}

ncclResult_t ncclCommStatsGetIteration(struct ncclCommStats* stats, 
                                       int iteration,
                                       struct ncclIterationStats* out) {
  if (stats == NULL || out == NULL) return ncclInvalidArgument;
  
  pthread_mutex_lock(&stats->mutex);
  
  if (iteration < 0 || iteration >= stats->numIterations) {
    pthread_mutex_unlock(&stats->mutex);
    return ncclInvalidArgument;
  }
  
  memcpy(out, &stats->iterations[iteration], sizeof(struct ncclIterationStats));
  
  pthread_mutex_unlock(&stats->mutex);
  
  return ncclSuccess;
}

ncclResult_t ncclCommStatsGetAll(struct ncclCommStats* stats,
                                  struct ncclIterationStats** out,
                                  int* numIterations) {
  if (stats == NULL || out == NULL || numIterations == NULL) {
    return ncclInvalidArgument;
  }
  
  pthread_mutex_lock(&stats->mutex);
  
  *numIterations = stats->numIterations;
  *out = stats->iterations;
  
  pthread_mutex_unlock(&stats->mutex);
  
  return ncclSuccess;
}

static const char* ncclFuncToString(ncclFunc_t func) {
  switch (func) {
    case ncclFuncBroadcast: return "Broadcast";
    case ncclFuncReduce: return "Reduce";
    case ncclFuncAllGather: return "AllGather";
    case ncclFuncReduceScatter: return "ReduceScatter";
    case ncclFuncAllReduce: return "AllReduce";
    case ncclFuncSend: return "Send";
    case ncclFuncRecv: return "Recv";
    default: return "Unknown";
  }
}

ncclResult_t ncclCommStatsExportToFile(struct ncclCommStats* stats, const char* filename, struct ncclComm* comm) {
  if (stats == NULL || filename == NULL) return ncclInvalidArgument;
  
  FILE* f = fopen(filename, "w");
  if (f == NULL) {
    // Try to get error information
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg), "Failed to open file %s for writing", filename);
    return ncclSystemError;
  }
  
  pthread_mutex_lock(&stats->mutex);
  
  // Determine batches per epoch dynamically
  // Priority: 1) Environment variable (set by training code), 2) Auto-detect
  int batches_per_epoch = 0;
  char* batches_env = getenv("NCCL_STATS_BATCHES_PER_EPOCH");
  if (batches_env) {
    batches_per_epoch = atoi(batches_env);
  }
  
  // If not set by environment variable, try to auto-detect
  if (batches_per_epoch <= 0 && stats->numIterations > 0) {
    // Auto-detect: try to find a reasonable batch count that divides iterations
    // Look for common batch counts (4, 8, 16, 32, 64, etc.) that make sense
    int best_batches = 0;
    int best_score = 0;
    
    // Try common batch counts
    int test_counts[] = {4, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 100, 128, 200};
    int num_tests = sizeof(test_counts) / sizeof(test_counts[0]);
    
    for (int t = 0; t < num_tests; t++) {
      int test_batches = test_counts[t];
      if (test_batches > stats->numIterations) continue;
      
      int test_epochs = (stats->numIterations + test_batches - 1) / test_batches;
      int remainder = stats->numIterations % test_batches;
      
      // Score: prefer exact division, then small remainder
      int score = 0;
      if (remainder == 0) {
        score = 10000; // Exact division is best
      } else {
        score = 10000 - remainder * 10; // Smaller remainder is better
      }
      
      // Prefer reasonable epoch counts (1-1000)
      if (test_epochs >= 1 && test_epochs <= 1000) {
        // Bonus for common epoch counts
        if (test_epochs <= 100) score += 100;
        if (test_epochs <= 50) score += 50;
        if (test_epochs <= 20) score += 50;
        
        if (score > best_score) {
          best_score = score;
          best_batches = test_batches;
        }
      }
    }
    
    // If we found a good match, use it
    if (best_batches > 0 && best_score > 5000) {
      batches_per_epoch = best_batches;
    }
    
    // If still not determined, try pattern detection
    if (batches_per_epoch <= 0) {
      // Look for patterns in the iteration data
      int first_iter = -1;
      int second_iter = -1;
      for (int i = 0; i < stats->numIterations && i < 100; i++) {
        if (stats->iterations[i].numOps > 0) {
          if (first_iter == -1) {
            first_iter = i;
          } else if (second_iter == -1) {
            second_iter = i;
            break;
          }
        }
      }
      
      // If we can detect a pattern, use it
      if (first_iter >= 0 && second_iter > first_iter) {
        int detected = second_iter - first_iter;
        if (detected > 0 && detected <= stats->numIterations) {
          int test_epochs = (stats->numIterations + detected - 1) / detected;
          if (test_epochs >= 1 && test_epochs <= 1000) {
            batches_per_epoch = detected;
          }
        }
      }
    }
    
    // Last resort: estimate based on total iterations
    if (batches_per_epoch <= 0) {
      if (stats->numIterations <= 100) {
        batches_per_epoch = stats->numIterations; // Single epoch
      } else {
        // Try to find a divisor that makes sense (powers of 2)
        for (int test = 8; test <= 256 && test < stats->numIterations; test *= 2) {
          if (stats->numIterations % test == 0) {
            batches_per_epoch = test;
            break;
          }
        }
        // If no exact divisor, use a reasonable estimate
        if (batches_per_epoch <= 0) {
          batches_per_epoch = (stats->numIterations + 9) / 10; // Rough estimate
          if (batches_per_epoch < 4) batches_per_epoch = 4;
          if (batches_per_epoch > 256) batches_per_epoch = 256;
        }
      }
    }
  }
  
  // Calculate number of epochs using multiple methods
  // Method 1: Use environment variable if set (most accurate)
  int total_epochs = 0;
  char* epochs_env = getenv("NCCL_STATS_TOTAL_EPOCHS");
  if (epochs_env) {
    total_epochs = atoi(epochs_env);
    // If we have the actual epoch count, use it directly
    if (total_epochs > 0) {
      // Recalculate batches_per_epoch based on actual epoch count
      if (stats->numIterations > 0 && total_epochs > 0) {
        batches_per_epoch = (stats->numIterations + total_epochs - 1) / total_epochs;
      }
    }
  }
  
  // Method 2: Detect epoch boundaries using timestamp gaps
  // Epoch boundaries typically have longer gaps (due to logging, model saving, etc.)
  if (total_epochs <= 0 && stats->numIterations > 1) {
    // Calculate average time between consecutive iterations
    double total_gap = 0.0;
    int gap_count = 0;
    for (int i = 1; i < stats->numIterations; i++) {
      if (stats->iterations[i].numOps > 0 && stats->iterations[i-1].numOps > 0) {
        double gap = stats->iterations[i].startTime - stats->iterations[i-1].endTime;
        if (gap > 0) {
          total_gap += gap;
          gap_count++;
        }
      }
    }
    
    double avg_gap = (gap_count > 0) ? (total_gap / gap_count) : 0.0;
    // Epoch boundaries typically have gaps > 3x average gap
    double epoch_boundary_threshold = avg_gap * 3.0;
    // Minimum threshold: 0.1 seconds (epoch boundaries are usually > 100ms)
    if (epoch_boundary_threshold < 0.1) {
      epoch_boundary_threshold = 0.1;
    }
    
    // Count epoch boundaries
    int detected_epochs = 1; // Start with 1 epoch
    for (int i = 1; i < stats->numIterations; i++) {
      if (stats->iterations[i].numOps > 0 && stats->iterations[i-1].numOps > 0) {
        double gap = stats->iterations[i].startTime - stats->iterations[i-1].endTime;
        if (gap > epoch_boundary_threshold) {
          detected_epochs++;
        }
      }
    }
    
    if (detected_epochs > 0 && detected_epochs <= 1000) {
      total_epochs = detected_epochs;
      // Recalculate batches_per_epoch
      if (stats->numIterations > 0) {
        batches_per_epoch = (stats->numIterations + total_epochs - 1) / total_epochs;
      }
    }
  }
  
  // Method 3: Fallback to calculation based on batches_per_epoch
  if (total_epochs <= 0) {
    if (batches_per_epoch > 0 && stats->numIterations > 0) {
      total_epochs = (stats->numIterations + batches_per_epoch - 1) / batches_per_epoch;
    } else if (stats->numIterations > 0) {
      // Last resort: treat all iterations as one epoch
      total_epochs = 1;
      batches_per_epoch = stats->numIterations;
    }
  }
  
  fprintf(f, "{\n");
  fprintf(f, "  \"version\": \"1.0\",\n");
  fprintf(f, "  \"enabled\": %s,\n", stats->enabled ? "true" : "false");
  fprintf(f, "  \"total_iterations\": %d,\n", stats->numIterations);
  fprintf(f, "  \"batches_per_epoch\": %d,\n", batches_per_epoch);
  fprintf(f, "  \"total_epochs\": %d,\n", total_epochs);
  fprintf(f, "  \"epochs\": [\n");
  
  // Detect epoch boundaries using timestamp gaps (if not using fixed batches_per_epoch)
  int* epoch_boundaries = NULL;
  int num_boundaries = 0;
  
  // If we detected epochs using timestamps, find the actual boundaries
  if (total_epochs > 1 && stats->numIterations > 1) {
    // Calculate average time between consecutive iterations
    double total_gap = 0.0;
    int gap_count = 0;
    for (int i = 1; i < stats->numIterations; i++) {
      if (stats->iterations[i].numOps > 0 && stats->iterations[i-1].numOps > 0) {
        double gap = stats->iterations[i].startTime - stats->iterations[i-1].endTime;
        if (gap > 0) {
          total_gap += gap;
          gap_count++;
        }
      }
    }
    
    double avg_gap = (gap_count > 0) ? (total_gap / gap_count) : 0.0;
    double epoch_boundary_threshold = avg_gap * 3.0;
    if (epoch_boundary_threshold < 0.1) {
      epoch_boundary_threshold = 0.1;
    }
    
    // Allocate array for epoch boundaries
    epoch_boundaries = (int*)malloc((total_epochs + 1) * sizeof(int));
    if (epoch_boundaries) {
      epoch_boundaries[0] = 0; // First epoch starts at iteration 0
      num_boundaries = 1;
      
      // Find epoch boundaries
      for (int i = 1; i < stats->numIterations; i++) {
        if (stats->iterations[i].numOps > 0 && stats->iterations[i-1].numOps > 0) {
          double gap = stats->iterations[i].startTime - stats->iterations[i-1].endTime;
          if (gap > epoch_boundary_threshold && num_boundaries < total_epochs) {
            epoch_boundaries[num_boundaries++] = i;
          }
        }
      }
      epoch_boundaries[num_boundaries] = stats->numIterations; // Last boundary
      
      // If timestamp detection failed (only found 1 boundary), don't use it
      // Fall back to fixed batch calculation
      if (num_boundaries <= 1) {
        free(epoch_boundaries);
        epoch_boundaries = NULL;
        num_boundaries = 0;
      }
    }
  }
  
  // Group iterations by epoch
  // Always prefer fixed batch calculation if batches_per_epoch is valid
  // Timestamp detection is only used as fallback when batches_per_epoch is unknown
  for (int epoch = 0; epoch < total_epochs; epoch++) {
    int start_iter, end_iter;
    
    // Prefer fixed batch calculation (more reliable when batches_per_epoch is known)
    if (batches_per_epoch > 0) {
      start_iter = epoch * batches_per_epoch;
      end_iter = (epoch + 1) * batches_per_epoch - 1;
      if (end_iter >= stats->numIterations) {
        end_iter = stats->numIterations - 1;
      }
    } else if (epoch_boundaries && num_boundaries > epoch) {
      // Use detected boundaries only if batches_per_epoch is not available
      start_iter = epoch_boundaries[epoch];
      end_iter = epoch_boundaries[epoch + 1] - 1;
      if (end_iter >= stats->numIterations) {
        end_iter = stats->numIterations - 1;
      }
    } else {
      // Last resort: equal distribution
      start_iter = (epoch * stats->numIterations) / total_epochs;
      end_iter = ((epoch + 1) * stats->numIterations) / total_epochs - 1;
      if (end_iter >= stats->numIterations) {
        end_iter = stats->numIterations - 1;
      }
    }
    
    // Count iterations in this epoch
    int epoch_iter_count = 0;
    for (int i = start_iter; i <= end_iter && i < stats->numIterations; i++) {
      if (stats->iterations[i].numOps > 0) {
        epoch_iter_count++;
      }
    }
    
    // Get UI and DSCP values from dscpAdapter if available
    double ui_value = 0.0;
    int dscp_value = 0;
    if (comm != NULL && comm->dscpAdapter.enabled && epoch < comm->dscpAdapter.numEpochs) {
      pthread_mutex_lock(&comm->dscpAdapter.mutex);
      struct ncclEpochStats* epochStats = &comm->dscpAdapter.epochs[epoch];
      ui_value = epochStats->ui;
      dscp_value = epochStats->dscp;
      pthread_mutex_unlock(&comm->dscpAdapter.mutex);
    }
    
    fprintf(f, "    {\n");
    fprintf(f, "      \"epoch\": %d,\n", epoch);
    fprintf(f, "      \"start_iteration\": %d,\n", start_iter);
    fprintf(f, "      \"end_iteration\": %d,\n", end_iter);
    fprintf(f, "      \"num_iterations\": %d,\n", epoch_iter_count);
    fprintf(f, "      \"ui\": %.6f,\n", ui_value);
    fprintf(f, "      \"dscp\": %d,\n", dscp_value);
    fprintf(f, "      \"iterations\": [\n");
    
    int written = 0;
    for (int i = start_iter; i <= end_iter && i < stats->numIterations; i++) {
      struct ncclIterationStats* iter = &stats->iterations[i];
      
      // Skip empty iterations
      if (iter->numOps == 0) continue;
      
      fprintf(f, "        {\n");
      fprintf(f, "          \"iteration\": %d,\n", iter->iteration);
      fprintf(f, "          \"total_bytes\": %zu,\n", iter->totalBytes);
      fprintf(f, "          \"start_time\": %.9f,\n", iter->startTime);
      fprintf(f, "          \"end_time\": %.9f,\n", iter->endTime);
      fprintf(f, "          \"duration\": %.9f,\n", 
              iter->endTime > iter->startTime ? iter->endTime - iter->startTime : 0.0);
      fprintf(f, "          \"num_ops\": %d,\n", iter->numOps);
      fprintf(f, "          \"operations\": [\n");
      
      // Export all operations in this iteration
      for (int j = 0; j < iter->numOps; j++) {
        struct ncclCommOpRecord* op = &iter->ops[j];
        fprintf(f, "            {\n");
        fprintf(f, "              \"func\": \"%s\",\n", ncclFuncToString(op->func));
        fprintf(f, "              \"bytes\": %zu,\n", op->bytes);
        fprintf(f, "              \"start_time\": %.9f,\n", op->startTime);
        fprintf(f, "              \"end_time\": %.9f,\n", op->endTime);
        fprintf(f, "              \"duration\": %.9f,\n", 
                op->endTime > op->startTime ? op->endTime - op->startTime : 0.0);
        fprintf(f, "              \"rank\": %d,\n", op->rank);
        fprintf(f, "              \"iteration\": %d\n", op->iteration);
        fprintf(f, "            }%s\n", j < iter->numOps - 1 ? "," : "");
      }
      
      fprintf(f, "          ]\n");
      fprintf(f, "        }%s\n", (++written < epoch_iter_count) ? "," : "");
    }
    
    fprintf(f, "      ]\n");
    fprintf(f, "    }%s\n", epoch < total_epochs - 1 ? "," : "");
  }
  
  fprintf(f, "  ]\n");
  fprintf(f, "}\n");
  
  // Free allocated memory
  if (epoch_boundaries) {
    free(epoch_boundaries);
  }
  
  pthread_mutex_unlock(&stats->mutex);
  
  fclose(f);
  
  return ncclSuccess;
}

ncclResult_t ncclCommStatsExportRangeToFile(struct ncclCommStats* stats, 
                                             const char* filename,
                                             int startIteration, 
                                             int endIteration) {
  if (stats == NULL || filename == NULL) return ncclInvalidArgument;
  if (startIteration < 0 || endIteration < startIteration) return ncclInvalidArgument;
  
  FILE* f = fopen(filename, "w");
  if (f == NULL) {
    return ncclSystemError;
  }
  
  pthread_mutex_lock(&stats->mutex);
  
  // Count iterations in range
  int countInRange = 0;
  for (int i = startIteration; i <= endIteration && i < stats->numIterations; i++) {
    if (stats->iterations[i].numOps > 0) {
      countInRange++;
    }
  }
  
  fprintf(f, "{\n");
  fprintf(f, "  \"version\": \"1.0\",\n");
  fprintf(f, "  \"enabled\": %s,\n", stats->enabled ? "true" : "false");
  fprintf(f, "  \"start_iteration\": %d,\n", startIteration);
  fprintf(f, "  \"end_iteration\": %d,\n", endIteration);
  fprintf(f, "  \"num_iterations\": %d,\n", countInRange);
  fprintf(f, "  \"iterations\": [\n");
  
  int written = 0;
  // Export iterations in range
  for (int i = startIteration; i <= endIteration && i < stats->numIterations; i++) {
    struct ncclIterationStats* iter = &stats->iterations[i];
    
    // Skip empty iterations
    if (iter->numOps == 0) continue;
    
    fprintf(f, "    {\n");
    fprintf(f, "      \"iteration\": %d,\n", iter->iteration);
    fprintf(f, "      \"total_bytes\": %zu,\n", iter->totalBytes);
    fprintf(f, "      \"start_time\": %.9f,\n", iter->startTime);
    fprintf(f, "      \"end_time\": %.9f,\n", iter->endTime);
    fprintf(f, "      \"duration\": %.9f,\n", 
            iter->endTime > iter->startTime ? iter->endTime - iter->startTime : 0.0);
    fprintf(f, "      \"num_ops\": %d,\n", iter->numOps);
    fprintf(f, "      \"operations\": [\n");
    
    // Export all operations in this iteration
    for (int j = 0; j < iter->numOps; j++) {
      struct ncclCommOpRecord* op = &iter->ops[j];
      fprintf(f, "        {\n");
      fprintf(f, "          \"func\": \"%s\",\n", ncclFuncToString(op->func));
      fprintf(f, "          \"bytes\": %zu,\n", op->bytes);
      fprintf(f, "          \"start_time\": %.9f,\n", op->startTime);
      fprintf(f, "          \"end_time\": %.9f,\n", op->endTime);
      fprintf(f, "          \"duration\": %.9f,\n", 
              op->endTime > op->startTime ? op->endTime - op->startTime : 0.0);
      fprintf(f, "          \"rank\": %d,\n", op->rank);
      fprintf(f, "          \"iteration\": %d\n", op->iteration);
      fprintf(f, "        }%s\n", j < iter->numOps - 1 ? "," : "");
    }
    
    fprintf(f, "      ]\n");
    fprintf(f, "    }%s\n", (++written < countInRange) ? "," : "");
  }
  
  fprintf(f, "  ]\n");
  fprintf(f, "}\n");
  
  pthread_mutex_unlock(&stats->mutex);
  
  fclose(f);
  
  return ncclSuccess;
}

//----------longliu8 add----------