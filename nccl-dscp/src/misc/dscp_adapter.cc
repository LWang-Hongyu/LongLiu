/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

//----------longliu8 add----------

#include "dscp_adapter.h"
#include "comm_stats.h"
#include "comm.h"
#include "utils.h"
#include "devcomm.h"
#include "net.h"
#include "ibvwrap.h"
#include "net_ib_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// DSCP mapping: 7 levels from lowest to highest (excluding highest priority)
// [BE, AF21, AF32, AF31, AF42, AF41, AF43]
static const int DEFAULT_DSCP_MAPPING[7] = {0, 18, 28, 26, 48, 50, 52};

// Global active adapter pointer for epoch trigger functions (ncclDscpEpochStart/End)
static struct ncclDscpAdapter* g_activeAdapter = NULL;

#define DSCP_EXPORT extern "C" __attribute__((visibility("default")))

DSCP_EXPORT ncclResult_t ncclDscpAdapterInit(struct ncclDscpAdapter* adapter,
                                  double sloThreshold, 
                                  int rank) {
  if (adapter == NULL) return ncclInvalidArgument;
  
  memset(adapter, 0, sizeof(struct ncclDscpAdapter));
  
  int pthread_ret = pthread_mutex_init(&adapter->mutex, NULL);
  if (pthread_ret != 0) {
    return ncclSystemError;
  }
  
  adapter->sloThreshold = sloThreshold > 0 ? sloThreshold : 1.2;
  adapter->rank = rank;
  adapter->enabled = 1; // Enabled by default
  adapter->currentDscp = 26; // Default: AF31 (medium priority)
  adapter->numEpochs = 0;
  adapter->numPriorities = 0;
  adapter->minPriority = 0.0;
  adapter->maxPriority = 0.0;
  adapter->useDynamicMapping = 0;
  adapter->idealBandwidth = 0.0;
  adapter->firstEpochStartTime = 0.0;
  
  // Initialize EMA bandwidth fields
  adapter->emaBandwidth = 0.0;
  adapter->emaAlpha = 0.3;
  adapter->emaInitialized = 0;
  
  // Initialize epoch trigger flags
  adapter->pendingStartEpoch = -1;
  adapter->pendingEndEpoch = -1;
  
  // Initialize DSCP mapping
  memcpy(adapter->dscpMapping, DEFAULT_DSCP_MAPPING, sizeof(DEFAULT_DSCP_MAPPING));
  
  // Register as the active adapter for epoch trigger functions
  g_activeAdapter = adapter;
  
  INFO(NCCL_INIT, "DSCP Adapter initialized with SLO threshold %.1f (rank %d)", sloThreshold, rank);
  return ncclSuccess;
}

DSCP_EXPORT ncclResult_t ncclDscpAdapterDestroy(struct ncclDscpAdapter* adapter) {
  if (adapter == NULL) return ncclInvalidArgument;
  
  pthread_mutex_destroy(&adapter->mutex);
  return ncclSuccess;
}

DSCP_EXPORT ncclResult_t ncclDscpAdapterSetEnabled(struct ncclDscpAdapter* adapter, int enabled) {
  if (adapter == NULL) return ncclInvalidArgument;
  
  pthread_mutex_lock(&adapter->mutex);
  adapter->enabled = enabled ? 1 : 0;
  pthread_mutex_unlock(&adapter->mutex);
  
  return ncclSuccess;
}

// Exported C function for PyTorch to trigger epoch start via ctypes.
DSCP_EXPORT ncclResult_t ncclDscpEpochStart(int epoch) {
  if (g_activeAdapter == NULL || !g_activeAdapter->enabled) return ncclSuccess;
  pthread_mutex_lock(&g_activeAdapter->mutex);
  g_activeAdapter->pendingStartEpoch = epoch;
  pthread_mutex_unlock(&g_activeAdapter->mutex);
  return ncclSuccess;
}

// Exported C function for PyTorch to trigger epoch end via ctypes.
DSCP_EXPORT ncclResult_t ncclDscpEpochEnd(int epoch) {
  if (g_activeAdapter == NULL || !g_activeAdapter->enabled) return ncclSuccess;
  pthread_mutex_lock(&g_activeAdapter->mutex);
  g_activeAdapter->pendingEndEpoch = epoch;
  pthread_mutex_unlock(&g_activeAdapter->mutex);
  return ncclSuccess;
}

// Check and process epoch trigger flags set by ncclDscpEpochStart/End.
// Called from enqueue.cc on each NCCL op. Processes end first (to finalize
// previous epoch and update DSCP), then start (to begin the new epoch).
DSCP_EXPORT ncclResult_t ncclDscpAdapterCheckEpochTriggers(struct ncclDscpAdapter* adapter,
                                                struct ncclCommStats* stats) {
  if (adapter == NULL || !adapter->enabled) return ncclSuccess;

  pthread_mutex_lock(&adapter->mutex);
  int startEpoch = adapter->pendingStartEpoch;
  int endEpoch = adapter->pendingEndEpoch;
  adapter->pendingStartEpoch = -1;
  adapter->pendingEndEpoch = -1;
  pthread_mutex_unlock(&adapter->mutex);

  // Process end first (finalize previous epoch so its stats are available)
  if (endEpoch >= 0) {
    // IMPORTANT: Update previous epoch's Ui BEFORE EndEpoch, because
    // EndEpoch's EMA guard reads adapter->epochs[epoch-1].ui.  This
    // must already be set (otherwise Ui_prev=0.0 for all epochs).
    if (endEpoch >= 1) {
      double priority = 0.0;
      int dscp = 26;
      NCCLCHECK(ncclDscpAdapterUpdateDscpForNextEpoch(adapter, endEpoch - 1, &priority, &dscp));
    }

    int endIter = (stats != NULL && stats->numIterations > 0) ? stats->numIterations - 1 : 0;
    NCCLCHECK(ncclDscpAdapterEndEpoch(adapter, endEpoch, endIter, stats));
  }

  // Process start next (begin new epoch)
  if (startEpoch >= 0) {
    // numIterations was already incremented by ncclCommStatsStartOp
    // (called just before this in enqueue.cc).  Subtract 1 to get the
    // index of the iteration that was just allocated for the new epoch.
    int startIter = (stats != NULL && stats->numIterations > 0) ? stats->numIterations - 1 : 0;
    NCCLCHECK(ncclDscpAdapterStartEpoch(adapter, startEpoch, startIter));
  }

  return ncclSuccess;
}

DSCP_EXPORT ncclResult_t ncclDscpAdapterStartEpoch(struct ncclDscpAdapter* adapter,
                                       int epoch,
                                       int startIteration) {
  if (adapter == NULL || !adapter->enabled) return ncclSuccess;
  if (epoch < 0 || epoch >= NCCL_DSCP_MAX_EPOCHS) return ncclInvalidArgument;
  
  pthread_mutex_lock(&adapter->mutex);
  
  struct ncclEpochStats* epochStats = &adapter->epochs[epoch];
  epochStats->epoch = epoch;
  epochStats->startIteration = startIteration;
  epochStats->startTime = ncclCommStatsGetTime(); // Use NCCL monotonic clock
  epochStats->endTime = 0.0;
  epochStats->totalBytes = 0;
  epochStats->commDuration = 0.0;
  epochStats->computeDuration = 0.0;
  epochStats->numIterations = 0;
  epochStats->ui = 0.0;
  epochStats->dscp = adapter->currentDscp;
  
  // Record first epoch start time
  if (epoch == 0) {
    adapter->firstEpochStartTime = epochStats->startTime;
  }
  
  if (epoch >= adapter->numEpochs) {
    adapter->numEpochs = epoch + 1;
  }
  
  pthread_mutex_unlock(&adapter->mutex);
  
  return ncclSuccess;
}

DSCP_EXPORT ncclResult_t ncclDscpAdapterEndEpoch(struct ncclDscpAdapter* adapter,
                                      int epoch,
                                      int endIteration,
                                      struct ncclCommStats* stats) {
  if (adapter == NULL || !adapter->enabled) return ncclSuccess;
  if (epoch < 0 || epoch >= NCCL_DSCP_MAX_EPOCHS) return ncclInvalidArgument;
  if (stats == NULL) return ncclInvalidArgument;
  
  pthread_mutex_lock(&adapter->mutex);
  
  struct ncclEpochStats* epochStats = &adapter->epochs[epoch];
  epochStats->endIteration = endIteration;
  epochStats->endTime = ncclCommStatsGetTime(); // Use NCCL monotonic clock
  
  // Aggregate statistics from iterations in this epoch
  size_t totalBytes = 0;
  double totalCommDuration = 0.0;
  double minStartTime = 0.0;
  double maxEndTime = 0.0;
  int iterationCount = 0;
  int firstIter = 1;
  
  for (int iter = epochStats->startIteration; iter <= endIteration && iter < stats->numIterations; iter++) {
    struct ncclIterationStats* iterStats = &stats->iterations[iter];
    
    if (iterStats->numOps == 0) continue;
    
    totalBytes += iterStats->totalBytes;
    iterationCount++;
    
    // Calculate communication duration for this iteration
    double iterCommDuration = 0.0;
    for (int op = 0; op < iterStats->numOps; op++) {
      if (iterStats->ops[op].endTime > iterStats->ops[op].startTime) {
        iterCommDuration += (iterStats->ops[op].endTime - iterStats->ops[op].startTime);
      }
    }
    totalCommDuration += iterCommDuration;
    
    // Track min start and max end times
    if (firstIter || iterStats->startTime < minStartTime) {
      minStartTime = iterStats->startTime;
    }
    if (iterStats->endTime > maxEndTime) {
      maxEndTime = iterStats->endTime;
    }
    firstIter = 0;
  }
  
  epochStats->totalBytes = totalBytes;
  epochStats->commDuration = totalCommDuration;
  epochStats->numIterations = iterationCount;
  
  // Calculate compute duration = total wall-clock time - communication time
  double totalDuration = epochStats->endTime - epochStats->startTime;
  epochStats->computeDuration = (totalDuration > totalCommDuration) ? 
                                 (totalDuration - totalCommDuration) : 0.0;
  
  // Update numEpochs to include this completed epoch
  if (epoch >= adapter->numEpochs) {
    adapter->numEpochs = epoch + 1;
  }
  
  // Note: UI and DSCP will be saved in ncclDscpAdapterUpdateDscpForNextEpoch
  // when this epoch's statistics are used to calculate priority for the next epoch.
  // For now, initialize with default values (will be updated later)
  epochStats->ui = 0.0;
  epochStats->dscp = adapter->currentDscp;
  
  // Update EMA bandwidth for T_target online calibration.
  // Uses exponential moving average to converge smoothly to the true
  // uncontended bandwidth.
  //
  // Guard condition: EMA is updated only when the PREVIOUS epoch's urgency
  // index indicates the job was ahead of schedule (Ui < 1.0), meaning the
  // network was in a low-contention state. If the previous epoch had Ui >= 1.0,
  // the job was behind its SLO deadline — the network was contested, and the
  // current bandwidth measurement likely reflects transient congestion rather
  // than the true ideal capacity. Skipping the update prevents pollution of the
  // EMA estimate by contended-bandwidth samples.
  //
  // Epoch 0 (first epoch) always updates EMA for cold-start initialization.
  if (epochStats->commDuration > 0 && epochStats->totalBytes > 0) {
    double newBw = (epochStats->totalBytes * 8.0 / 1e9) / epochStats->commDuration;
    
    // Determine whether the network was in a low-contention state.
    // Use the previous epoch's Ui (which was already computed by
    // UpdateDscpForNextEpoch in the previous cycle).
    int lowContention = (epoch == 0) ||
        (adapter->epochs[epoch - 1].ui < 1.0);
    
    if (!adapter->emaInitialized) {
      // Cold start: seed EMA with the first measured value
      adapter->emaBandwidth = newBw;
      adapter->emaInitialized = 1;
      INFO(NCCL_ENV, "DSCP EMA [Epoch %d]: COLD_START, seed=%.2f Gbps", epoch, newBw);
    } else if (lowContention) {
      // Low contention: EMA tracks the true ideal bandwidth
      double prevEma = adapter->emaBandwidth;
      adapter->emaBandwidth = adapter->emaAlpha * newBw +
                              (1.0 - adapter->emaAlpha) * adapter->emaBandwidth;
      INFO(NCCL_ENV, "DSCP EMA [Epoch %d]: UPDATE (Ui_prev=%.4f < 1.0), "
           "newBw=%.2f, EMA: %.2f→%.2f Gbps",
           epoch, adapter->epochs[epoch - 1].ui, newBw, prevEma, adapter->emaBandwidth);
    } else {
      INFO(NCCL_ENV, "DSCP EMA [Epoch %d]: SKIP (Ui_prev=%.4f >= 1.0, contested), "
           "newBw=%.2f, EMA held=%.2f Gbps",
           epoch, adapter->epochs[epoch - 1].ui, newBw, adapter->emaBandwidth);
    }
    // else: Ui >= 1.0 in previous epoch → network was contested.
    // Skip EMA update to prevent pollution by transient congestion.
    
    // Keep idealBandwidth in sync for backward compatibility
    adapter->idealBandwidth = adapter->emaBandwidth;
  }
  
  pthread_mutex_unlock(&adapter->mutex);
  
  return ncclSuccess;
}

DSCP_EXPORT ncclResult_t ncclDscpAdapterCalculatePriority(struct ncclDscpAdapter* adapter,
                                               int epoch,
                                               double* priority) {
  if (adapter == NULL || priority == NULL) return ncclInvalidArgument;
  if (!adapter->enabled) return ncclSuccess;
  
  pthread_mutex_lock(&adapter->mutex);
  
  // Need at least 2 epochs to calculate priority
  if (adapter->numEpochs < 2 || epoch < 0 || epoch >= adapter->numEpochs) {
    pthread_mutex_unlock(&adapter->mutex);
    return ncclInvalidArgument;
  }
  
  struct ncclEpochStats* targetEpoch = &adapter->epochs[epoch];
  struct ncclEpochStats* epoch1 = &adapter->epochs[1]; // Use epoch 1 as reference
  
  // Use EMA bandwidth if available; fall back to raw epoch calculation
  double bw = adapter->emaBandwidth;
  if (bw <= 0.0 && epoch1->commDuration > 0) {
    bw = (epoch1->totalBytes * 8.0 / 1e9) / epoch1->commDuration;
  }
  if (bw <= 0.0) {
    pthread_mutex_unlock(&adapter->mutex);
    return ncclInvalidArgument;
  }
  
  // Calculate ideal communication time per epoch (using EMA bandwidth)
  double idealCommTimePerEpoch = (epoch1->totalBytes * 8.0) / (bw * 1e9);
  
  // Ideal time per epoch = compute time + ideal communication time
  double idealTimePerEpoch = epoch1->computeDuration + idealCommTimePerEpoch;
  
  // Calculate ai (actual accumulated time) and ei (expected accumulated time)
  if (adapter->firstEpochStartTime == 0.0) {
    adapter->firstEpochStartTime = adapter->epochs[0].startTime;
  }
  
  if (targetEpoch->endTime == 0.0 || adapter->firstEpochStartTime == 0.0) {
    pthread_mutex_unlock(&adapter->mutex);
    return ncclInvalidArgument;
  }
  
  double ai = targetEpoch->endTime - adapter->firstEpochStartTime;
  
  // 方案3: 单独处理Epoch 0，使用实际时间；其他epoch使用理想时间
  // ei = Epoch 0的实际时间 + (epoch编号) * 理想每epoch时间
  double ei;
  if (epoch == 0) {
    // Epoch 0: 使用实际时间
    double epoch0Duration = adapter->epochs[0].endTime - adapter->epochs[0].startTime;
    if (epoch0Duration <= 0.0) {
      epoch0Duration = adapter->epochs[0].commDuration + adapter->epochs[0].computeDuration;
    }
    ei = adapter->sloThreshold * epoch0Duration;
  } else {
    // Epoch 1及之后: 使用理想时间
    // Epoch 0的实际时间 + (epoch编号) * 理想每epoch时间
    double epoch0Duration = adapter->epochs[0].endTime - adapter->epochs[0].startTime;
    if (epoch0Duration <= 0.0) {
      epoch0Duration = adapter->epochs[0].commDuration + adapter->epochs[0].computeDuration;
    }
    ei = adapter->sloThreshold * (epoch0Duration + epoch * idealTimePerEpoch);
  }
  
  if (ei > 0) {
    *priority = ai / ei;
  } else {
    pthread_mutex_unlock(&adapter->mutex);
    return ncclInvalidArgument;
  }
  
  pthread_mutex_unlock(&adapter->mutex);
  
  return ncclSuccess;
}

DSCP_EXPORT int ncclDscpAdapterPriorityToDscp(struct ncclDscpAdapter* adapter,
                                    double priority) {
  if (adapter == NULL) return 26; // Default: AF31
  
  pthread_mutex_lock(&adapter->mutex);
  
  // Update priority history
  if (adapter->numPriorities < NCCL_DSCP_MAX_EPOCHS) {
    adapter->priorityHistory[adapter->numPriorities++] = priority;
    
    // Update min/max
    if (adapter->numPriorities == 1) {
      adapter->minPriority = priority;
      adapter->maxPriority = priority;
    } else {
      if (priority < adapter->minPriority) adapter->minPriority = priority;
      if (priority > adapter->maxPriority) adapter->maxPriority = priority;
    }
    
    // Enable dynamic mapping after 2 priorities
    if (adapter->numPriorities >= 2) {
      adapter->useDynamicMapping = 1;
    }
  }
  
  int dscp;
  
  // Use dynamic mapping if available
  if (adapter->useDynamicMapping && 
      (adapter->maxPriority - adapter->minPriority) > 0.1) {
    // Dynamic mapping: normalize priority to [0, 1] based on historical range
    double range = adapter->maxPriority - adapter->minPriority;
    double buffer = range * 0.1; // 10% buffer
    if (buffer < 0.1) buffer = 0.1;
    
    double minNorm = adapter->minPriority - buffer;
    double maxNorm = adapter->maxPriority + buffer;
    
    double normalized = (priority - minNorm) / (maxNorm - minNorm);
    if (normalized < 0.0) normalized = 0.0;
    if (normalized > 1.0) normalized = 1.0;
    
    // Map to 7 levels
    int levelIndex = (int)(normalized * 7);
    if (levelIndex > 6) levelIndex = 6;
    
    dscp = adapter->dscpMapping[levelIndex];
  } else {
    // Default fixed-threshold mapping (7 levels, excluding highest priority)
    // Values map to switch SP 6→TC6 (strict-priority) for top 3 levels
    if (priority >= 1.6) {
      dscp = adapter->dscpMapping[6]; // (52) -> SP6 strict: highest of 7 levels
    } else if (priority >= 1.4) {
      dscp = adapter->dscpMapping[5]; // (50) -> SP6 strict
    } else if (priority >= 1.2) {
      dscp = adapter->dscpMapping[4]; // (48) -> SP6 strict
    } else if (priority >= 1.0) {
      dscp = adapter->dscpMapping[3]; // (26) -> SP3 TC3 DWRR-90%
    } else if (priority >= 0.8) {
      dscp = adapter->dscpMapping[2]; // (28) -> SP3 TC3 DWRR-90%
    } else if (priority >= 0.6) {
      dscp = adapter->dscpMapping[1]; // (18) -> SP2 TC0 DWRR-10%
    } else {
      dscp = adapter->dscpMapping[0]; // (0) -> SP0 TC0 DWRR-10%
    }
  }
  
  pthread_mutex_unlock(&adapter->mutex);
  
  return dscp;
}

// Forward declarations from net_ib.cc for RDMA QP priority control
extern void ncclIbSetTc(int tc);
extern int ncclIbGetTc(void);

DSCP_EXPORT ncclResult_t ncclDscpAdapterUpdateDscpForNextEpoch(struct ncclDscpAdapter* adapter,
                                                     int currentEpoch,
                                                     double* priority,
                                                     int* dscp) {
  if (adapter == NULL || priority == NULL || dscp == NULL) return ncclInvalidArgument;
  if (!adapter->enabled) return ncclSuccess;
  
  // Calculate priority for current epoch (this function handles mutex internally)
  double calculatedPriority = 0.0;
  ncclResult_t ret = ncclDscpAdapterCalculatePriority(adapter, currentEpoch, &calculatedPriority);
  
  // Get current DSCP value (need mutex for this)
  pthread_mutex_lock(&adapter->mutex);
  int currentDscpValue = adapter->currentDscp;
  pthread_mutex_unlock(&adapter->mutex);
  
  // Even if calculation fails, we should still save the values (0.0 for UI, current DSCP)
  if (ret != ncclSuccess) {
    calculatedPriority = 0.0;
    *dscp = currentDscpValue;
  } else {
    *priority = calculatedPriority;
    *dscp = ncclDscpAdapterPriorityToDscp(adapter, calculatedPriority);
  }
  
  // Save UI and DSCP to the epoch's statistics
  pthread_mutex_lock(&adapter->mutex);
  if (currentEpoch >= 0 && currentEpoch < adapter->numEpochs) {
    struct ncclEpochStats* epochStats = &adapter->epochs[currentEpoch];
    epochStats->ui = calculatedPriority;
    epochStats->dscp = *dscp;
  }
  adapter->currentDscp = *dscp;
  pthread_mutex_unlock(&adapter->mutex);
  
  if (ret != ncclSuccess) return ncclSuccess;
  
  // Map DSCP to IB SL and TC, then update RDMA QP
  int ibSl = (*dscp * 15) / 63;
  int ibTc = *dscp << 2;
  if (ibSl > 15) ibSl = 15;
  if (ibTc > 255) ibTc = 255;
  if (ibTc < 0) ibTc = 0;
  
  // Update global TC for new QPs created after this point
  ncclIbSetTc(ibTc);
  
  // Dynamically update existing IB QPs (RDMA data channel)
  if (adapter->comm != NULL) {
    ncclResult_t qpRet = ncclDscpAdapterUpdateIbQpPriority(adapter, adapter->comm, *dscp);
    if (qpRet != ncclSuccess && adapter->rank == 0) {
      WARN("DSCP adapter: Failed to update IB QP priorities: %d", qpRet);
    }
  }
  
  // Print status (rank 0 only)
  if (adapter->rank == 0) {
    INFO(NCCL_ENV, "DSCP Adapter [Epoch %d]: Ui=%.4f, DSCP=%d, IB_SL=%d, IB_TC=%d", 
         currentEpoch, *priority, *dscp, ibSl, ibTc);
  }
  
  return ncclSuccess;
}

// Map DSCP to IB Service Level (SL) and Traffic Class (TC)
// DSCP range: 0-63
// IB_SL range: 0-15 (Service Level for routing and priority)
// IB_TC range: 0-7  (Traffic Class for QoS)
static void mapDscpToIbPriority(int dscp, int* sl, int* tc) {
  // Map DSCP to IB SL and TC for RoCEv2
  // SL: linear mapping 0-15 (used for IB, not critical for RoCE)
  *sl = (dscp * 15) / 63;
  
  // TC/traffic_class: DSCP occupies upper 6 bits of IP ToS byte
  // grh.traffic_class = DSCP << 2 (lower 2 bits = ECN, set to 0)
  *tc = dscp << 2;
  
  // Ensure values are within valid range
  if (*sl > 15) *sl = 15;
  if (*tc > 255) *tc = 255;
  if (*sl < 0) *sl = 0;
  if (*tc < 0) *tc = 0;
}

// Forward declaration of IB update function
extern ncclResult_t ncclIbUpdateQpPriority(struct ibv_qp* qp, int sl, int tc, uint8_t link_layer);

// Update IB QP priority dynamically for all QPs in the communicator
DSCP_EXPORT ncclResult_t ncclDscpAdapterUpdateIbQpPriority(struct ncclDscpAdapter* adapter,
                                                 struct ncclComm* comm,
                                                 int dscp) {
  if (adapter == NULL || comm == NULL) return ncclInvalidArgument;
  if (!adapter->enabled) return ncclSuccess;
  
  // Map DSCP to IB SL and TC
  int ibSl, ibTc;
  mapDscpToIbPriority(dscp, &ibSl, &ibTc);
  
  // Check if we're using IB transport
  if (comm->ncclNet == NULL) {
    return ncclSuccess; // No network transport
  }
  
  // Get link layer type from IB devices
  // Try to get it from the first available channel's connector
  uint8_t link_layer = IBV_LINK_LAYER_ETHERNET; // Default to RoCE (RoCE is more common)
  
  // Try to determine actual link layer from comm structure
  // The link layer is stored in ncclIbSendComm/ncclIbRecvComm's gidInfo structure
  for (int ch = 0; ch < comm->nChannels && ch < 1; ch++) {  // Check first channel only
    struct ncclChannel* channel = &comm->channels[ch];
    if (channel->peers == NULL) continue;
    
    for (int p = 0; p < comm->nRanks && p < 1; p++) {  // Check first peer only
      if (channel->peers[p] == NULL) continue;
      
      for (int conn = 0; conn < NCCL_MAX_CONNS; conn++) {
        struct ncclConnector* sendConn = &channel->peers[p]->send[conn];
        if (sendConn->connected && sendConn->transportResources != NULL) {
          // Cast to IB send comm to access gidInfo
          // Note: This assumes the transport is IB
          // The gidInfo.link_layer field contains the actual link layer type
          struct ncclIbSendComm* ibSendComm = (struct ncclIbSendComm*)sendConn->transportResources;
          if (ibSendComm != NULL) {
            link_layer = ibSendComm->gidInfo.link_layer;
            break;  // Found link layer, exit loops
          }
        }
        
        // Also check recv connector
        struct ncclConnector* recvConn = &channel->peers[p]->recv[conn];
        if (recvConn->connected && recvConn->transportResources != NULL) {
          struct ncclIbRecvComm* ibRecvComm = (struct ncclIbRecvComm*)recvConn->transportResources;
          if (ibRecvComm != NULL) {
            link_layer = ibRecvComm->gidInfo.link_layer;
            break;  // Found link layer, exit loops
          }
        }
      }
      if (link_layer != IBV_LINK_LAYER_ETHERNET) break;  // Found non-default, exit
    }
    if (link_layer != IBV_LINK_LAYER_ETHERNET) break;  // Found non-default, exit
  }
  
  int totalQpsUpdated = 0;
  int totalQpsSkipped = 0;
  
  // Iterate through all channels
  for (int ch = 0; ch < comm->nChannels; ch++) {
    struct ncclChannel* channel = &comm->channels[ch];
    
    if (channel->peers == NULL) continue;
    
    // Update send connectors
    for (int p = 0; p < comm->nRanks; p++) {
      if (channel->peers[p] == NULL) continue;
      
      for (int conn = 0; conn < NCCL_MAX_CONNS; conn++) {
        struct ncclConnector* sendConn = &channel->peers[p]->send[conn];
        if (sendConn->connected && sendConn->transportResources != NULL) {
          // Check if this is an IB send comm
          struct ncclIbSendComm* ibSendComm = (struct ncclIbSendComm*)sendConn->transportResources;
          
          // Update all QPs in this send comm
          for (int q = 0; q < ibSendComm->nqps && q < 128; q++) {
            if (ibSendComm->qps[q] != NULL) {
              ncclResult_t ret = ncclIbUpdateQpPriority(ibSendComm->qps[q], ibSl, ibTc, link_layer);
              if (ret == ncclSuccess) {
                totalQpsUpdated++;
              } else {
                totalQpsSkipped++;
              }
            }
          }
        }
        
        // Update recv connectors
        struct ncclConnector* recvConn = &channel->peers[p]->recv[conn];
        if (recvConn->connected && recvConn->transportResources != NULL) {
          // Check if this is an IB recv comm
          struct ncclIbRecvComm* ibRecvComm = (struct ncclIbRecvComm*)recvConn->transportResources;
          
          // Update all QPs in this recv comm
          for (int q = 0; q < ibRecvComm->nqps && q < 128; q++) {
            if (ibRecvComm->qps[q] != NULL) {
              ncclResult_t ret = ncclIbUpdateQpPriority(ibRecvComm->qps[q], ibSl, ibTc, link_layer);
              if (ret == ncclSuccess) {
                totalQpsUpdated++;
              } else {
                totalQpsSkipped++;
              }
            }
          }
        }
      }
    }
  }
  
  if (adapter->rank == 0 && (totalQpsUpdated > 0 || totalQpsSkipped > 0)) {
    INFO(NCCL_ENV, "DSCP adapter: Updated %d IB QPs (SL=%d, TC=%d), skipped %d", 
         totalQpsUpdated, ibSl, ibTc, totalQpsSkipped);
  }
  
  return ncclSuccess;
}
//----------longliu8 add----------