/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

//----------longliu8 add----------

#include "comm_stats.h"
#include "comm.h"

// Initialize communication statistics for a communicator
ncclResult_t ncclCommStatsInitForComm(struct ncclComm* comm) {
  if (comm == NULL) return ncclInvalidArgument;
  return ncclCommStatsInit(&comm->commStats);
}

// Destroy communication statistics for a communicator
ncclResult_t ncclCommStatsDestroyForComm(struct ncclComm* comm) {
  if (comm == NULL) return ncclInvalidArgument;
  return ncclCommStatsDestroy(&comm->commStats);
}

//----------longliu8 add----------