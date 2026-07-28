/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Shared IB transport type definitions.
 * Included by both net_ib.cc and dscp_adapter.cc to ensure struct
 * layout consistency.
 ************************************************************************/

#ifndef NCCL_NET_IB_TYPES_H_
#define NCCL_NET_IB_TYPES_H_

#include "nccl.h"
#include "net.h"
#include "socket.h"
#include "core.h"
#include "ibvwrap.h"

/* ---- Defines ---- */
#ifndef NCCL_NET_IB_MAX_RECVS
#define NCCL_NET_IB_MAX_RECVS 8
#endif

#ifndef NCCL_IB_MAX_QPS
#define NCCL_IB_MAX_QPS 128
#endif

#ifndef MAX_REQUESTS
#define MAX_REQUESTS (NCCL_NET_MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS)
#endif

/* ---- Structs ---- */

struct ncclIbGidInfo {
  uint8_t link_layer;
  union ibv_gid localGid;
  union ibv_gid remoteGid;
};

struct ncclIbRequest {
  struct ncclIbVerbs* verbs;
  int type;
  int events;
  struct ncclSocket* sock;
  struct ncclIbGidInfo* gidInfo;
  int nreqs;
  union {
    struct {
      int size;
      void* data;
      uint32_t lkey;
      int offset;
    } send;
    struct {
      int sizes[NCCL_NET_IB_MAX_RECVS];
    } recv;
  };
};

struct ncclIbVerbs {
  int dev;
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  uint64_t pad[1];
  struct ncclIbRequest reqs[MAX_REQUESTS];
};

struct ncclIbSendFifo {
  uint64_t addr;
  int      size;
  uint32_t rkey;
  uint32_t nreqs;
  uint32_t tag;
  uint64_t idx;
};

struct ncclIbSendComm {
  struct ncclIbVerbs verbs;
  struct ncclIbSendFifo fifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t fifoHead;
  struct ncclIbRequest* fifoReqs[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS+1];
  struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];
  struct ncclSocket sock;

  int ready;
  struct ibv_qp* qps[NCCL_IB_MAX_QPS];
  int nqps;
  int qpIndex;
  struct ibv_mr* fifoMr;
  int ar;
  struct ncclIbGidInfo gidInfo;
};

struct ncclIbGpuFlush {
  int enabled;
  int hostMem;
  struct ibv_mr* hostMr;
  struct ibv_sge sge;
  struct ibv_qp* qp;
};

struct ncclIbRemFifo {
  struct ncclIbSendFifo elems[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t fifoTail;
  uint64_t addr;
  uint32_t rkey;
  uint32_t flags;
  struct ibv_mr* mr;
  struct ibv_sge sge;
};

struct ncclIbRecvComm {
  struct ncclIbVerbs verbs;
  struct ncclIbRemFifo remFifo;
  struct ncclSocket sock;
  int ready;
  struct ibv_qp* qps[NCCL_IB_MAX_QPS];
  int nqps;
  int qpIndex;
  struct ncclIbGpuFlush gpuFlush;
  struct ncclIbGidInfo gidInfo;
};

#endif /* NCCL_NET_IB_TYPES_H_ */