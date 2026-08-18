#!/usr/bin/env python3
"""
udp_dual_flow_sender.py — 受控双流实验：UDP 持续饱和流发送端

以指定 DSCP（ToS）持续发送 UDP 流，用于判定 SP（严格优先级）队列严格性。
与接收端 udp_flow_receiver.py 配对使用。

节流：令牌桶式忙等，按已发字节/耗时动态校准，保持平均速率等于目标速率。
IP_TOS：DSCP8→ToS32(P6/tc:0)，DSCP16→ToS64(P3/tc:2)（ToS = DSCP << 2）。

Usage:
  python3 udp_dual_flow_sender.py --dst-ip 192.10.10.226 --port 6200 \
      --dscp 8 --rate-gbps 30 --duration 60 --label p6 --csv out.csv
"""
import argparse
import csv
import os
import socket
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst-ip", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--dscp", type=int, required=True, help="DSCP 值：8(P6)/16(P3)/24")
    ap.add_argument("--rate-gbps", type=float, required=True)
    ap.add_argument("--duration", type=float, required=True, help="发送时长（秒）")
    ap.add_argument("--label", default="flow")
    ap.add_argument("--payload", type=int, default=1472)
    ap.add_argument("--csv", default=None, help="每 0.5s 发送速率 CSV")
    args = ap.parse_args()

    dst = (args.dst_ip, args.port)
    payload = os.urandom(args.payload)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, args.dscp << 2)
    # 不 connect()：未连接 UDP socket 忽略对端 ICMP port unreachable，
    # 避免 226 端口未监听时直接抛 ECONNREFUSED。

    pkt_bytes = len(payload)
    bytes_per_pkt = pkt_bytes
    rate_bps = args.rate_gbps * 1e9
    pkt_interval_s = (bytes_per_pkt * 8) / rate_bps  # 每包目标间隔（s）

    print(f"[SENDER {args.label}] -> {dst} DSCP={args.dscp} "
          f"rate={args.rate_gbps}Gbps dur={args.duration}s "
          f"pkt={pkt_bytes}B ({1e6*pkt_interval_s:.2f}us/pkt)", flush=True)

    csvf = None
    cw = None
    if args.csv:
        csvf = open(args.csv, "w", newline="")
        cw = csv.writer(csvf)
        cw.writerow(["ts", "sent_bytes", "rate_gbps"])

    # 线性节流：维护"已用时间预算"，慢于预算则 sleep，快于预算则 spin 等待
    t0 = time.perf_counter()
    sent_bytes = 0
    win_t = t0
    win_bytes = 0

    while True:
        now = time.perf_counter()
        if now - t0 >= args.duration:
            break
        sock.sendto(payload, dst)
        sent_bytes += bytes_per_pkt
        win_bytes += bytes_per_pkt

        # 期望时刻 = 已发字节数 * 每包间隔
        target_t = t0 + sent_bytes * pkt_interval_s
        wait = target_t - time.perf_counter()
        if wait > 0.0004:      # >400us 用 sleep
            time.sleep(wait)
        elif wait > 0:         # 短等待忙等
            while time.perf_counter() < target_t:
                pass

        if csvf and time.perf_counter() - win_t >= 0.5:
            dt = time.perf_counter() - win_t
            cw.writerow([f"{time.perf_counter():.3f}", win_bytes,
                         f"{win_bytes * 8 / dt / 1e9:.3f}"])
            csvf.flush()
            win_t = time.perf_counter()
            win_bytes = 0

    dur = time.perf_counter() - t0
    real = sent_bytes * 8 / dur / 1e9
    print(f"[SENDER {args.label}] DONE sent={sent_bytes/1e9:.2f}GB "
          f"dur={dur:.2f}s real={real:.2f}Gbps", flush=True)
    if csvf:
        csvf.close()


if __name__ == "__main__":
    main()
