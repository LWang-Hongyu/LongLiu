#!/usr/bin/env python3
"""
udp_flow_receiver.py — 受控双流实验：UDP 流接收端（per-second 到达率统计）

接收指定端口的 UDP 流，记录总接收字节与每 0.5s 到达速率。
大 SO_RCVBUF 避免接收端成为瓶颈；输出 CSV 供分析。

Usage:
  python3 udp_flow_receiver.py --port 6200 --duration 60 --label p6 \
      --csv out_rx_p6.csv [--payload 1472]
"""
import argparse
import csv
import socket
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--duration", type=float, required=True, help="接收时长（秒）")
    ap.add_argument("--label", default="flow")
    ap.add_argument("--payload", type=int, default=1472)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 128 * 1024 * 1024)
    sock.settimeout(1.0)
    sock.bind(("0.0.0.0", args.port))

    print(f"[RX {args.label}] :{args.port} dur={args.duration}s "
          f"rcvbuf=128MB", flush=True)

    csvf = open(args.csv, "w", newline="")
    cw = csv.writer(csvf)
    cw.writerow(["ts", "recv_bytes", "rate_gbps"])

    t0 = time.perf_counter()
    total = 0
    win_t = t0
    win_bytes = 0
    while True:
        try:
            data, _ = sock.recvfrom(65536)
        except socket.timeout:
            if time.perf_counter() - t0 >= args.duration:
                break
            continue
        if time.perf_counter() - t0 >= args.duration:
            break
        total += len(data)
        win_bytes += len(data)
        if time.perf_counter() - win_t >= 0.5:
            dt = time.perf_counter() - win_t
            cw.writerow([f"{time.perf_counter():.3f}", win_bytes,
                         f"{win_bytes * 8 / dt / 1e9:.3f}"])
            csvf.flush()
            win_t = time.perf_counter()
            win_bytes = 0

    dur = time.perf_counter() - t0
    real = total * 8 / dur / 1e9
    print(f"[RX {args.label}] DONE recv={total/1e9:.2f}GB "
          f"dur={dur:.2f}s real={real:.2f}Gbps", flush=True)
    csvf.close()


if __name__ == "__main__":
    main()
