"""策略对比实验：Fair vs LongLiu。"""

import sys
sys.path.insert(0, "/home/why/LongLiu_rebuild/sim-nextgen")

from longliu_sim.network import SingleLinkTopology
from longliu_sim.job import Job
from longliu_sim.policy import Fair, LongLiu
from longliu_sim.core import Simulator


MODELS = [
    ("small", 97, 0.12),
    ("medium", 156, 0.15),
    ("large", 270, 0.25),
]


def make_jobs(n, seed=42):
    import random
    random.seed(seed)
    jobs = []
    for i in range(n):
        _, mb, _ = random.choice(MODELS)
        # intv = comm * ratio, ratio 在 1.5~4.0 之间
        comm_ms = mb * 8 * 1024 * 1024 / 40e9 * 1000.0
        intv_ms = comm_ms * random.uniform(1.5, 4.0)
        # 紧 SLO 的 job 占 1/3，其余较松
        ci = 1.2 if i % 3 == 0 else (1.8 if i % 3 == 1 else 2.5)
        target = int(30000 / intv_ms * 0.7)
        jobs.append(Job(
            jid=f"J{i}",
            model="m",
            mb_per_iter=mb,
            iter_interval_ms=intv_ms,
            target_iters=max(1, target),
            slo_ci=ci,
            start_time_ms=0.0
        ))
    return jobs


def run(policy, jobs, duration_ms=30000):
    topo = SingleLinkTopology(num_hosts=2, bw_bps=40e9)
    sim = Simulator(topo, policy, duration_ms=duration_ms)
    for job in jobs:
        sim.submit(job)
    return sim.run()


def main():
    for n in [10, 20, 30]:
        jobs = make_jobs(n)
        print(f"\n=== {n} jobs, 40Gbps link, 30s ===")
        print(f"{'Policy':>10} {'Total':>6} {'AvgIter':>8} {'SLO%':>6} {'Jain':>6}")
        for P in [Fair, LongLiu]:
            result = run(P(), jobs)
            stats = result.per_job_stats()
            ratios = [s["completed_iters"] / s["target_iters"]
                      for s in stats.values()]
            jain = _jain_fairness(ratios)
            print(f"{P.__name__:>10} {result.total_iterations():>6} "
                  f"{result.avg_iteration_ms():>8.1f} "
                  f"{result.slo_attainment()*100:>6.1f} {jain:>6.3f}")


def _jain_fairness(ratios):
    if not ratios:
        return 0.0
    s1 = sum(ratios)
    s2 = sum(r * r for r in ratios)
    return s1 * s1 / (len(ratios) * s2) if s2 > 0 else 0.0


if __name__ == "__main__":
    main()
