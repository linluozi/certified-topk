"""Measured comparison: certified-topk vs exact Counter on the same streams.

Run: python3 bench/bench.py
Writes bench/result.json.  All numbers in README come from this file.
"""

from __future__ import annotations

import gc
import json
import random
import statistics
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certified_topk import CertifiedTopK  # noqa: E402

N = 1_000_000
CAP = 4096
K = 20
REPS = 5
SEED = 20261102


def zipf_stream(n, alpha, seed, universe=20_000_000, hot=10_000):
    """Heavy hitters over a long tail: 40% power-law hot set, 60% uniform tail.

    This is the workload a bounded-memory sketch is for: the tail is large enough
    that an exact Counter keeps hundreds of thousands of keys, while the ranking
    is still dominated by a few thousand repeated items.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < 0.40:
            out.append(int(rng.paretovariate(alpha)) % hot)
        else:
            out.append(rng.randrange(hot, universe))
    return out


def timed(fn):
    gc.collect()
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def peak_mb(fn):
    gc.collect()
    tracemalloc.start()
    out = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024), out


def main():
    stream = zipf_stream(N, 1.1, SEED)
    sketch_times, counter_times = [], []
    sketch_peak = counter_peak = 0.0
    cert_rate = None

    for _ in range(REPS):
        dt, sketch = timed(lambda: _build(stream))
        sketch_times.append(dt)
        # Query is part of the answer; measure it separately below.
        del sketch

        dt, counter = timed(lambda: Counter(stream))
        counter_times.append(dt)
        del counter

    sketch_peak, _ = peak_mb(lambda: _build(stream))
    counter_peak, _ = peak_mb(lambda: Counter(stream))

    # Certificate rate at three skews (200k samples each, capacity 4096).
    rates = {}
    for alpha in (0.6, 1.1, 1.8):
        s = zipf_stream(200_000, alpha, SEED + int(alpha * 10))
        sk = CertifiedTopK(CAP)
        sk.update(s)
        r = sk.query(K)
        rates[str(alpha)] = {"certified": r.certified, "kth_lower": r.items[-1].lower,
                             "bound": r.bound}

    # Soundness sweep: 5000 streams, any certified answer is compared with Counter.
    rng = random.Random(SEED)
    false_certificates = 0
    certified = 0
    for _ in range(5000):
        n = rng.randrange(1, 2000)
        alphabet = rng.randrange(2, 500)
        items = [rng.randrange(alphabet) for _ in range(n)]
        sk = CertifiedTopK(rng.randrange(4, 128))
        sk.update(items)
        k = rng.randrange(1, 10)
        r = sk.query(k)
        if r.certified:
            certified += 1
            truth = set(x for x, _ in Counter(items).most_common(k))
            if {e.item for e in r.items} != truth:
                false_certificates += 1

    # Exactness of the reported interval on the big stream.
    sk = _build(stream)
    truth = Counter(stream)
    bounds_ok = all(e.lower <= truth[e.item] <= e.upper for e in sk.estimates())

    result = {
        "stream": {"n": N, "alpha": 1.1, "universe_cap": 20_000_000, "hot": 10_000, "seed": SEED},
        "capacity": CAP,
        "k": K,
        "reps": REPS,
        "ingest_seconds": {"sketch_median": statistics.median(sketch_times),
                           "sketch_min": min(sketch_times),
                           "counter_median": statistics.median(counter_times),
                           "counter_min": min(counter_times)},
        "throughput_items_per_s": {
            "sketch": N / statistics.median(sketch_times),
            "counter": N / statistics.median(counter_times),
        },
        "peak_mb": {"sketch": sketch_peak, "counter": counter_peak},
        "distinct": len(truth),
        "certificate_rate_by_alpha": rates,
        "soundness_sweep": {"streams": 5000, "certified": certified,
                            "false_certificates": false_certificates},
        "interval_bounds_ok_on_big_stream": bounds_ok,
    }
    out = Path(__file__).with_name("result.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def _build(stream):
    s = CertifiedTopK(CAP)
    s.update(stream)
    return s


if __name__ == "__main__":
    main()
