# certified-topk

**Bounded-memory heavy hitters that refuse to guess.** A 4,096-counter Space-Saving
sketch plus a machine-checkable exactness certificate: an answer is marked
`certified` only when the k-th item's *lower* frequency bound is strictly above every
rival's *upper* bound, including items the sketch already evicted.

Upstream: 原创 (original algorithm implementation; Space-Saving is the published
base algorithm, the certificate layer and the abort semantics are this repository's).

## What it does, with numbers

On a 1,000,000-item stream with **592,588 distinct** keys (40% power-law hot set,
60% uniform long tail), capacity 4,096, k=20:

| | this sketch | exact `collections.Counter` |
|---|---|---|
| ingest time (median of 5) | **1.183 s** | 0.089 s |
| throughput | 845,175 items/s | 11,203,906 items/s |
| peak traced memory | **2.97 MB** | 30.00 MB |
| memory ratio | **10.1× less** | 1× |
| answer | top-20 + interval per item | exact top-20 |

That is the honest trade: **10.1× less memory for 13.3× more CPU**. This is not a
speedup claim. The sketch is useful when the key space is large and you need a
guarantee, not when you need the fastest possible count.

Certificate behaviour at 200,000 items, capacity 4,096, k=20:

| Pareto α | certified | k-th lower bound | best rival upper bound |
|---|---|---|---|
| 0.6 | ✅ | 378 | 326 |
| 1.1 | ✅ | 159 | 141 |
| 1.8 | ❌ (refuses) | 33 | 38 |

The α=1.8 row is the point: when the data cannot separate rank 20 from rank 21,
the tool returns `certified: false` and a numeric gap instead of a confident-looking
list.

### Soundness sweep

5,000 random streams (1–2,000 items, 4–128 counters, k=1–9): **526 answers were
certified, 0 were wrong**. Every reported interval also contained the true count on
the 1M stream (`interval_bounds_ok_on_big_stream: true`). The sweep is in
`bench/bench.py`; the result file is `bench/result.json`.

## Install / run

Python 3.9+, no dependencies.

```bash
python3 -m pip install .
python3 -m certified_topk --capacity 4096 --k 20 events.txt
```

`events.txt` is one item per line. Exit code **0** = certified, **1** = uncertified,
**2** = usage/input error. `--json` accepts a JSON array (or JSON-lines records with
an `item` field).

```python
from certified_topk import CertifiedTopK

sketch = CertifiedTopK(capacity=4096)
for event in stream:
    sketch.add(event)

result = sketch.query(k=20)
if result.certified:
    print(result.items)          # safe to act on
else:
    print(result.reason)         # "k-th lower bound 33 <= ... 38"
```

Each `Estimate` exposes `count`, `error`, `lower` and `upper`. The invariant is
`lower <= true_count <= upper` for every tracked item; untracked items are bounded
by the eviction ceiling (`result.bound`).

## Why a certificate

Top-k sketches are usually evaluated by average recall, which hides the cases where
the answer is a coin flip. This library makes the failure mode explicit: if the
evidence does not separate the k-th item from its best rival, the caller gets
`certified: false` and the exact gap. A pipeline can then wait for more data, fall
back to exact counting, or page a human — instead of shipping the wrong top-k.

The certificate is conservative by construction: it compares a lower bound against
an upper bound, so it can return `false` when the answer happens to be right, but it
cannot return `true` when the answer is wrong (sweep: 0/526).

## Limits (measured, not hidden)

- **Not faster than `Counter`** in CPython; the per-item Python loop and lazy heap
  cost 13.3×. Use it for memory, not speed.
- Capacity is fixed at construction. A stream with more than `capacity` genuinely
  competitive items will legitimately stay uncertified.
- Ties at the k-th boundary are never certified (strict `>`), even if an arbitrary
  tie-break would happen to match the exact answer.
- The eviction ceiling is a global bound for all untracked items, not per item.

## Reproduce

```bash
python3 -m unittest discover -s tests -v     # 7 tests
python3 bench/bench.py                       # writes bench/result.json
```

The CI workflow runs the same two commands on Python 3.11 and 3.12.

## License

MIT.
