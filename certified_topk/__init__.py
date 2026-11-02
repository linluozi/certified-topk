"""certified-topk: bounded-memory heavy hitters with a machine-checkable exactness certificate.

The sketch is Space-Saving.  Every tracked item carries an error bound; an item's
true frequency always lies in ``[count - error, count]``.  A top-k answer is only
returned as *certified* when the k-th lower bound is strictly above every rival's
upper bound (including items the sketch has already evicted).  Otherwise the
caller is told the answer is uncertified and by how much.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable


@dataclass(frozen=True)
class Estimate:
    item: Hashable
    count: int
    error: int

    @property
    def lower(self) -> int:
        return self.count - self.error

    @property
    def upper(self) -> int:
        return self.count


@dataclass(frozen=True)
class TopKResult:
    k: int
    items: tuple[Estimate, ...]
    certified: bool
    bound: int
    reason: str
    capacity: int
    seen: int
    distinct: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "certified": self.certified,
            "bound": self.bound,
            "reason": self.reason,
            "capacity": self.capacity,
            "seen": self.seen,
            "distinct": self.distinct,
            "items": [
                {"item": e.item, "count": e.count, "error": e.error,
                 "lower": e.lower, "upper": e.upper}
                for e in self.items
            ],
        }


class CertifiedTopK:
    """Space-Saving sketch with an exactness certificate.

    ``capacity`` is the number of counters, not the number of distinct items.
    Memory is O(capacity), independent of the stream length.
    """

    __slots__ = ("capacity", "counts", "errors", "_heap", "_serial",
                 "_untracked_upper", "_seen")

    def __init__(self, capacity: int = 1024) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.counts: dict[Hashable, int] = {}
        self.errors: dict[Hashable, int] = {}
        self._heap: list[tuple[int, int, Hashable]] = []
        self._serial = 0
        self._untracked_upper = 0
        self._seen = 0

    def _maybe_compact(self) -> None:
        # Lazy heap entries are bounded: at most 4x the live counters.  Without
        # this, a long stream would grow the heap without bound.
        if len(self._heap) <= 4 * self.capacity:
            return
        base = self._serial
        self._heap = [(count, base + n, item)
                      for n, (item, count) in enumerate(self.counts.items())]
        self._serial = base + len(self._heap)
        heapq.heapify(self._heap)

    # -- ingest ---------------------------------------------------------
    def add(self, item: Hashable, increment: int = 1) -> None:
        if increment < 1:
            raise ValueError("increment must be >= 1")
        self._seen += increment
        counts = self.counts
        if item in counts:
            counts[item] += increment
            self._serial += 1
            heapq.heappush(self._heap, (counts[item], self._serial, item))
            self._maybe_compact()
            return

        if len(counts) < self.capacity:
            counts[item] = increment
            self.errors[item] = 0
            self._serial += 1
            heapq.heappush(self._heap, (increment, self._serial, item))
            self._maybe_compact()
            return

        # Evict the smallest live counter.  The heap is lazy: stale entries are
        # skipped until the top entry still matches the item's current count.
        while self._heap:
            count, _, victim = self._heap[0]
            if counts.get(victim) == count:
                break
            heapq.heappop(self._heap)
        else:  # pragma: no cover - invariant failure
            raise RuntimeError("empty counter heap with a full table")

        heapq.heappop(self._heap)
        del counts[victim]
        del self.errors[victim]
        self._untracked_upper = max(self._untracked_upper, count)
        counts[item] = count + increment
        self.errors[item] = count
        self._serial += 1
        heapq.heappush(self._heap, (count + increment, self._serial, item))
        self._maybe_compact()

    def update(self, items: Iterable[Hashable]) -> None:
        for item in items:
            self.add(item)

    # -- query ----------------------------------------------------------
    def query(self, k: int = 10) -> TopKResult:
        if k < 1:
            raise ValueError("k must be >= 1")
        ranked = sorted(
            (Estimate(item, count, self.errors[item]) for item, count in self.counts.items()),
            key=lambda e: (-e.count, -e.lower, str(e.item)),
        )
        if not ranked:
            return TopKResult(k, (), True, 0, "empty stream", self.capacity, 0, 0)

        head = tuple(ranked[:k])
        if len(head) < k:
            # Fewer tracked items than requested.  This can only be certified
            # when the table never evicted anything (the universe is fully known);
            # otherwise the missing ranks cannot be named, so refuse.
            if self._untracked_upper == 0 and len(ranked) < self.capacity:
                return TopKResult(k, head, True, head[-1].lower,
                                  "table never evicted; complete universe", self.capacity,
                                  self._seen, len(ranked))
            bound = self._untracked_upper
            return TopKResult(k, head, False, bound,
                              "fewer tracked items than k and evictions occurred; "
                              "missing ranks cannot be named", self.capacity,
                              self._seen, len(ranked))

        kth = head[-1]
        rivals = ranked[k:]
        rival_upper = max((e.upper for e in rivals), default=0)
        untracked = self._untracked_upper
        bound = max(rival_upper, untracked)
        certified = all(e.lower > bound for e in head)
        if certified:
            reason = (f"k-th lower bound {kth.lower} > best rival upper bound "
                      f"{bound} (tracked rivals {rival_upper}, evicted ceiling {untracked})")
        else:
            reason = (f"k-th lower bound {kth.lower} <= best rival upper bound "
                      f"{bound}; {bound - kth.lower + 1} more observations needed "
                      f"to separate the k-th item")
        return TopKResult(k, head, certified, bound, reason, self.capacity,
                          self._seen, len(ranked))

    def estimates(self) -> list[Estimate]:
        return [Estimate(i, c, self.errors[i]) for i, c in self.counts.items()]

    def __len__(self) -> int:
        return len(self.counts)
