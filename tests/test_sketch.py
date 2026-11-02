import json
import random
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certified_topk import CertifiedTopK  # noqa: E402


def exact_topk(items, k):
    return set(item for item, _ in Counter(items).most_common(k))


class SketchTests(unittest.TestCase):
    def test_empty(self):
        r = CertifiedTopK(8).query(3)
        self.assertTrue(r.certified)
        self.assertEqual(r.items, ())

    def test_exact_when_capacity_covers_distinct(self):
        items = []
        for rank, name in enumerate(["a", "b", "c", "d", "e"]):
            items.extend([name] * (100 - rank * 10))
        s = CertifiedTopK(64)
        s.update(items)
        r = s.query(3)
        self.assertTrue(r.certified)
        self.assertEqual({e.item for e in r.items}, {"a", "b", "c"})
        for e in r.items:
            self.assertEqual(e.count, items.count(e.item))
            self.assertEqual(e.error, 0)

    def test_bounds_hold_for_every_tracked_item(self):
        rng = random.Random(7)
        for _ in range(200):
            items = [rng.randrange(40) for _ in range(rng.randrange(1, 800))]
            cap = rng.randrange(1, 20)
            s = CertifiedTopK(cap)
            s.update(items)
            truth = Counter(items)
            for e in s.estimates():
                self.assertLessEqual(e.lower, truth[e.item])
                self.assertLessEqual(truth[e.item], e.upper)

    def test_certificate_is_sound(self):
        """No certified answer may ever disagree with the exact top-k set."""
        rng = random.Random(20261102)
        certified = 0
        checked = 0
        for _ in range(1500):
            n = rng.randrange(1, 500)
            alphabet = rng.randrange(1, 80)
            # Mix uniform and heavily skewed streams.
            if rng.random() < 0.5:
                items = [rng.randrange(alphabet) for _ in range(n)]
            else:
                items = [int(rng.paretovariate(1.2)) % alphabet for _ in range(n)]
            k = rng.randrange(1, 8)
            s = CertifiedTopK(rng.randrange(1, 24))
            s.update(items)
            r = s.query(k)
            checked += 1
            if r.certified:
                certified += 1
                self.assertEqual({e.item for e in r.items}, exact_topk(items, k),
                                 msg=f"false certificate: {items!r} k={k} {r.as_dict()!r}")
        self.assertGreater(certified, 100)
        self.assertEqual(checked, 1500)

    def test_single_counter_still_reports(self):
        s = CertifiedTopK(1)
        s.update(["a", "a", "b", "b"])
        r = s.query(1)
        self.assertEqual(len(r.items), 1)
        self.assertFalse(r.certified)  # b's lower bound (1) does not beat a's ceiling (2)

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            CertifiedTopK(0)
        with self.assertRaises(ValueError):
            CertifiedTopK(4).add("x", 0)
        with self.assertRaises(ValueError):
            CertifiedTopK(4).query(0)

    def test_cli_exit_codes(self):
        root = Path(__file__).resolve().parents[1]
        data = "\n".join(["a"] * 20 + ["b"] * 5 + ["c"] * 2) + "\n"
        p = subprocess.run([sys.executable, "-m", "certified_topk", "--capacity", "8",
                            "--k", "2", "-"], input=data, text=True,
                           capture_output=True, cwd=root)
        self.assertEqual(p.returncode, 0, p.stderr)
        out = json.loads(p.stdout)
        self.assertTrue(out["certified"])
        self.assertEqual([e["item"] for e in out["items"]], ["a", "b"])

        p = subprocess.run([sys.executable, "-m", "certified_topk", "--k", "0", "-"],
                           input="a\n", text=True, capture_output=True, cwd=root)
        self.assertEqual(p.returncode, 2)


if __name__ == "__main__":
    unittest.main()
