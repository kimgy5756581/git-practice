import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from league import schedule, snapshot, summarize, unpack


def game(a, b, seed, cash, valid=True):
    return {"job": {"agents": [{"id": a}, {"id": b}], "seed": seed}, "cash": cash, "valid": valid}


class LeagueTests(unittest.TestCase):
    def test_round_robin_and_challenger(self):
        agents = [{"id": x} for x in ("a", "b", "candidate")]
        self.assertEqual(len(schedule(agents, [1, 2])), 12)
        jobs = schedule(agents, [1, 2], "candidate")
        self.assertEqual(len(jobs), 8)
        self.assertTrue(all("candidate" in [a["id"] for a in j["agents"]] for j in jobs))

    def test_paired_orientation_and_invalid_exclusion(self):
        games = [game("a", "b", 1, [20, 10]), game("b", "a", 1, [10, 30]),
                 game("a", "b", 2, [99, 0]), game("b", "a", 2, [0, 99], False)]
        result = summarize(games, ["a", "b"])
        row = next(r for r in result["table"] if r["id"] == "a")
        self.assertEqual(row["mean_margin"], 15)
        self.assertEqual(row["seed_pairs"], 1)
        self.assertIsNone(row["scenario_std"])
        self.assertEqual(len(result["invalid_pairs"]), 1)

    def test_incomplete_pair_is_not_ranked(self):
        result = summarize([game("a", "b", 1, [30, 0])], ["a", "b"])
        self.assertIsNone(result["table"][0]["mean_margin"])

    def test_archive_traversal_and_links_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, link in (("../bad.py", False), ("main.py", True)):
                archive = root / "source.tar.gz"
                with tarfile.open(archive, "w:gz") as stream:
                    item = tarfile.TarInfo(name)
                    if link:
                        item.type = tarfile.SYMTYPE
                        item.linkname = "/outside"
                        stream.addfile(item)
                    else:
                        item.size = 1
                        stream.addfile(item, io.BytesIO(b"x"))
                with self.assertRaises(ValueError):
                    unpack(archive, root / "unpack")

    def test_dependency_changes_affect_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "main.py").write_text("def agent(obs): return {}")
            settings = root / "settings.json"
            settings.write_text('{"value": 1}')
            spec = {"id": "test", "path": str(root / "main.py"), "files": ["main.py", "settings.json"]}
            first = snapshot(spec, root / "first")
            settings.write_text('{"value": 2}')
            second = snapshot(spec, root / "second")
            self.assertNotEqual(first["bundle_sha256"], second["bundle_sha256"])
            self.assertEqual(json.loads((root / "first/settings.json").read_text())["value"], 1)


if __name__ == "__main__":
    unittest.main()
