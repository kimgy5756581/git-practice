"""Reproducible round-robin or challenger evaluation of complete agents."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return path


def unpack(source, destination):
    total = 0
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for item in archive.infolist():
                rel = safe_member(item.filename)
                if item.is_dir():
                    continue
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive links are not supported")
                total += item.file_size
                if total > 100_000_000:
                    raise ValueError("Archive exceeds 100 MB")
                target = destination / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))
    else:
        with tarfile.open(source) as archive:
            for item in archive:
                rel = safe_member(item.name)
                if item.isdir():
                    continue
                if not item.isfile():
                    raise ValueError("Only regular archive files are supported")
                total += item.size
                if total > 100_000_000:
                    raise ValueError("Archive exceeds 100 MB")
                target = destination / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(item).read())
    if not (destination / "main.py").is_file():
        raise ValueError("Archive needs main.py at root")


def snapshot(spec, destination):
    source = (HERE / spec["path"]).resolve()
    destination.mkdir(parents=True)
    if source.is_file() and source.suffix != ".py":
        unpack(source, destination)
        entry = destination / "main.py"
    else:
        if source.is_dir():
            source = source / "main.py"
        if not source.is_file():
            raise FileNotFoundError(source)
        names = spec.get("files", [source.name])
        for name in names:
            rel = safe_member(name)
            origin = (source.parent / rel).resolve()
            if not origin.is_relative_to(source.parent.resolve()):
                raise ValueError("Agent dependency escapes source folder")
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
        entry = destination / source.name
    if not entry.is_file():
        raise ValueError("Entry source was not included in files")
    hashes = {p.relative_to(destination).as_posix(): digest(p) for p in sorted(destination.rglob("*")) if p.is_file()}
    return {**spec, "path": str(entry), "source_path": str(source), "files_sha256": hashes,
            "bundle_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def schedule(agents, seeds, candidate=None):
    pairs = list(itertools.combinations(agents, 2))
    if candidate:
        pairs = [(a, b) for a, b in pairs if candidate in (a["id"], b["id"])]
    return [{"agents": list(order), "seed": seed}
            for a, b in pairs for seed in seeds for order in ((a, b), (b, a))]


def summarize(results, agent_ids):
    # A seed pair, not an individual seat, is the observation used for variance.
    groups = defaultdict(list)
    rejected = []
    for result in results:
        job = result["job"]
        a, b = [a["id"] for a in job["agents"]]
        groups[(*sorted((a, b)), job["seed"])].append(result)
    opponents = defaultdict(lambda: defaultdict(list))
    pair_rows = []
    for (a, b, seed), games in sorted(groups.items()):
        seats = {tuple(x["id"] for x in g["job"]["agents"]) for g in games}
        if len(games) != 2 or len(seats) != 2 or not all(g.get("valid") for g in games):
            rejected.append({"a": a, "b": b, "seed": seed})
            continue
        margins, points = [], []
        for game in games:
            index = 0 if game["job"]["agents"][0]["id"] == a else 1
            margin = game["cash"][index] - game["cash"][1-index]
            margins.append(margin)
            points.append(1 if margin > 0 else 0.5 if margin == 0 else 0)
        margin = statistics.mean(margins)
        point = statistics.mean(points)
        pair_rows.append({"a": a, "b": b, "seed": seed, "margin_a": margin, "points_a": point})
        opponents[a][b].append((margin, point))
        opponents[b][a].append((-margin, 1-point))
    table = []
    for agent in agent_ids:
        by_opp = opponents[agent]
        margins = [m for rows in by_opp.values() for m, _ in rows]
        means = {opp: statistics.mean(m for m, _ in rows) for opp, rows in by_opp.items()}
        table.append({"id": agent, "seed_pairs": len(margins), "opponents": len(by_opp),
                      "mean_margin": statistics.mean(means.values()) if means else None,
                      "points": statistics.mean(statistics.mean(p for _, p in rows) for rows in by_opp.values()) if by_opp else None,
                      "median_margin": statistics.median(margins) if margins else None,
                      "observed_min_margin": min(margins) if margins else None,
                      "scenario_std": statistics.stdev(margins) if len(margins) > 1 else None,
                      "worst_opponent_mean": min(means.values()) if means else None,
                      "matchups": means})
    table.sort(key=lambda row: (row["points"] if row["points"] is not None else -1,
                                row["mean_margin"] if row["mean_margin"] is not None else float("-inf")), reverse=True)
    return {"table": table, "paired_results": pair_rows, "invalid_pairs": rejected,
            "valid_games": sum(bool(r.get("valid")) for r in results), "games": len(results)}


def write_report(out, summary, mode):
    def number(v):
        return "n/a" if v is None else f"{v:,.1f}"
    lines = ["# Local Idea League", "", f"Mode: {mode}. Full manifest, seeds and hashes: `run.json`.",
             f"Valid games: {summary['valid_games']}/{summary['games']}. Rejected seed pairs: {len(summary['invalid_pairs'])}.", "",
             "Cash is local game cash, not Kaggle rating. Points = wins + half draws, averaged equally across opponents.",
             "Both seats share a seed. Standard deviation below is across paired opponent/seed scenarios, not a confidence interval.",
             "Different opponent coverage is not a fair ranking; challenger mode is not round-robin.", "",
             "| Agent | Opponents | Seed pairs | Points % | Mean margin | Median | Worst opponent mean | Scenario std |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["table"]:
        points = row["points"] * 100 if row["points"] is not None else None
        lines.append(f"| {row['id']} | {row['opponents']} | {row['seed_pairs']} | {number(points)} | {number(row['mean_margin'])} | {number(row['median_margin'])} | {number(row['worst_opponent_mean'])} | {number(row['scenario_std'])} |")
    lines += ["", "## Matchups", "", "Rows are mean cash differences after averaging both seats for each seed.", "",
              "| Agent | Opponent | Mean margin |", "| --- | --- | ---: |"]
    for row in summary["table"]:
        for opp, mean in sorted(row["matchups"].items()):
            lines.append(f"| {row['id']} | {opp} | {number(mean)} |")
    lines += ["", "Each games/<id>/ folder contains job, result, stderr and engine logs; replays are optional.",
              "Agent-internal swallowed exceptions may still return PASS. Inspect action counts and replays before interpreting low scores.",
              "This local runner calls the registered entrypoint (default agent), not Kaggle's last-callable upload discovery.",
              "Local wall-clock timing includes IPC overhead and is not a server runtime certification.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def execute(job_file, timeout, env):
    folder = job_file.parent
    with (folder / "worker.stdout.log").open("w", encoding="utf-8") as stdout, (folder / "worker.stderr.log").open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen([sys.executable, "-u", str(HERE / "worker.py"), str(job_file)], stdout=stdout, stderr=stderr, env=env)
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            else:
                proc.kill()
            proc.wait()
            returncode = -1
    result_path = folder / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {"job": json.loads(job_file.read_text(encoding="utf-8")), "valid": False,
                  "error": "Worker timed out or exited without a result"}
    result["returncode"] = returncode
    if returncode != 0:
        result["valid"] = False
    dump(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--agents", help="Comma-separated registered agent IDs")
    parser.add_argument("--candidate", type=Path, help="New idea: .py, folder, .zip or .tar.gz")
    parser.add_argument("--seed-set", choices=("smoke", "development", "holdout"), default="smoke")
    parser.add_argument("--seeds", help="Explicit comma-separated integer seeds")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--replays", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    registry = {a["id"]: a for a in manifest["agents"]}
    if args.list:
        for spec in registry.values():
            print(f"{spec['id']}: {spec['idea']}")
        return
    ids = args.agents.split(",") if args.agents else manifest["default_agents"]
    if len(ids) != len(set(ids)) or any(x not in registry for x in ids):
        parser.error("Agent IDs must be unique and registered")
    specs = [dict(registry[x]) for x in ids]
    for spec in specs:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", spec["id"]):
            parser.error("Invalid agent ID")
        spec["path"] = str((args.manifest.resolve().parent / spec["path"]).resolve())
    mode = "round-robin"
    if args.candidate:
        candidate = args.candidate.resolve()
        spec = {"id": "candidate", "path": str(candidate), "idea": "User-supplied new implementation"}
        if candidate.is_dir():
            spec["files"] = [p.relative_to(candidate).as_posix() for p in candidate.rglob("*") if p.is_file() and
                             "__pycache__" not in p.parts and ".git" not in p.parts and p.suffix not in (".pyc", ".ipynb")]
        if "candidate" in ids:
            parser.error("candidate ID is reserved")
        specs.append(spec)
        ids.append("candidate")
        mode = "challenger"
    if len(specs) < 2 or args.workers < 1:
        parser.error("Need at least two agents and one worker")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else manifest["seed_sets"][args.seed_set]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error("Seeds must be nonempty and unique")
    out = (args.out or HERE / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    if out.exists():
        parser.error("Output directory exists; use a fresh name to preserve evidence")
    out.mkdir(parents=True)
    agents = [snapshot(spec, out / "agents" / spec["id"]) for spec in specs]
    bundles = [a["bundle_sha256"] for a in agents]
    if len(set(bundles)) != len(bundles):
        print("Warning: identical source bundles in this pool; inspect run.json", flush=True)
    jobs = schedule(agents, seeds, "candidate" if mode == "challenger" else None)
    env = os.environ.copy()
    deps = ROOT / "old" / "평가기" / ".deps"
    if deps.is_dir():
        env["PYTHONPATH"] = str(deps) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    dump(out / "run.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
                           "seed_set": args.seed_set if not args.seeds else "explicit", "seeds": seeds,
                           "agents": agents, "manifest": manifest, "python": sys.version,
                           "runner_sha256": {p.name: digest(p) for p in (HERE / "league.py", HERE / "worker.py", HERE / "agent_host.py")},
                           "workers": args.workers, "timeout": args.timeout})
    paths = []
    for index, job in enumerate(jobs):
        folder = out / "games" / f"{index:04d}"
        folder.mkdir(parents=True)
        job.update({"engine_version": manifest["engine_version"], "replays": args.replays})
        dump(folder / "job.json", job)
        paths.append(folder / "job.json")
    results = []
    print(f"{mode}: {len(agents)} agents, {len(seeds)} seeds, {len(jobs)} games -> {out}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, path, args.timeout, env): path for path in paths}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            names = [s["id"] for s in result["job"]["agents"]]
            print(f"{len(results)}/{len(jobs)} {names} seed={result['job']['seed']} valid={result.get('valid')} cash={result.get('cash')} error={str(result.get('error', ''))[-160:]}", flush=True)
            summary = summarize(results, ids)
            dump(out / "summary.json", summary)
            write_report(out, summary, mode)
    if summary["valid_games"] != len(jobs):
        sys.exit(2)


if __name__ == "__main__":
    main()
