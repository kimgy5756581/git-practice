"""One game process, with two isolated agent processes."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback


class AgentClient:
    def __init__(self, spec, seed, directory, seat):
        self.errors = []
        self.calls = 0
        self.max_seconds = 0.0
        self.commands = Counter()
        self.digest = hashlib.sha256()
        self.log = (directory / f"agent_{seat}.stderr.log").open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).with_name("agent_host.py")),
             spec["path"], spec.get("entrypoint", "agent"), str(seed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding="utf-8", cwd=str(Path(spec["path"]).parent),
        )
        self.messages = queue.Queue()
        def read():
            for line in self.proc.stdout:
                self.messages.put(line)
            self.messages.put(None)
        threading.Thread(target=read, daemon=True).start()

    def receive(self, seconds):
        try:
            line = self.messages.get(timeout=seconds)
        except queue.Empty:
            raise TimeoutError("Agent response deadline exceeded")
        if line is None:
            raise RuntimeError(f"Agent exited: {self.proc.poll()}")
        result = json.loads(line)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    def ready(self):
        if not self.receive(90).get("ready"):
            raise RuntimeError("Agent did not initialize")

    def act(self, observation, configuration):
        self.calls += 1
        try:
            config = dict(configuration)
            request = {"observation": observation, "configuration": config}
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
            timeout = float(config.get("actTimeout", 1)) + float(observation.get("remainingOverageTime", 0)) + 2
            reply = self.receive(timeout)
            action = reply["action"]
            self.max_seconds = max(self.max_seconds, reply["seconds"])
            self.digest.update(json.dumps(action, sort_keys=True).encode())
            if isinstance(action, dict):
                for command in [action.get("farmer")] + list(action.get("hands", []) or []) + list(action.get("market", []) or []):
                    if isinstance(command, list) and command:
                        self.commands[str(command[0])] += 1
            return action
        except Exception:
            self.errors.append({"step": observation.get("step"), "error": traceback.format_exc()})
            raise

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    directory = Path(sys.argv[1]).resolve().parent
    result = {"job": job, "valid": False}
    clients = []
    started = time.perf_counter()
    try:
        import kaggle_environments
        from kaggle_environments import make
        from kaggle_environments.envs.kaggriculture import kaggriculture as engine
        result["engine_version"] = kaggle_environments.__version__
        result["engine_sha256"] = hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest()
        if result["engine_version"] != job["engine_version"]:
            raise RuntimeError("Installed engine version does not match manifest")
        env = make("kaggriculture", configuration={"seed": job["seed"], "episodeSteps": 720}, debug=False)
        for seat, spec in enumerate(job["agents"]):
            client = AgentClient(spec, job["seed"], directory, seat)
            clients.append(client)
            client.ready()
        # Kaggle slices positional args using co_argcount. Plain two-arg wrappers
        # avoid bound-method argument counting and keep normal timeout accounting.
        def seat0(obs, config):
            return clients[0].act(obs, config)
        def seat1(obs, config):
            return clients[1].act(obs, config)
        env.run([seat0, seat1])
        result["configuration"] = dict(env.configuration)
        result["resolved_seed"] = env.info.get("seed")
        result["states"] = len(env.steps)
        result["statuses"] = [s.status for s in env.state]
        result["rewards"] = [s.reward for s in env.state]
        result["cash"] = [f["money"] for f in env.state[0].observation["farms"]]
        result["snapshots"] = []
        seen = set()
        for step in env.steps:
            obs = step[0].observation
            day = obs.get("day")
            if day not in (0, 5, 10, 15, 20, 29) or day in seen:
                continue
            seen.add(day)
            farms = []
            for farm in obs.get("farms", []):
                counts = Counter()
                for row in farm["tiles"]:
                    for tile in row:
                        if isinstance(tile, dict):
                            for key in ("crop", "animal"):
                                if tile.get(key):
                                    counts[tile[key]] += 1
                farms.append({"cash": farm["money"], "hands": len(farm["hands"]), "counts": dict(counts)})
            result["snapshots"].append({"day": day, "farms": farms})
        logs = getattr(env, "logs", [])
        (directory / "engine_logs.json").write_text(json.dumps(logs, ensure_ascii=False, default=str), encoding="utf-8")
        def errors(value):
            if isinstance(value, dict):
                if value.get("stderr"):
                    yield value["stderr"]
                for v in value.values():
                    yield from errors(v)
            elif isinstance(value, list):
                for v in value:
                    yield from errors(v)
        result["engine_errors"] = list(errors(logs))
        result["valid"] = (result["statuses"] == ["DONE", "DONE"] and
                           result["states"] == 720 and not result["engine_errors"] and
                           all(c.calls == 719 and not c.errors for c in clients))
        if job.get("replays"):
            with gzip.open(directory / "replay.json.gz", "wt", encoding="utf-8") as stream:
                json.dump(env.toJSON(), stream, ensure_ascii=False)
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        for client in clients:
            client.close()
        result["agents"] = [{"calls": c.calls, "errors": c.errors, "max_action_seconds": c.max_seconds,
                             "action_sha256": c.digest.hexdigest(), "commands": dict(c.commands)} for c in clients]
        result["seconds"] = time.perf_counter() - started
        (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
