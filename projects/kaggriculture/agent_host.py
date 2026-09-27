"""One persistent process per agent and game; JSON-lines protocol on stdout."""
import contextlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import sys
import time
import traceback


def main():
    path = Path(sys.argv[1]).resolve()
    random.seed(int(sys.argv[3]))
    wire = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        try:
            sys.path.insert(0, str(path.parent))
            spec = importlib.util.spec_from_file_location("league_agent", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            agent = getattr(module, sys.argv[2])
            signature = inspect.signature(agent)
            two_args = True
            try:
                signature.bind({}, {})
            except TypeError:
                signature.bind({})
                two_args = False
            print(json.dumps({"ready": True}), file=wire, flush=True)
            for line in sys.stdin:
                request = json.loads(line)
                started = time.perf_counter()
                try:
                    args = [request["observation"], request["configuration"]]
                    action = agent(*(args if two_args else args[:1]))
                    response = {"action": action, "seconds": time.perf_counter() - started}
                    encoded = json.dumps(response, allow_nan=False)
                except BaseException:
                    encoded = json.dumps({"error": traceback.format_exc()})
                print(encoded, file=wire, flush=True)
        except BaseException:
            print(json.dumps({"error": traceback.format_exc()}), file=wire, flush=True)


if __name__ == "__main__":
    main()
