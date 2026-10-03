"""Cold-start one worker: stop its container, wait for the fresh one.

    python cluster/cold_start.py --worker WorkerB

A `modal deploy` restarts every server in the app. Stopping one worker's container restarts
only that worker: min_containers=1 makes Modal start a fresh container with an empty KV
cache, no compiled kernels in memory and no CUDA graphs captured. Prints the seconds until
/health returns 200 again.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import modal

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "tutor-cluster"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", default="WorkerB")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()

    url = json.loads((ROOT / "cluster" / "workers.json").read_text())[args.worker]
    containers = modal.Server.from_name(APP_NAME, args.worker)._experimental_list_containers()
    ids = [c.container_id for c in containers]
    if not ids:
        sys.exit(f"{args.worker}: no running container")
    t0 = time.time()
    for cid in ids:
        print(f"stopping {args.worker} container {cid}")
        subprocess.run([sys.executable, "-m", "modal", "container", "stop", "--yes", cid], check=True)
    seen_down = False
    while time.time() - t0 < args.timeout:
        try:
            code = httpx.get(f"{url}/health", timeout=30).status_code
        except httpx.HTTPError:
            code = None
        seen_down = seen_down or code != 200
        if code == 200 and seen_down:
            new = [c.container_id for c in modal.Server.from_name(APP_NAME, args.worker)._experimental_list_containers()]
            print(json.dumps({"worker": args.worker, "stopped": ids, "new_containers": new,
                              "healthy_after_s": round(time.time() - t0, 1)}))
            return
        time.sleep(3)
    sys.exit(f"{args.worker}: not healthy after {args.timeout:.0f} s")


if __name__ == "__main__":
    main()
