"""Write cluster/workers.json with the stable URL of each deployed worker.

Run after `modal deploy cluster/modal_vllm.py`:
    python cluster/write_workers.py
"""

import json
from pathlib import Path

import modal

APP_NAME = "tutor-cluster"
WORKERS = ["WorkerA", "WorkerB"]
OUT = Path(__file__).parent / "workers.json"


def main():
    urls = {}
    for name in WORKERS:
        url = modal.Server.from_name(APP_NAME, name).get_url()
        if not url:
            raise SystemExit(f"{name}: no URL. Is {APP_NAME} deployed?")
        urls[name] = url.rstrip("/")
        print(f"{name}: {urls[name]}")
    OUT.write_text(json.dumps(urls, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
