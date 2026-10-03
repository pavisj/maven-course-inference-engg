"""Write the stable URLs of the deployed servers.

Run after `modal deploy cluster/modal_vllm.py`:
    python cluster/write_workers.py

- cluster/workers.json   WorkerA and WorkerB: the cluster (scraped every few seconds)
- cluster/overflow.json  Overflow: the leave target. Kept apart so nothing scrapes it,
                         because any request keeps a scale-from-zero server awake.
"""

import json
from pathlib import Path

import modal

APP_NAME = "tutor-cluster"
WORKERS = ["WorkerA", "WorkerB"]
OVERFLOW = ["Overflow"]
HERE = Path(__file__).parent


def urls_for(names: list[str], required: bool) -> dict[str, str]:
    urls = {}
    for name in names:
        try:
            url = modal.Server.from_name(APP_NAME, name).get_url()
        except modal.exception.NotFoundError:
            url = None
        if not url:
            if required:
                raise SystemExit(f"{name}: no URL. Is {APP_NAME} deployed?")
            print(f"{name}: not deployed")
            continue
        urls[name] = url.rstrip("/")
        print(f"{name}: {urls[name]}")
    return urls


def main():
    for names, out, required in [(WORKERS, HERE / "workers.json", True), (OVERFLOW, HERE / "overflow.json", False)]:
        urls = urls_for(names, required)
        out.write_text(json.dumps(urls, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
