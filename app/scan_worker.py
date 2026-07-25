"""Standalone entry point for running one poll cycle in an isolated OS process.

Runs as `python scan_worker.py`, reading a JSON args object from stdin and
writing newline-delimited JSON progress/result messages to stdout. Logging
goes to stderr so it doesn't interleave with the stdout protocol. Loading
YOLO/CLIP here means the models, and all the RAM/VRAM they hold, are released
to the OS the moment this process exits, instead of lingering in the
long-lived web server process for the life of the container.
"""

import json
import logging
import os
import sys
import threading

# Some dependencies (notably ultralytics, on first-ever model load) print
# plain text straight to stdout instead of going through logging. That would
# corrupt our newline-delimited JSON protocol, so grab a private duplicate of
# the real stdout fd for our own messages, then point the process's stdout at
# stderr so any stray library output lands there instead.
_protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)


def _emit(payload: dict) -> None:
    _protocol_out.write(json.dumps(payload) + "\n")
    _protocol_out.flush()


def main() -> int:
    args = json.loads(sys.stdin.read())

    from pathlib import Path
    from embedder import load_embed_cache
    from poller import run_poll_cycle, migrate_ref_bboxes

    data_dir = args["data_dir"]
    load_embed_cache(Path(data_dir))
    if args.get("migrate"):
        migrate_ref_bboxes(Path(data_dir))

    low_conf_out: list = []
    live_counts: dict = {}

    def on_date(date_str):
        _emit({"type": "date", "date": date_str})

    def report_counts():
        while not stop_reporting.wait(1.0):
            _emit({"type": "counts", "counts": dict(live_counts)})

    stop_reporting = threading.Event()
    reporter = threading.Thread(target=report_counts, daemon=True)
    reporter.start()

    try:
        run_poll_cycle(
            data_dir,
            on_date=on_date,
            low_conf_out=low_conf_out,
            live_counts=live_counts,
            manual=args.get("manual", False),
            scan_until=args.get("scan_until"),
            scan_since=args.get("scan_since"),
        )
    except Exception as e:
        _emit({"type": "error", "error": str(e)})
        return 1
    finally:
        stop_reporting.set()
        reporter.join(timeout=5)

    _emit({"type": "done", "counts": live_counts, "low_conf": low_conf_out})
    return 0


if __name__ == "__main__":
    code = main()
    # PyTorch/CUDA can abort (SIGABRT) during normal interpreter shutdown while
    # tearing down CUDA context/stream threads. This process's only job was to
    # run once and exit, so skip that teardown entirely: os._exit() drops
    # straight to the OS syscall, no atexit handlers, no destructors, nothing
    # left for the CUDA runtime to race against.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
