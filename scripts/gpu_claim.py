#!/usr/bin/env python3
"""SAE-side claim helper for the /tmp/hades_gpu_coord/ protocol.

Coordinates GPU 0 and GPU 2 access with SMR-GC (cv-semseg). GPU 1 is SAE's
exclusive lane (no claim needed). GPUs 3 and 4 are SMR-GC exclusive
(never claim).

Protocol agreed 2026-04-27 between SAE Claude and SMR-GC Claude. Claim
file at /tmp/hades_gpu_coord/gpu{0,2}.claim:

    owner=sae
    pid=<int>
    started_at=<ISO 8601 UTC>
    heartbeat=<ISO 8601 UTC>   # holder refreshes every 60s
    purpose=<free text>

Take rule: GPU is free if claim file is missing OR (now - heartbeat) > 300s.
Atomic acquire: write .tmp + rename. Release: rm. Crash recovery via stale
window. Sweep guard: if a tmux session named '*-sweep' is alive, treat
GPU 0 and 2 as taken (SMR-GC autopilot may grab a 4080) until SMR-GC's
wrapper update ships.

CLI:
    python scripts/gpu_claim.py status
    python scripts/gpu_claim.py check 2
    python scripts/gpu_claim.py run --gpu 2 --purpose flat_gpt2_w12288_s1 -- uv run python -m foo
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import signal
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

COORD_DIR = Path("/tmp/hades_gpu_coord")
SHARED_GPUS = (0, 2)
EXCLUSIVE_SAE = (1,)
SMR_GC_EXCLUSIVE = (3, 4)
STALE_SECONDS = 300
HEARTBEAT_INTERVAL = 60
OWNER = "sae"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: str) -> _dt.datetime | None:
    try:
        return _dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc
        )
    except (ValueError, TypeError):
        return None


def _claim_path(gpu: int) -> Path:
    return COORD_DIR / f"gpu{gpu}.claim"


def _parse_claim(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    try:
        text = path.read_text()
    except OSError:
        return None
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def _is_stale(claim: dict[str, str], now: _dt.datetime) -> bool:
    hb_dt = _parse_iso(claim.get("heartbeat", ""))
    if hb_dt is None:
        return True
    return (now - hb_dt).total_seconds() > STALE_SECONDS


def _sweep_session_alive() -> bool:
    """SMR-GC runs benchmark sweeps in tmux sessions named '*-sweep'."""
    try:
        proc = subprocess.run(
            ["tmux", "list-sessions"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        return False
    return any("-sweep" in line for line in proc.stdout.splitlines())


def is_available(gpu: int) -> tuple[bool, str]:
    if gpu in EXCLUSIVE_SAE:
        return True, "exclusive SAE lane (no claim needed)"
    if gpu in SMR_GC_EXCLUSIVE:
        return False, f"GPU {gpu} is SMR-GC exclusive (off-limits)"
    if gpu not in SHARED_GPUS:
        return False, f"GPU {gpu} not in protocol scope"
    if _sweep_session_alive():
        return False, "SMR-GC '*-sweep' tmux session is alive"
    claim = _parse_claim(_claim_path(gpu))
    if claim is None:
        return True, "no claim file"
    now = _dt.datetime.now(_dt.timezone.utc)
    if _is_stale(claim, now):
        return True, f"stale claim from owner={claim.get('owner', '?')}"
    if claim.get("owner") == OWNER and claim.get("pid") == str(os.getpid()):
        return True, "ours, refreshing"
    return (
        False,
        f"held by owner={claim.get('owner', '?')} "
        f"pid={claim.get('pid', '?')} purpose={claim.get('purpose', '?')}",
    )


def _write_claim_atomic(gpu: int, fields: dict[str, str]) -> None:
    p = _claim_path(gpu)
    tmp = p.with_suffix(p.suffix + f".tmp.{os.getpid()}")
    body = "".join(f"{k}={v}\n" for k, v in fields.items())
    tmp.write_text(body)
    os.rename(tmp, p)


def _refresh_heartbeat(gpu: int) -> None:
    p = _claim_path(gpu)
    claim = _parse_claim(p)
    if (
        claim is None
        or claim.get("owner") != OWNER
        or claim.get("pid") != str(os.getpid())
    ):
        return
    claim["heartbeat"] = _now_iso()
    _write_claim_atomic(gpu, claim)


def _release(gpu: int) -> None:
    p = _claim_path(gpu)
    claim = _parse_claim(p)
    if (
        claim is not None
        and claim.get("owner") == OWNER
        and claim.get("pid") == str(os.getpid())
    ):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def acquire(gpu: int, purpose: str) -> bool:
    if gpu in EXCLUSIVE_SAE:
        return True
    avail, _ = is_available(gpu)
    if not avail:
        return False
    COORD_DIR.mkdir(exist_ok=True)
    _write_claim_atomic(
        gpu,
        {
            "owner": OWNER,
            "pid": str(os.getpid()),
            "started_at": _now_iso(),
            "heartbeat": _now_iso(),
            "purpose": purpose,
        },
    )
    return True


@contextmanager
def hold(gpu: int, purpose: str):
    if gpu in EXCLUSIVE_SAE:
        yield
        return
    if not acquire(gpu, purpose):
        _, reason = is_available(gpu)
        raise RuntimeError(f"cannot acquire GPU {gpu}: {reason}")
    stop = threading.Event()

    def _hb_loop() -> None:
        while not stop.wait(HEARTBEAT_INTERVAL):
            _refresh_heartbeat(gpu)

    t = threading.Thread(target=_hb_loop, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        _release(gpu)


def cli_status() -> int:
    COORD_DIR.mkdir(exist_ok=True)
    files = sorted(COORD_DIR.glob("gpu*.claim"))
    if not files:
        print(f"{COORD_DIR}: no active claims")
        return 0
    now = _dt.datetime.now(_dt.timezone.utc)
    for p in files:
        c = _parse_claim(p)
        if c is None:
            continue
        stale = _is_stale(c, now)
        print(
            f"{p.name}: owner={c.get('owner')} pid={c.get('pid')} "
            f"purpose={c.get('purpose')} heartbeat={c.get('heartbeat')} "
            f"stale={stale}"
        )
    return 0


def cli_check(gpu: int) -> int:
    avail, reason = is_available(gpu)
    print(f"GPU {gpu}: {'available' if avail else 'TAKEN'} — {reason}")
    return 0 if avail else 1


def cli_run(gpu: int, purpose: str, cmd: list[str]) -> int:
    try:
        ctx = hold(gpu, purpose)
        ctx.__enter__()
    except RuntimeError as e:
        print(f"gpu_claim: {e}", file=sys.stderr)
        return 3
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        proc = subprocess.Popen(cmd, env=env)

        def _forward(sig: int, _frame: object) -> None:
            proc.send_signal(sig)

        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, _forward)
        return proc.wait()
    finally:
        ctx.__exit__(None, None, None)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/gpu_claim", description=__doc__
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p_check = sub.add_parser("check")
    p_check.add_argument("gpu", type=int)
    p_run = sub.add_parser("run")
    p_run.add_argument("--gpu", type=int, required=True)
    p_run.add_argument("--purpose", required=True)
    p_run.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cmd == "status":
        return cli_status()
    if args.cmd == "check":
        return cli_check(args.gpu)
    if args.cmd == "run":
        argv = args.argv
        if argv and argv[0] == "--":
            argv = argv[1:]
        if not argv:
            print(
                "usage: gpu_claim run --gpu N --purpose STR -- CMD ARGS...",
                file=sys.stderr,
            )
            return 2
        return cli_run(args.gpu, args.purpose, argv)
    return 2


if __name__ == "__main__":
    sys.exit(main())
