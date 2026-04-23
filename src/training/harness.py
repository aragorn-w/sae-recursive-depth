"""Shared harness for training entry points.

Consolidates the boilerplate mandated by `.claude/rules/training.md`:
  - rule 1: `wandb.init` is called for every real training body; scaffold
    stubs do not create cloud runs.
  - rule 2: `results_io.append_result` in a `finally:` block.
  - rule 3: `set_all_seeds(seed)` before any model/data code.
  - rule 5: artifact directory `experiments/artifacts/<experiment_id>/`.

Entry points use `experiment_context(...)` as a context manager. Real
training bodies call `ctx.init_wandb()` once they are ready to log. Scaffold
bodies set `ctx.status = "scaffold_stub"` and return; the harness then exits
the process with code 99 so `run_loop.sh` knows to silent-skip the row.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.training.config import ExperimentNotFound, load_row
from src.training.results_io import append_result
from src.training.seed import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "experiments" / "artifacts"
WANDB_PROJECT = "sae-recursive-depth"

# Exit code emitted by the harness when the body reported scaffold_stub.
# run_loop.sh recognizes this and silent-skips the row (no ntfy, no extra
# TSV row, no commit, adds the id to an in-session skip set so the outer
# loop does not re-pick it).
EXIT_SCAFFOLD_STUB = 99


@dataclass
class ExpContext:
    """Everything a training entry point needs from the harness."""

    experiment_id: str
    row: dict[str, Any]
    artifact_dir: Path
    seed: int
    smoke: bool
    wandb_run: Any | None = None
    start_time: float = field(default_factory=time.time)
    # Populated by the body. Used to construct the results.tsv row on exit.
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    notes: str = ""

    def init_wandb(self) -> Any | None:
        """Lazily initialize W&B. Call once from the training body.

        Scaffold stubs do not call this, so stub runs never create a cloud
        run. Real training bodies call it before their first `wandb.log`.
        """
        if self.wandb_run is not None:
            return self.wandb_run
        self.wandb_run = _init_wandb(self)
        return self.wandb_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Standard flag set for every training / analysis entry point.

    `--experiment-id` is required unless `--smoke` is set. `--smoke` bypasses
    the matrix lookup and runs the harness against a synthetic row so the
    pipeline can be exercised without modifying the protected matrix.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def _synthetic_row(arch_hint: str) -> dict[str, Any]:
    """A fake row used only when `--smoke` is passed."""
    return {
        "id": f"smoke_{arch_hint}",
        "description": f"smoke run for {arch_hint}",
        "base_model": "smoke",
        "level0_arch": arch_hint,
        "level0_source": "smoke",
        "site": "residual",
        "layer": 0,
        "width": 0,
        "depth": 0,
        "seed": 0,
        "dict_ratio": None,
        "sparsity": 0,
        "gpu_preference": "",
        "dependencies": [],
        "decision_gates": [],
        "estimated_gpu_hours": 0.0,
        "outputs": [],
        "status": "smoke",
    }


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _init_wandb(ctx: ExpContext) -> Any | None:
    """Init W&B per training rule 1. Smoke runs use mode='disabled'."""
    try:
        import wandb  # type: ignore
    except ImportError:
        sys.stderr.write("[harness] wandb not installed; continuing without it\n")
        return None
    try:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=ctx.experiment_id,
            config=ctx.row,
            mode="disabled" if ctx.smoke else "online",
            reinit=True,
        )
        return run
    except Exception as e:
        sys.stderr.write(f"[harness] wandb.init failed ({e!r}); continuing\n")
        return None


@contextmanager
def experiment_context(arch_hint: str) -> Iterator[ExpContext]:
    """Set up the harness, hand control to the entry point, finalize on exit.

    Usage:
        def main() -> None:
            with experiment_context(arch_hint="batchtopk") as ctx:
                if ctx.smoke:
                    return
                ctx.init_wandb()
                # ... training code writes into ctx.metrics ...
                ctx.metrics["variance_explained"] = ...
                ctx.status = "ok"

    Scaffold bodies set `ctx.status = "scaffold_stub"` and return without
    calling `ctx.init_wandb()`. The harness sees the status and exits the
    process with code `EXIT_SCAFFOLD_STUB` so `run_loop.sh` can silent-skip.
    """
    args = parse_args()
    if not args.smoke and not args.experiment_id:
        sys.stderr.write(
            "[harness] --experiment-id is required unless --smoke is passed\n"
        )
        sys.exit(2)

    if args.smoke:
        row = _synthetic_row(arch_hint)
        experiment_id = row["id"]
    else:
        try:
            row = load_row(args.experiment_id)
        except ExperimentNotFound as e:
            sys.stderr.write(f"[harness] {e}\n")
            sys.exit(3)
        experiment_id = row["id"]

    seed = int(row.get("seed", 0) or 0)
    set_all_seeds(seed)

    artifact_dir = ARTIFACTS_ROOT / experiment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    ctx = ExpContext(
        experiment_id=experiment_id,
        row=row,
        artifact_dir=artifact_dir,
        seed=seed,
        smoke=args.smoke,
    )

    exc: BaseException | None = None
    try:
        yield ctx
        if ctx.status == "running":
            ctx.status = "smoke_ok" if args.smoke else "ok"
    except KeyboardInterrupt as e:
        exc = e
        ctx.status = "interrupted"
        ctx.notes = ctx.notes or "KeyboardInterrupt"
    except Exception as e:
        exc = e
        ctx.status = "failed"
        ctx.notes = ctx.notes or f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        elapsed = time.time() - ctx.start_time
        wandb_url = ""
        try:
            if ctx.wandb_run is not None and hasattr(ctx.wandb_run, "url"):
                wandb_url = ctx.wandb_run.url or ""
        except Exception:
            pass
        row_out = {
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "experiment_id": ctx.experiment_id,
            "status": ctx.status,
            "base_model": row.get("base_model", ""),
            "level0_arch": row.get("level0_arch", ""),
            "depth": row.get("depth", ""),
            "seed": ctx.seed,
            "width": row.get("width", ""),
            "variance_explained": ctx.metrics.get("variance_explained"),
            "pwmcc": ctx.metrics.get("pwmcc"),
            "mmcs": ctx.metrics.get("mmcs"),
            "pwmcc_null_mean": ctx.metrics.get("pwmcc_null_mean"),
            "pwmcc_null_std": ctx.metrics.get("pwmcc_null_std"),
            "dead_latent_fraction": ctx.metrics.get("dead_latent_fraction"),
            "gpu_hours": elapsed / 3600.0,
            "commit_sha": _git_sha(),
            "wandb_run_url": wandb_url,
            "notes": ctx.notes,
        }
        try:
            append_result(row_out)
        except Exception as write_err:
            sys.stderr.write(f"[harness] append_result failed: {write_err!r}\n")
        try:
            if ctx.wandb_run is not None:
                ctx.wandb_run.finish()
        except Exception:
            pass
        if exc is not None:
            raise exc
        if ctx.status == "scaffold_stub":
            sys.exit(EXIT_SCAFFOLD_STUB)
