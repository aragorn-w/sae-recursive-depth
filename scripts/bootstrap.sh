#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/home/wanga/school/math498c/sae-recursive-depth"
NTFY_TOPIC="${NTFY_TOPIC:-sae-wanga-research}"

cd "$REPO_ROOT"

echo "[bootstrap] repo root: $REPO_ROOT"

# uv install check
# uv is the project's python package manager; see https://docs.astral.sh/uv/
if ! command -v uv >/dev/null 2>&1; then
    echo "[bootstrap] uv not found; installing via official installer"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1090
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
echo "[bootstrap] uv: $(uv --version)"
echo ""

# directory scaffolding
mkdir -p src/training src/metrics src/analysis src/data
mkdir -p experiments/logs experiments/artifacts
mkdir -p handoffs vendored
mkdir -p scripts/git-hooks
mkdir -p .claude/rules .claude/commands
touch src/__init__.py src/training/__init__.py src/metrics/__init__.py src/analysis/__init__.py src/data/__init__.py

# python environment
# uv sync creates/updates .venv from pyproject.toml and resolves uv.lock deterministically
# reference: https://docs.astral.sh/uv/reference/cli/#uv-sync
uv sync
UV_PYTHON="$REPO_ROOT/.venv/bin/python"

# git hook installation
# core.hooksPath tells git to look for hooks in the named directory instead of .git/hooks
# reference: git-config(1)
git config core.hooksPath scripts/git-hooks
ln -sf "$REPO_ROOT/scripts/pre_commit_immutability_guard.sh" scripts/git-hooks/pre-commit
chmod +x scripts/pre_commit_immutability_guard.sh scripts/git-hooks/pre-commit

# cron heartbeat
# schedule: 0 8,20 * * * with TZ=America/Denver
# uv run ensures the heartbeat runs inside the project venv without pre-activation
CRON_LINE="0 8,20 * * * TZ=America/Denver cd $REPO_ROOT && $HOME/.local/bin/uv run python scripts/heartbeat.py >> experiments/logs/heartbeat.log 2>&1"
if ! crontab -l 2>/dev/null | grep -F "scripts/heartbeat.py" >/dev/null; then
    (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
    echo "[bootstrap] installed cron heartbeat at 8am and 8pm America/Denver"
else
    echo "[bootstrap] cron heartbeat already installed; leaving unchanged"
fi

# cuda and gpu verification
# nvidia-smi -L lists each GPU by index and name; useful to confirm we see all 5
# reference: nvidia-smi(1)
echo ""
echo "[bootstrap] GPU inventory:"
nvidia-smi -L || echo "[bootstrap] WARNING: nvidia-smi not found or no NVIDIA driver"

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -ne 5 ]; then
    echo "[bootstrap] WARNING: expected 5 GPUs, found $GPU_COUNT"
fi

# cuda runtime check through torch (inside the uv-managed venv)
"$UV_PYTHON" -c "import torch; print(f'[bootstrap] torch {torch.__version__}, cuda available={torch.cuda.is_available()}, device count={torch.cuda.device_count()}')"

# ntfy subscription test
# curl flags: --max-time limits total seconds, --silent hides progress bar, --show-error still prints errors, --fail returns non-zero on HTTP >= 400
# reference: curl(1)
echo ""
echo "[bootstrap] testing ntfy topic $NTFY_TOPIC"
curl --max-time 10 --silent --show-error --fail \
    -H "Title: bootstrap-complete" \
    -H "Priority: low" \
    -H "Tags: bootstrap,setup" \
    -d "bootstrap.sh finished on $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "https://ntfy.sh/$NTFY_TOPIC" || echo "[bootstrap] ntfy test failed; check network and retry"

echo ""
echo "==================================================================="
echo "[bootstrap] automated setup complete"
echo "==================================================================="
echo ""
echo "Manual steps you still need to do:"
echo ""
echo "  1. W&B login:"
echo "       uv run wandb login"
echo ""
echo "  2. HuggingFace login (required for Gemma-2-2B gated access):"
echo "       uv run huggingface-cli login"
echo "     Then accept the Gemma license at https://huggingface.co/google/gemma-2-2b"
echo ""
echo "  3. Subscribe to ntfy topic on your phone and desktop:"
echo "       https://ntfy.sh/$NTFY_TOPIC"
echo "     Confirm you received the bootstrap-complete notification just sent."
echo ""
echo "  4. Verify the cron heartbeat fires by running manually:"
echo "       uv run python scripts/heartbeat.py"
echo ""
echo "  5. When ready to start the autonomous loop:"
echo "       tmux new-session -d -s sae-runner 'bash scripts/run_loop.sh'"
echo "       tmux attach -t sae-runner"
echo ""
