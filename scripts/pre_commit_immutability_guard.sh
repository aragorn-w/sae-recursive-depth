#!/usr/bin/env bash
# scripts/pre_commit_immutability_guard.sh
# Pre-commit hook. Aborts commit if any staged path matches the protected-paths list.
# Installed via `git config core.hooksPath scripts/git-hooks` and a symlink from
# scripts/git-hooks/pre-commit -> ../pre_commit_immutability_guard.sh

set -u

PROTECTED_PATHS=(
    "CLAUDE.md"
    "EXPERIMENTS.yaml"
    "DECISIONS.md"
    "SPEC.md"
    "proposal_v3.docx"
    "lab_notebook.md"
    "README.md"
    ".claude/rules/"
    ".claude/agents/"
    "vendored/"
)

is_runner_commit() {
    local msg
    msg=$(cat "${1:-.git/COMMIT_EDITMSG}" 2>/dev/null || echo "")
    if [[ "$msg" == \[SAE-LOOP\]* ]]; then
        return 0
    fi
    return 1
}

staged=$(git diff --cached --name-only --diff-filter=ACMRD)
if [[ -z "$staged" ]]; then
    exit 0
fi

violations=()
while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    for p in "${PROTECTED_PATHS[@]}"; do
        if [[ "$p" == */ ]]; then
            if [[ "$f" == "$p"* ]]; then
                violations+=("$f (matches protected prefix $p)")
                break
            fi
        else
            if [[ "$f" == "$p" ]]; then
                violations+=("$f (protected file)")
                break
            fi
        fi
    done
done <<< "$staged"

if (( ${#violations[@]} == 0 )); then
    exit 0
fi

if is_runner_commit ".git/COMMIT_EDITMSG"; then
    echo "[SAE-LOOP] immutability-guard: runner attempted to modify protected paths:" >&2
    for v in "${violations[@]}"; do
        echo "  - $v" >&2
    done
    echo "commit aborted. protected paths may only be modified by the human." >&2
    exit 1
fi

echo "immutability-guard warning: commit touches protected paths:" >&2
for v in "${violations[@]}"; do
    echo "  - $v" >&2
done
echo "this appears to be a human commit (message does not start with [SAE-LOOP])." >&2
echo "if you intended to edit a protected file, re-run with: git commit --no-verify" >&2
echo "aborting by default for safety." >&2
exit 1
