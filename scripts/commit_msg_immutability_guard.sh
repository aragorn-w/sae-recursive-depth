#!/usr/bin/env bash
# scripts/commit_msg_immutability_guard.sh
# commit-msg hook. Aborts when (a) the commit message starts with [SAE-LOOP]
# (the runner prefix per CLAUDE.md rule 7) AND (b) any staged path matches
# the protected-paths list.
#
# Human commits (any non-[SAE-LOOP] prefix) pass through unconditionally;
# the operator has authority to edit protected files.
#
# Installed via `git config core.hooksPath scripts/git-hooks` and a symlink:
# scripts/git-hooks/commit-msg -> ../commit_msg_immutability_guard.sh

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

MSG_FILE="${1:-}"
if [[ -z "$MSG_FILE" || ! -f "$MSG_FILE" ]]; then
    # Not invoked with a message file (shouldn't happen for commit-msg).
    # Fail open rather than blocking unrelated git operations.
    exit 0
fi

first_line=$(head -1 "$MSG_FILE" 2>/dev/null || echo "")
if [[ "$first_line" != \[SAE-LOOP\]* ]]; then
    exit 0
fi

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

echo "[SAE-LOOP] immutability-guard: runner attempted to modify protected paths:" >&2
for v in "${violations[@]}"; do
    echo "  - $v" >&2
done
echo "commit aborted. protected paths may only be modified by the human (use a non-[SAE-LOOP] prefix)." >&2
exit 1
