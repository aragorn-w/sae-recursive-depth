#!/usr/bin/env bash
# scripts/ntfy_send.sh
# Usage: ntfy_send.sh <priority> <title> <message> [tags]
# Priority: min | low | default | high | urgent
# Tags: optional comma-separated list (e.g., "warning,skull")

set -u

PRIORITY="${1:-default}"
TITLE="${2:-[SAE] untitled}"
MESSAGE="${3:-}"
TAGS="${4:-}"
TOPIC="${NTFY_TOPIC:-sae-wanga-research}"

CURL_ARGS=(
    -s
    -X POST
    -H "Title: $TITLE"
    -H "Priority: $PRIORITY"
)

if [[ -n "$TAGS" ]]; then
    CURL_ARGS+=(-H "Tags: $TAGS")
fi

CURL_ARGS+=(-d "$MESSAGE" "https://ntfy.sh/$TOPIC")

curl "${CURL_ARGS[@]}" --max-time 10 >/dev/null 2>&1 || {
    echo "ntfy post failed (non-fatal): $TITLE" >&2
    exit 0
}
