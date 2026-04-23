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
TOPIC="${NTFY_TOPIC-sae-wanga-research}"

# Kill switch: NTFY_TOPIC="" (exported as empty) disables notifications for
# this invocation. Used during bring-up / scaffold iteration so scaffold
# stubs don't burn the phone. An unset variable still resolves to the
# default topic.
if [[ -z "$TOPIC" ]]; then
    exit 0
fi

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
