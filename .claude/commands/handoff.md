---
description: Generate a session handoff markdown file summarizing work done, next steps, and blockers.
---

# /handoff

Produce a handoff document at `handoffs/YYYY-MM-DD-HHMM.md` (America/Denver local time) capturing the current state of the project. Invoked at the end of any interactive human session so the next session (human or runner) has full context without replaying the chat.

## Required content

The handoff file must contain these sections in this order.

### 1. Timestamp and session summary

Local time, git commit SHA at session end, and a 2-3 sentence summary of what was worked on.

### 2. What was done

Bullet list of concrete actions taken this session. Commits made, experiments launched or completed, figures generated, bugs fixed, decisions updated. Reference experiment ids from `EXPERIMENTS.yaml` where relevant.

### 3. Current matrix position

Read `experiments/state.json` and report:
- Total experiments in matrix
- Count by status (pending, running, complete, failed, skipped_by_gate)
- Currently running experiment id (if any)
- Next experiment the runner will dispatch

### 4. Recent gate outcomes

Read the last 10 rows of `experiments/results.tsv`. For each row with a gate decision, list the experiment id, which gate fired, the metric value, and the action taken.

### 5. What's next

Bullet list of concrete next steps. Include any experiments that will be dispatched by the runner in the next 12 hours based on dependency graph.

### 6. Blockers

Anything the runner cannot self-resolve. API quota issues, missing HuggingFace access (e.g., Gemma gated repo), disk space concerns, W&B outages, unusual decision-gate triggers that need human review.

### 7. Notes for next session

Free-form notes, questions to investigate, things noticed but not addressed. Anything the next session should know.

## Procedure

1. Compute local timestamp as `YYYY-MM-DD-HHMM` in America/Denver.
2. Read `experiments/state.json` and `experiments/results.tsv`.
3. Run `git rev-parse HEAD` for the current commit SHA.
4. Assemble sections 1-7 using actual data from the files above. Do not fabricate values; if a section has nothing to report, write "nothing to report this session".
5. Write the file to `handoffs/<timestamp>.md`.
6. Commit with message `[SAE-LOOP] handoff <timestamp>` and push.
7. Send ntfy at priority `low` with title `handoff-generated` and message containing the handoff filename.

## Style

Prose in the handoff must avoid em dashes. Use plain sentences. No interpretation of whether results "look promising" or "look concerning". Interpretation goes in `lab_notebook.md` by the human only.
