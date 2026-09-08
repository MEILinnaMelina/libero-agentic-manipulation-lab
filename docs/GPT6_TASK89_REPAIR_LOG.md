# GPT-6 task 8 and 9 state 3 repair

The user explicitly requested adjustments and another GPT-6 run of the current
failed task states after `gpt6-single-state3-20260908-v1` completed 8/10.
Scope is task 8 and task 9, official state 3. Older results remain unchanged.
State 3 is now used for targeted development; this is not an untouched test set.

## Changes before rerun v1

- Task 8: measured pot bodies span approximately 8.1 cm horizontally, while the
  old target centers were only 8 cm apart. Increase center separation to at least
  11 cm, accounting for current body widths and keeping centers within the cook
  region. Assign targets by goal object identity, independent of selection order.
  Official On and Turnon predicates and physical contact evaluation are unchanged.
- Task 9: expose cavity insertion requirements in scene observations. The official
  Open predicate is true below -1.3 rad, but the existing opening skill targets
  the midpoint of [-2.094, -1.3]. The new readiness check uses that same target
  with a 0.05 rad tolerance. It is a skill precondition, not a changed success rule.
- Before grasping a mug for microwave insertion, require opening readiness and
  a handle strategy. The model receives those constraints and chooses every
  skill; invalid requests return explicit precondition feedback before motion.
  This prevents silently replacing a model-selected strategy or injecting a fixed
  action sequence. Object coordinates and trajectories remain deterministic.
- Extend `run_gpt_check.py` with explicit selected task IDs and a matching
  denominator, preserving the old default ten-task command. Unexpected execution
  interruptions are now labeled in the check status.

Offline validation: 13 tests passed, including positive body clearance in either
selection order and prevention of cavity grasps before opening readiness or with
an unsuitable strategy. These tests make no API calls and run no simulation.

Authorized rerun: `gpt6-t08-t09-state3-repair-v1`, tasks 8 and 9, state 3, exact
GPT-6 Astra, one worker, USD 3 independent ledger, 600 actions, at most 24 API
calls and 3 replans per episode. Both cameras, actions, model request/response
metadata, source snapshot and all episode outcomes are retained.

## Rerun v1 result

Both episodes succeeded on their first execution after these adjustments.

| Task | Prior state 3 result | Rerun state 3 | GPT calls | Replans |
|---|---|---|---:|---:|
| 8 | Failed at 600 steps | Success at 317 steps | 8 | 0 |
| 9 | Failed at 316 steps | Success at 351 steps | 6 | 0 |

Task 8 again selected pot 2 before pot 1; the second placement now completed
without any failed skill. Task 9 selected open_door, handle grasp, lift,
transport, place and close_door, with no precondition or motion failures.

All 14 returned model IDs were exactly `gpt-6-astra`. Total input 37,047 tokens,
output 1,171 tokens; usage-priced estimate USD 0.5215325, no unknown usage or API
errors. Combined episode runtime was 79.953 seconds. Both dual-camera MP4s were
decoded: 322 and 356 frames at 20 fps, matching 317/351 actions plus five settling
steps. Their final official goal predicates and final trace success flags were
all true. No success threshold, initial state or action budget was changed.

Execution code SHA-256:
`06ee8718755a31b2ef9bf358b8e5f616e02afa8ba6300d075182d8e7045b407c`.
Full results and videos:
`reports/gpt6-t08-t09-state3-repair-v1/RESULTS_ZH.md`.

Tasks 0 through 7 were not rerun under this updated implementation. The old
8/10 result and this targeted 2/2 development rerun remain separate; combining
them would not constitute a ten-task common-version success rate.
