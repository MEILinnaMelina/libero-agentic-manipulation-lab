# GPT-6 ten-task three-state check

On 2026-09-08 the user authorized another GPT-6 API run covering each of the
ten tasks in three different initial states. States 4, 5 and 6 were selected
before rollout, giving exactly 30 unique task/state pairs. Each pair executes
once. This bounded check does not resume the old incomplete formal campaign.

Run ID: `gpt6-all10-states456-20260908-v1`.
The manipulation implementation, skill constraints and GPT prompt are unchanged
from the successful targeted task 8/9 repair. Only orchestration was extended:
`run_gpt_check.py --states 4,5,6 --budget-usd 20` generates an explicit schedule,
keeps the legacy single-state option, rejects duplicate/development/out-of-range
states, and records the correct task/state denominator. A dedicated ledger limits
the batch to USD 20; any API error stops later episodes. No paid access probe was
performed. Every request belongs to a scheduled episode.

Offline checks before launch: 14 tests passed, including exact 30-pair coverage
and rejection of duplicate or inappropriate state selections. No tests invoked
the model or simulation. Episodes run sequentially under one immutable source
hash, with 600 actions, at most 24 GPT calls and 3 replans each. All attempts,
including failures, remain visible; there is no tuning or rerun within this batch.

Detailed results, raw usage and all video links are recorded under
`reports/gpt6-all10-states456-20260908-v1/`. These results measure official task
success; the additional RoboEval-style quality metrics discussed earlier have
not been introduced by this run.

## Observations during execution

- Task 1, state 6: butter reached the basket, but cream cheese did not. Three
  cream-cheese grasp attempts (top, top, side) reported `control_error`, with
  the actual end effector about 8 mm above the requested grasp height. GPT then
  requested `clear_obstruction` on the milk; this also failed to reach its
  target. The episode ended at 541 actions with `replan_budget`, 8 successful
  API responses and four failed skills. This is an execution failure, not an
  account-balance/API failure. The trace establishes target-reaching failure;
  it does not by itself prove which physical contact caused it. No retry or
  adjustment was made during this batch.
- Task 3, state 5: GPT grasped the bowl and immediately requested placement;
  the official containment predicate remained false after release
  (`placement_unstable`). Subsequent bottom-drawer opening, bowl clearance,
  and top-drawer closing requests each failed with `control_error`. Both
  required final predicates (bottom drawer closed, bowl contained) were false.
  The episode ended at 339 actions and five API calls with `replan_budget`.
  The trace records both unsuccessful planning/recovery choices and failure
  to reach control targets; this is not an API error.
- Task 7, state 6: one `control_error` was recovered through one replan; the
  episode succeeded in 383 actions and 11 calls. A failed skill is therefore
  not counted as a failed episode if recovery reaches official success.
- Task 9, state 4: the mug reached the microwave interior, but the door did
  not satisfy the official closed predicate. The first closing attempt moved
  the hinge from -1.9604 to -0.0376 radians (target -0.0025), then reported
  `mechanism_contact`. Subsequent closing/opening recovery attempts reported
  `control_error`. The episode exhausted 600 actions and eight API calls;
  final containment was true and door-closed was false. No tolerance change
  or post-hoc success override was applied.

## Completed batch

All 30 scheduled episodes completed: **27 successes (90%)**, three failures,
no API errors and no unexecuted pairs. There were 212 GPT-6 Astra calls, all
with observed usage; the ledger total is **USD 8.2216475**, estimated using the
saved token prices rather than a billing invoice. No extra calls were made
for testing, reruns or verification.

| Task | State 4 | State 5 | State 6 | Successes |
|---|---|---|---|---:|
| 0 | Success | Success | Success | 3/3 |
| 1 | Success | Success | Replan budget | 2/3 |
| 2 | Success | Success | Success | 3/3 |
| 3 | Success | Replan budget | Success | 2/3 |
| 4 | Success | Success | Success | 3/3 |
| 5 | Success | Success | Success | 3/3 |
| 6 | Success | Success | Success | 3/3 |
| 7 | Success | Success | Success | 3/3 |
| 8 | Success | Success | Success | 3/3 |
| 9 | Step budget | Success | Success | 2/3 |

Execution SHA-256:
`8c7738eecf1f8926ae7e89af6bd15366c777a5f68617c6f97ee99fc246a188dc`.
The complete source archive is saved beside the run identity. The offline
auditor `docs/tools/audit_gpt_runs.py` checks results against that identity,
reconciles request counts and costs with the ledger, verifies official goal
predicates, decodes every video and compares frame counts with trajectories.
It generates the video index and one final-frame contact sheet per state.

Final offline verification passed: all 30 videos decode at 512 x 256, 20 fps,
with frame counts equal to recorded actions plus five settling steps; all
27 successful episodes satisfy every official final goal predicate. The
three contact sheets were visually inspected. All 212 response metadata
files reconcile with the budget ledger, with zero unknown-usage requests.
Total episode wall time was 1,101.689 seconds (18 minutes 22 seconds).
The working execution source still matches the archived run hash.
