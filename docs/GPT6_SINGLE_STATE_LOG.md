# GPT-6 ten-task single-state check

On 2026-09-08 the user explicitly authorized GPT-6 API execution of ten tasks,
one selected official initial state per task. This supersedes the earlier
fixed-only restriction for this bounded check; it does not resume the old
470-episode campaign or authorize successful-only repeated attempts.

Run: `gpt6-single-state3-20260908-v1`.
All task IDs 0 through 9 use official state 3, selected before any rollout.
The entry point is `scripts/run_gpt_check.py`, with one worker, an isolated
USD 10 reservation ledger, the existing 600-action and 24-model-call budgets,
and a complete per-run source archive. Any API error stops the remaining tasks.
Unexecuted tasks remain explicitly unexecuted. The old global frozen identity
and old formal run directories are preserved.

Pre-run inspection found that `clear_obstruction`, already implemented and used
by the fixed repair plan, was missing from the GPT skill schema. The skill is
now exposed to the model; physical destination selection remains deterministic.
No manipulation geometry, physics, success predicate or control budget changed.
Budget reserve/settle accept a per-run ledger path, so this check cannot consume
the historical USD 600 campaign allowance as its own budget.

Offline verification: 11 tests passed, including the new skill-schema regression
and isolated-ledger reservation/cap/settlement regression. Tests made no API calls.
There was no separate paid probe; the first request belongs to task 0's episode.

Final results and video audit are recorded in
`reports/gpt6-single-state3-20260908-v1/RESULTS_ZH.md` and `audit.json`.
This is a single-state check, not a 470-episode benchmark or a claim about all
initial states. The fixed 24/24 development validation remains a separate result.

## Completed result

All ten scheduled episodes completed: tasks 0 through 7 succeeded; task 8 hit
the 600-action limit, and task 9 exhausted three semantic replans. All 81 API
responses reported `gpt-6-astra`; no API errors or unknown usage records remain.
Input: 217,477 tokens. Output: 6,622 tokens. Usage-priced cost: USD 3.048955,
not an invoice. All ten 512x256, 20 fps videos decoded with exact trace/frame
counts. Eleven offline tests passed before execution; no additional simulation
or paid rerun followed the ten scheduled episodes.

Task 8 placed moka_pot_2 first. Placement of moka_pot_1 was blocked twice;
the planner temporarily set pot 1 on the table and attempted to rearrange pot 2.
The 600-step limit was reached with pot 1's On predicate still false, while
pot 2's On and the stove Turnon predicates were true. This is physical placement
and recovery-budget failure, not an API problem.

Task 9 chose a top/rim grasp instead of the handle strategy used in fixed repair.
The initial door hinge was about -1.432 rad (range -2.094 to 0); the planner
treated it as open and skipped the initial opening skill. Two placement
approaches were blocked. A later door-opening attempt and a side regrasp also
failed IK/collision checks. The run ended at 316 actions and 10 model calls,
with both containment and closure predicates false. These recorded differences
identify plausible contributors; isolating their individual effects requires
a separate experiment, which was not run.

Execution code SHA-256:
`72933d0b088bfc8aa543662d85c801b90a0dcb63b901e36392fb174a911fee1b`.
