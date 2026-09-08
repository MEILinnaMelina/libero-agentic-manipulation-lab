# LIBERO-10 / GPT-6 Agentic v2 evaluation

September 7–8 work report: [详细 Word 工作报告](reports/work_report/LIBERO_工作报告_2026-09-07至08.docx).
Archive version: [v0.1.0-fixed-repair](VERSION.md).

Independent state-assisted adaptation of `E:\djf\RoboEval-main` to the locally pinned LIBERO benchmark. All source/assets are copied into `vendor`; original repositories receive no experiment outputs. See [METHOD.md](docs/METHOD.md) for privileged observations, source-to-port mapping and limitations, and [CHANGELOG.md](docs/CHANGELOG.md) for tuning history.

## Latest GPT check

Three-state follow-up on 2026-09-08: all ten tasks ran once on each of official
states 4, 5 and 6, using one unchanged manipulation version and GPT-6 Astra.
**27/30 successful (90%)**, 212 API calls, usage-priced estimate **USD 8.2216**;
no API errors. Tasks 1, 3 and 9 each succeeded in 2/3 states; all others in 3/3.
See [30-episode results and videos](reports/gpt6-all10-states456-20260908-v1/RESULTS_ZH.md)
and [execution/failure log](docs/GPT6_THREE_STATE_LOG.md).
This is a bounded three-state check, not the full 470-episode benchmark.

Targeted follow-up: after adjusting tasks 8 and 9, both succeeded on state 3
using GPT-6 Astra: **2/2**, 14 calls, estimated **USD 0.5215**, zero replans.
See [rerun results and videos](reports/gpt6-t08-t09-state3-repair-v1/RESULTS_ZH.md)
and [repair details](docs/GPT6_TASK89_REPAIR_LOG.md).
Tasks 0–7 were not rerun in this changed implementation; this does not replace
the original ten-episode result or establish a common-version 10/10 rate.

On 2026-09-08, an explicitly authorized ten-episode GPT-6 Astra check used state 3
once for every task: **8/10 successful**, 81 API calls, usage-priced estimate
**USD 3.0490**. Tasks 8 and 9 failed; no episodes were rerun. All ten videos were
decoded and checked. See [results and videos](reports/gpt6-single-state3-20260908-v1/RESULTS_ZH.md)
and [execution log](docs/GPT6_SINGLE_STATE_LOG.md).
This single-state check is separate from both the 24/24 fixed development
validation and the old incomplete formal campaign.

## Fixed semantic repair workflow

The old full GPT campaign remains paused. Fixed repair runs use deterministic plans with network connections disabled. See [repair log](docs/FIXED_REPAIR_LOG.md) and [results/video index](reports/fixed_repair/SUMMARY.md). Repairs change the implementation identity; do not mix these results with or resume the old frozen GPT campaign.

```powershell
# No API calls; unique run ID required; official development states only.
.\.venv-sim\python.exe scripts\debug_fixed.py --tasks 1,2,3,4,5,6,8,9 --states 0,1,2 --run-id fixed-repair-example
```

The runner retains all failed attempts, stops on the first failed fixed skill, and records every episode with both cameras. Original physics, official success predicates and the 600-action limit are preserved. Detailed results are development evidence, not held-out benchmark estimates. API commands below document the earlier protocol and are not part of this repair workflow.

## Reproduce on this Windows machine

Use PowerShell from `E:\djf\LIBERO-GPT6-Eval`:

```powershell
# Environment already created. To rebuild at the same project path:
$env:CONDA_PKGS_DIRS="$PWD\work\conda-pkgs"
$env:TEMP="$PWD\work\tmp"
$env:TMP=$env:TEMP
conda create --prefix .venv-sim python=3.8.13 pip -y
.\.venv-sim\python.exe -m pip install -r configs\requirements-sim.lock
.\.venv-sim\python.exe scripts\patch_windows.py
.\.venv-sim\python.exe -m pip check
.\.venv-sim\python.exe scripts\verify_sources.py
.\.venv-sim\python.exe scripts\eval.py manifest
.\.venv-sim\python.exe scripts\eval.py smoke
.\.venv-sim\python.exe -m pytest -q
```

`LIBERO_CONFIG_PATH` is configured automatically inside the project. Credentials are read only from `OPENAI_API_KEY` (optional `OPENAI_BASE_URL`); never put keys in source/config or output. No modern OpenAI SDK is required; Responses HTTPS JSON runs on Python 3.8. Exact `gpt-6-astra` is mandatory, with no other-model fallback.

```powershell
# API access and four-episode calibration pilot
.\.venv-sim\python.exe scripts\probe_api.py
.\.venv-sim\python.exe scripts\eval.py run --tasks 0,2 --states 0,1 --run-id pilot-example

# One task / all-task development, states 0,1,2 only
.\.venv-sim\python.exe scripts\eval.py run --tasks 0 --states 0 --run-id single-example
.\.venv-sim\python.exe scripts\eval.py run --run-id dev-example

# Calibrate and freeze only after completing development validation
.\.venv-sim\python.exe scripts\calibrate.py --run-id dev-example --write-config
.\.venv-sim\python.exe scripts\eval.py freeze

# Full formal evaluation: all 10 tasks, 47 unused states each
.\.venv-sim\python.exe scripts\eval.py run --split formal --run-id formal-gpt6

# Matched 5-state/task diagnostic conditions
.\.venv-sim\python.exe scripts\eval.py run --split formal --states 3,4,5,6,7 --method fixed --run-id formal-fixed
.\.venv-sim\python.exe scripts\eval.py run --split formal --states 3,4,5,6,7 --method gpt6_no_replan --run-id formal-no-replan

# Resume by repeating the exact command; summarize without executing episodes
.\.venv-sim\python.exe scripts\eval.py summarize --run-id formal-gpt6
```

The initial environment required one simulator process; four exhausted Windows commit memory, and that run remains separately archived. `max_episode_seconds` limits active execution, API requests have a timeout, no infrastructure retries occur, and missing/interrupted episodes stay visible. No successful-only selection is allowed. A fresh development version can rerun all development states with a new run ID, retaining prior evidence.

The final runtime fixes large MuJoCo workspace and BLAS thread reservations, validated against an identical recorded control trajectory. After these fixes the formal campaign uses three independent workers (`scripts/run_campaign.py --workers 3 --prefix frozen-v2`). The original four-worker memory-failure condition and the superseded early v1 formal condition are archived separately. See `docs/CHANGELOG.md` for their exclusion reasons; no favourable-result selection is used.

Replay a recorded success or failure without paying for another model call:

```powershell
.\.venv-sim\python.exe scripts\replay.py --episode runs\frozen-v2-gpt6\task00_state003 --output work\replay-example
```

The replay loads the same official initial state and executes saved actions through normal `env.step()`. It writes a video and the maximum state difference against the recorded trajectory. Use a new output directory. Final success terminates on the first official successful action, as in the upstream done-latching convention.

## Results and provenance

- `configs/task_manifest.json`: official task names, languages, BDDL/init file hashes, initial-state content hashes, splits.
- `configs/protocol.json`, `configs/frozen.json`: budgets and frozen identity.
- `docs/provenance/`: source commits/diffs, copied-file hashes, platform changes and hardware.
- `runs/<run>/run.json`: complete planned denominator and config snapshot.
- `runs/<run>/taskXX_stateYYY/`: initial/final scene, decisions, per-request request/response/metadata, skills, every action/state, dual-camera MP4, final result.
- `reports/<run>/`: task CSV, raw JSON results, JSON/Markdown summary and confidence intervals.
- `reports/cost_estimate.json`: actual pilot usage and projected standard-price cost/time. Usage estimates are not invoices.

Success is exclusively LIBERO `check_success`. Five zero stability steps precede 600 normal OSC actions at 20Hz. This protocol has 470 formal episodes because each task contains only 50 states and 3 were used for development. No claim of RGB-only VLA comparability or lifelong-learning protocol reproduction is made. Skill failure and API/environment failure are separate outcomes.

## Licenses

RoboEval source: Apache-2.0 (`vendor/RoboEval-main/LICENSE`). LIBERO source: MIT (`vendor/LIBERO/LICENSE`). Original notices and licenses are retained. New adapter modules disclose their derivation in `docs/METHOD.md`.
