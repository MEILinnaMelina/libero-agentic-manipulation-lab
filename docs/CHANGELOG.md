# Development changes

All control tuning uses official development states 0,1,2 only.

## Fixed-only repair branch (2026-09-08)

The user requested repair of tasks 1/2/3/4/5/6/8/9 using fixed semantic experiments only. `debug_fixed.py` disables network connections and creates a deterministic plan before execution, including explicit geometric obstruction clearance where needed. It never constructs a GPT planner. Failed attempts remain archived.

Repairs correct grip-site versus wrist-body orientation, Panda closing-axis selection, settling observations, thin-object grasp depth, mug/bowl rim and mug-handle selection, stove-fin contact, pot-knob grasp and two-pot spacing, book-to-slot alignment, drawer push clearance, and microwave front entry/release/door closure. Pose tracking tolerance is 0.10 rad; this is an algorithmic stopping criterion, not a change to controller gains or physics. Details and per-attempt evidence are in [FIXED_REPAIR_LOG.md](FIXED_REPAIR_LOG.md).

Unified v1 development verification achieved 23/24 successes. Its one failure exposed a clearance-radius boundary; the corrected state then succeeded in 574/600 actions. The v2 run is the final common-code confirmation; its authoritative results and recordings are linked in [the repair report](../reports/fixed_repair/SUMMARY.md). The previous paid formal experiment remains paused and its frozen identity is obsolete for the repaired code.

## Earlier development history

- Initial port: single-arm schema, full BDDL goals and state privilege, private MuJoCo IK, geometry-based grasp/transport/place, mechanism joint trajectories, fixed BDDL diagnostic plan. These are LIBERO adaptation changes and are not claimed to be identical original RoboEval skills.
- Windows compatibility: package logger/DLL/renderer corrections; missing termcolor/future dependencies added. Original environments and source repositories unchanged.
- Control validation v1: translation and gripper direction passed; orientation convergence test exposed too-wide tolerance. Tightened to 0.04 rad. Grasp, lift (~0.12 m) and transport passed. Container descent could contact a wall.
- Control validation v2: rotation now passes (~0.122 rad for 0.15 requested). Placement excludes parent fixture from obstacle candidates and accounts for object extent. Container contact remains a monitored control failure; model receives failure feedback.
- Protocol: actual resource enumeration found 50 states/task, leaving 47/task for formal evaluation after three development states.
- Control validation v3: release is allowed after bounded vertical support contact (horizontal error <2 cm; vertical error <4 cm); the placement predicate is still checked after release. Full grasp/lift/transport/place validation passes.
- Memory validation: compiler workspace allocation reduced with unchanged physical parameters. All 262 recorded actions and simulator states match exactly. A full 600-step zero-action failure trajectory validates budget enforcement and denominator retention.
- Development v2 four-worker run exhausted system commit memory; original successes, failures and interruptions remain archived. No entries from it are pooled with later runs.
- Development v3: all 30 episodes complete without simulation/API errors; task 0 and 7 have 3/3 successes, task 1 has 1/3, remaining tasks 0/3. Development coverage is not formal performance.
- Development v4: carried-object private-state collision transforms and <=4 cm path samples added. Side/handle semantic strategies select alternate geometric candidates. API prompts round observations to four decimals and include compact execution feedback; full-precision states remain in artifacts.
- Final runtime: one BLAS/OpenMP thread per worker; fixed diagnostic planner covers all three development states for all tasks before formal freeze. Formal ablations use matching states 3–7, five per task; main experiment uses states 3–49.
- V1 formal run was stopped during task 0 after a review of **development task 9** exposed omitted `conaffinity` contact masks in mechanism geometry selection. All existing v1 outputs and its frozen source archive remain separate; no v1 result is selected or pooled into v2. The correction uses development evidence, not held-out task performance, and applies uniformly to all mechanisms.
- Microwave fix: include either nonzero `contype` or `conaffinity`. All three development states now execute opening and closing trajectories; opening succeeds, while closing can still fail on tracking error. This is a skill/control failure, not a missing-geometry precondition.
- V2 stops immediately on the first official successful policy action, matching the upstream done-latching convention; it does not continue release/settling after benchmark termination. A dedicated test verifies the successful action is recorded before termination. All 30 development states are rerun under this final convention before refreezing.
