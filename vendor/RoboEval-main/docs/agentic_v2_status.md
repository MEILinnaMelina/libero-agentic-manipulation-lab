# Agentic v2 - Current Verified Status

Single source of truth for what is actually verified to work right now. If
this file disagrees with `agentic_v2_change_log.md`, `agentic_v2_runbook.md`,
or any commit message, **this file wins** - the others describe history,
intent, or a snapshot from an earlier commit, not necessarily current fact.
Update this file whenever a claim below stops being true.

## Environment

Use the `roboeval` conda env, not base Anaconda - the base env is missing
`mujoco`/`dm_control`/`imageio` and its `pytest` hangs at plugin
auto-discovery, which looks like a stuck test run:

```
C:\Users\melin\.conda\envs\roboeval\python.exe
```

`PYTHONPATH=<repo root>` (or run from repo root) is required when invoking
`examples/*.py` directly - `roboeval` is not pip-installed in this env.
`thirdparty/mujoco_menagerie` contains untracked, locally modified robot
XMLs (`panda_nohand_modified.xml` etc.); a fresh checkout or `git worktree`
does not have them and every env construction fails with
`FileNotFoundError` - run experiments from this working tree.

## Unit tests

`pytest tests/test_agentic_v2_*.py` in the `roboeval` env: **45/45 passed**
at the current commit.

## Scope: all nine RoboEval base tasks

`BASE_TASK_KEYS` now covers every base task the assignment lists:
`cube_handover`, `lift_pot`, `stack_two_blocks`, `vertical_cube_handover`,
`lift_tray`, `pack_box`, `pick_single_book`, `stack_single_book_shelf`,
`rotate_valve`. Two semantic skills were added for the box and the valves
(`close_flap`, `rotate`); everything else reuses the existing
grasp/bimanual_grasp/lift/handover/place skills. The LLM still only
chooses skills; every pose, offset and contact rule below is derived by
the deterministic layer from measured scene geometry.

## Phase 9 deterministic gate (`--planner fixed`, no LLM) - PASSED, 9/9 tasks

`benchmark_success` (RoboEval's raw metric), seeds 0-9:

| Task | Result | Commit | Notes |
|---|---|---|---|
| `lift_pot` | **10/10** | `42eb33b` | `outputs/det_gate_lift_pot_grasp_policy_regression`; seed 3 re-verified at the current tree |
| `cube_handover` | **10/10** | `42eb33b` | `outputs/det_gate_staged_both_seeds0-9`; seed 3 re-verified at the current tree |
| `stack_two_blocks` | **10/10** | `42eb33b` | same run; seed 0 re-verified at the current tree; behavior quality 0/10 (see note) |
| `vertical_cube_handover` | **10/10** | `a4a47f2` | `outputs/det_gate_new_tasks_a`; behavior quality 10/10 |
| `lift_tray` | **10/10** | `a4a47f2` | same run; behavior quality 10/10 |
| `pack_box` | **10/10** | `a4a47f2` | `outputs/det_gate_new_tasks_b`; behavior quality 0/10 - RoboEval's `slip_count` registers 1 per trial while the fingertip pushes the flap |
| `rotate_valve` | **10/10** | `a4a47f2` | same run; behavior quality 0/10 - `env_collision_count` 6 per trial: RoboEval counts the gripper's contact with the valve's own body as an environment collision |
| `pick_single_book` | **10/10** | current tree | `outputs/det_gate_new_tasks_c`; `no_slip` fails (the book creeps a few degrees in the pinch during the lift) |
| `stack_single_book_shelf` | **10/10** | current tree | same run |

The base tasks have no pose randomisation (only the handover rods vary
by seed), so a 10-seed gate mostly checks determinism; the seed-varying
tasks are `cube_handover`, `vertical_cube_handover` and `stack_two_blocks`.

`stack_two_blocks` behavior-quality note: every trial records exactly one
robot-environment contact in RoboEval's own `env_collision_count`, a
fingertip grazing the table by ~0.1 mm at the receiver's regrasp close on
the 4 cm block, which the grasp contact policy deliberately tolerates (up
to 4 mm, fingers only). It is a metric refinement, not a task failure.

Reproduce: `examples/evaluate_agentic_v2.py --launch --methods v2-fixed
--tasks <task> --seeds 0 1 2 3 4 5 6 7 8 9`.

Recordings of one successful seed-0 run per task (local only, not
committed - see `results/README.md`):
- first three tasks: `outputs/gif_success_seed0/v2-fixed/<task>/seed_000/trajectory.gif`
  (LLM-driven: `outputs/gif_api_success_seed0/v2-full/<task>/seed_000/trajectory.gif`)
- six new tasks: `outputs/gif_success_seed0_newtasks/v2-fixed/<task>/seed_000/trajectory.gif`
  (LLM-driven: `outputs/gif_api_success_seed0_newtasks/v2-full/<task>/seed_000/trajectory.gif`,
  recorded at `172818e`, every one `benchmark_success=1.0`)

## How each new task is solved (measured geometry, not tuned constants)

- **vertical_cube_handover** - the rod starts standing on end and drops 5 cm
  onto the table during the first settle. The grasp candidates are now
  computed *after* the open-gripper settle, tall objects are pinched
  `0.028 m` below their top (the palm face sits `0.035 m` above the pad
  centres - a 0.035 inset put the palm on the rod), and the staged
  handover uses the world-vertical half extent (0.10 m for the standing
  rod, not `canonical_size[2]/2 = 0.02`) to set it back down.
- **lift_tray** - both arms pinch the two long rim walls (2.6 cm thick,
  10 cm tall, at body-frame `x = +/-0.266`) from above, 3 cm below the
  top edge; the wall at larger world y goes to the left arm. Lift tilt is
  now measured as the change from the resting orientation (the tray's
  body Y is up, so the old body-Z tilt test was meaningless).
- **pack_box** - `close_flap`: the closed fingertip sweeps an arc of radius
  0.12 m about each flap's hinge, from just below the open flap to
  ~177 degrees, with the hand *trailing* the motion (fingers pointing along
  the arc's tangent, pitch clamped at 40 degrees up). A hand pointing
  straight down sits in the open flap's plane and cannot get under it
  (-8.7 mm palm/flap penetration at the first waypoint). The joint anchor
  sits at one *end* of the hinge (x = 0.386 for the left flap); the arc is
  centred on the plate's projection onto the hinge line instead, which is
  what the left arm needed (elbow at its limit otherwise). The right arm
  closes the flap on its side first, then the left. Each flap's normalised
  state ends at ~0.0003 (right) / ~-0.08 (left) against RoboEval's 0.1 bar.
- **rotate_valve** - `rotate`: the handwheel is a horizontal 6.2 cm disc,
  2.8 cm tall, with a vertical revolute joint (damping 0.01) on a 250 kg
  base. The arm pinches it from above (pads 6 mm above the wheel centre so
  the fingertips clear the valve body), twists the wrist about world +Z in
  0.35 rad steps until the normalised state passes 0.16 (RoboEval needs
  > 0.10), and can let go, wind back and re-grip if a wrist joint runs out
  of travel. The task's success is polled inside each twist without
  stopping the twist early (a 0.1002 result was measured when the executor
  stopped on success mid-twist).
- **pick_single_book / stack_single_book_shelf** - the book is **4 kg**
  (`book/book` body mass), lies flat on the counter (16.5 x 9.5 x 3.1 cm)
  and overhangs the counter's front edge by 3.3 cm. No top-down pinch fits
  the 8 cm aperture, so it is pinched across its thickness at the
  overhanging short edge with a horizontal hand (`edge_grasp`
  candidates). Pinched 1.3 cm in, the 6.9 cm lever puts ~2.7 Nm on two
  2 cm pads and the book creeps in the grip (2.8 cm rise for a 9 cm hand
  rise); standing it up over the pinch or raising it flat both saturate
  the 12 Nm wrist-pitch actuator (measured 0.15-0.29 rad steady error,
  hand sinking 10 cm). What works: with the book still on the counter,
  drag it back until ~7 cm overhangs (the pads slip along it, so the drag
  is repeated on the live overhang), open, slide the fingers 4.2 cm under
  the overhang and re-grip so the book's edge butts against the palm
  (that palm contact is now an allowed carry contact), then verify-lift.
  Placement keeps the flat resting orientation, leaves `depth + 1.2 cm`
  of the book overhanging the shelf's front edge so the lower finger never
  lands on the plank, lowers until the book touches `object:lower_shelf`,
  releases, and backs the hand straight out before retreating upward. The
  shelf planks are exposed to the planner as fixed objects
  (`lower_shelf`, `upper_shelf`, `fixed: true`) so `place` can target
  `on:lower_shelf`; the counter is labelled `scene:table`.

## Robustness fixes that fell out of the new tasks (all measured)

- **Idle-arm ratchet.** Every plan re-anchored the idle arm at its
  *measured* joints; position servos settle a few mm below target under
  gravity, so the idle hand walked downward plan after plan (7 cm over
  ~250 steps in `pack_box`) until it rested on the box and vetoed every
  plan for the working arm. IK seeds and hold plans now use the last
  *commanded* joints (`JointActionAdapter.last_commanded`).
- **Convergence wait.** The executor holds the final plan point until the
  joints are within 0.03 rad (up to 40 extra steps): a torque-limited
  joint lags a ramp well inside the 0.3 rad tracking tolerance yet left
  the hand ~10 cm short of a 17 cm raise.
- **Closed fingers are not a self-collision.** With the gripper fully
  closed on nothing the two finger pads touch; that was rejecting every IK
  candidate for the flap push.
- **Idle arm parked.** Single-arm skills (`rotate`, `close_flap`) raise the
  idle arm 5 cm first; in the valve scene both hands start 1 cm above the
  two wheels.
- **`canonical_size` honours per-geom placement** inside the body (the
  book mesh is rotated 90 degrees in its body; the tray is nine offset
  geoms; the pot's handles are now included: 0.39 x 0.52 x 0.22 m).
- **Fixture-aware skills.** `grasp`/`lift`/`handover` refuse `fixed`
  objects; `_resting_surface_z` ignores them; the prompt tells the LLM
  what `fixed` means.

## API / LLM runs

**Formal run at `42eb33b` - 30/30** (`v2-full`, OpenAI `gpt-5.6-terra`,
seeds 0-9, first three tasks). Summary in
`results/openai-full_42eb33b_seeds0-9_20260905.json`.

| Task | `benchmark_success` | mean LLM calls | mean replans | env/self collisions, slip |
|---|---|---|---|---|
| `cube_handover` | 10/10 | 2.0 | 0.0 | 0 / 0 / 0 |
| `lift_pot` | 10/10 | 3.0 | 1.0 | 0 / 0 / 0 |
| `stack_two_blocks` | 10/10 | 6.9 | 3.4 | 1 / 0 / 0 |

**Formal run at `b93f1bf` - 60/60** (`v2-full`, OpenAI `gpt-5.6-terra`,
seeds 0-9, the six new tasks, clean tree, 0 connection errors). Summary in
`results/openai-full_b93f1bf_seeds0-9_20260905.json`; full traces in
`outputs/openai-full_b93f1bf_seeds0-9_20260905`.

| Task | `benchmark_success` | mean LLM calls | mean replans | behavior quality |
|---|---|---|---|---|
| `vertical_cube_handover` | 10/10 | 2.0 | 0.0 | 10/10 |
| `lift_tray` | 10/10 | 1.0 | 0.0 | 10/10 |
| `pack_box` | 10/10 | 2.0 | 0.0 | 0/10 (slip metric, see gate note) |
| `rotate_valve` | 10/10 | 3.8 | 0.0 | 0/10 (env-collision metric, see gate note) |
| `pick_single_book` | 10/10 | 3.0 | 1.0 | 10/10 |
| `stack_single_book_shelf` | 10/10 | 3.0 | 0.6 | 9/10 |

What the LLM actually did: `lift_tray` is one `bimanual_grasp` (the tray
task's success does not require the lift - both holds plus no table
contact - and the verification lift already clears it); `pack_box` is two
`close_flap` requests, right arm then left; `rotate_valve` usually issues
a `grasp` on the wheel before `rotate` (harmless - the wheel cannot lift,
but the grasp verifies within tolerance); `pick_single_book`'s first
`lift` reports a short rise (the 4 kg book sags in the pinch) and the
replan repeats it to the benchmark height; `stack_single_book_shelf` goes
`grasp -> place` directly in half the seeds and via a `lift`/`transport`
in the rest. (A first attempt at this run from a `git worktree` at
`a4a47f2` failed before the first trial for the untracked-XML reason noted
under Environment, and was discarded.)

Together with the `42eb33b` run above that is **90/90 across all nine base
tasks** with the LLM planner, and 90/90 for the deterministic gate.

Not yet run: the other ablation rows in `METHOD_SPECS` (`v2-ik-only`,
`v2-fixed` via the API launcher, `v2-full-no-replan`, `v2-full-memory`)
and the two frozen v1 baselines (`v1-p22-independent`, `v1-p23-memory`,
which need `--v1-p22-root`/`--v1-p23-root` worktrees checked out at
`fb3876d`/`9400dda`, on `main`). The deterministic 10/10 gate results above are the
`v2-fixed` row's numbers in substance, but were produced by the gate
launcher, not the API matrix launcher.

## Experiment naming convention

Formal runs are named `{method}_{commit_short}_seeds{range}[_{date}]`, e.g.
`openai-full_42eb33b_seeds0-9_20260905` - not `v2`/`v3`/`v4`, which stop
meaning anything once there's a fourth revision. The commit SHA is the part
that actually disambiguates results; record it (`git rev-parse HEAD` plus a
dirty-tree flag) in every run's config, which `run_agentic_v2.py` already
does via `environment_metadata()`.

See `results/README.md` for what gets committed vs. stays in local
`outputs/`.
