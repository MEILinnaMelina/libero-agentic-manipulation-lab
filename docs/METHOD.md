# Agentic v2 → LIBERO adaptation

This is a simulator-state-assisted, BDDL-goal-assisted adaptation, not an unchanged RoboEval reproduction or RGB-only policy.

| Original component | Adaptation |
|---|---|
| `prompts.py`, strict semantic request, deterministic low-level authority | `planner.py`: same observe/one-skill/feedback boundary; single Panda arm; versioned schema; mechanism skills added |
| `llm_planner.py`, Responses API | Standard-library HTTPS JSON transport so Python 3.8 needs no modern SDK; exact requested/returned model saved; no retries/fallback |
| `replanner.py`, fixed vs online, failure ablation | `runner.py`: episode-local history, last-six feedback entries, maximum three failure replans; no-replan stops at first failure |
| dm_control, dual-arm multi-start IK and contact policy | `motion.py`: private MuJoCo MjData, world-frame Jacobian IK, joint limits, forbidden contact rejection |
| geometric grasp/placement candidates | `scene.py`, `skills.py`: collision geometry bounds, top-down axis candidates, region-derived placement and target clearances |
| grasp/lift/transport/place, flap/valve design | Single-arm OSC execution; drawer slide, appliance hinge arc, actual stove switch joint; joint properties read from LIBERO definitions |
| monitoring and artifacts | Per-skill contact/pose/placement checks; full action and simulator-state recording, two-camera video, official success, explicit error taxonomy |

Only this package and vendored LIBERO are imported at runtime. The old RoboEval package is archival provenance, never imported; its dm_control/mojo stack is not loaded. Original source licenses remain in vendor. Files adapted from RoboEval retain that project's Apache-2.0 terms; LIBERO retains MIT terms.

## Information visibility

- LLM: task language, **BDDL goal predicates**, named object/site world poses and approximate collision AABBs, articulated joint state/range/axis/anchor, robot joint and end-effector state, gripper aperture, contact pairs, verified held objects, per-goal predicate truth, raw official success, remaining steps, six recent decisions/feedback. No camera input; videos are evidence only.
- Deterministic skill and motion layer: all above plus full collision geometry, mechanism properties, robot Jacobians/joint limits and a private simulation-state copy. It chooses every numeric action.
- Evaluator: official initial states, official `check_success`, all diagnostics and runtime artifacts. Ground truth success is never inferred from LLM text or custom skill scores.

BDDL-informed fixed plans and region/goal usage are task-family knowledge, disclosed explicitly. No learned perception, training, demonstration downloads, cross-episode memory, object teleportation, predicate modification, or reward modification is used.

## Protocol

Task order 0, development state indices [0,1,2], formal indices [3..49], checked for duplicate content. Each task has 50 available official states, so only 47 unused states remain. Five upstream all-zero stability actions are outside the 600 policy-action budget; the gripper input in those five actions is zero. Environment horizon is 605. Source `bddl_base_domain.py` replaces `done` with `_check_success`; the runner also queries `check_success` explicitly. Outcomes distinguish success, budget exhaustion, planner finish, API, simulation and interrupted execution.

Resumption identity includes method, split, full schedule and code/config/manifest hashes. No automatic infrastructure retry. Completed failures never rerun under the same ID; started but unfinished attempts become interrupted failures. Missing scheduled episodes remain visible in the denominator. Run IDs with differing methods, splits, initial states or frozen code cannot be pooled.

## Platform deviations

Python 3.8.13, robosuite 1.4.0, MuJoCo 2.3.7. Inference subset of LIBERO dependencies only; no training stack. Three Windows-only robosuite packaging fixes: writable project logger path; DLL resolution from the installed MuJoCo package; GLFW selection on Windows rather than forced EGL. Exact before/after hashes and replacements are in `provenance/windows_patches.json`. These do not alter physics, controllers, assets or benchmark predicates.

The runtime uses stdlib JSON HTTPS and therefore does not need a separate modern SDK process (T07 condition does not arise). Each episode creates fresh planner memory; no Responses previous_response_id is used and storage is disabled.

## Validated memory configuration

The final adapter supplies `nstack=2000000` as a MuJoCo compilation workspace request. MuJoCo may round this up internally: the validated task-0 model reports 40,894,464 doubles, while its observed maximum stack use is 45,646. All physical model parameters, contact/constraint capacity settings, controllers and official states remain unchanged. The 262-action manipulation validation produced **bitwise-identical actions and all simulator states** before/after this allocation change (`reports/memory_equivalence.json`). Numerical/capacity warnings abort an episode as a simulation error. Every episode records compiled stack size and peak usage. BLAS/OpenMP thread counts are one per evaluation process to prevent large thread workspace reservations.

Carried-object collision validation transforms only the held object's free joint in the private MjData according to its observed attachment transform. It checks carried-object/environment contacts and samples Cartesian path segments at <=4 cm spacing. Live qpos is asserted unchanged by IK search. This remains a discrete candidate checker, not a proof of collision-free continuous dynamics; tracking error and slips can still occur during real OSC execution.

## Fixed semantic repair verification

The 2026-09-08 repair workflow uses no model/API: a deterministic BDDL-informed skill sequence is constructed before rollout, optionally prepending explicit clearance of nearby tall objects. There is no semantic replanning during execution. Geometry-aware skills may use feedback internally. The first failed skill terminates diagnostic episodes, avoiding cascaded precondition failures. Twelve additional grasp-settling actions count toward the unchanged 600-action budget; the five official reset-settling actions remain unchanged.

The adapter replaces the stock wrist-body quaternion observation with the actual grip-site quaternion to match OSC and private IK; the original quaternion is retained as `robot0_wrist_quat`. Grasp verification requires contact on both finger groups and successful physical lift. The official episode predicates remain unchanged. All attempted initial states are development indices 0/1/2; improved results are development validation, not fresh held-out performance or GPT planning results. See `reports/fixed_repair/SUMMARY.md` for the final run and video audit.
