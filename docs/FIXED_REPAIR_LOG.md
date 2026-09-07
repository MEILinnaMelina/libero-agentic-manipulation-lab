# Fixed semantic repair log

Scope: tasks 1, 2, 3, 4, 5, 6, 8, 9. Only deterministic fixed semantic experiments; no API calls. All tests retain official initial states, physics, action interface and success predicates, with original 600-action budget. Each attempt has a unique directory, action/state trace, result and video. This is development repair, not a held-out benchmark.

## Baseline and diagnostics

- Baseline `dev-fixed-v7`: each failed task 0/3. Task 1 cream-cheese grasp rejected due to arm/milk collision; butter grasp lacked two-finger contact. Many later `grasp_slip` reports were cascading precondition failures after the initial grasp failed, not separate physical slips.
- Added `scripts/debug_fixed.py`: always method=fixed, removes API key from child process, denies socket connections, preserves every attempt, stops at first skill failure to expose root cause. No change to official success evaluation.

## Shared defects found and repaired

- robosuite 1.4 reports grip-site position but wrist-body quaternion. OSC/IK operate on the grip site. Normalize quaternion to the actual grip-site world rotation, retaining raw wrist quaternion separately.
- Panda finger closing direction is local grip-site x, not y; candidate selection now uses the correct axis and both equivalent yaw directions. IK convergence tightened before collision classification.
- Some official initial objects (e.g. milk) fall after the five reset-settling actions. Add 12 hold actions inside the original 600-action budget before grasp planning, and compute targets from refreshed state.
- Thin-box descent previously stopped 1–2 cm too high, producing weak fingertip contact. Use 3 mm positioning tolerance for thin boxes. Two-finger detection includes finger mesh contacts as well as pads; actual lift displacement and hold are still checked.
- Tall-object AABB centers placed the palm against upper geometry. Use upper-body grasps; moka pots use the small top knob's actual collision geometry. Hollow bowls use a rim pinch.

## Task 1: cream cheese and butter in basket

- v01–v11 retain failed development attempts. Milk blocked the cheese approach; fixed plan now explicitly clears nearby tall obstacles to a free workspace point using physical grasp/move/release.
- **v12 state 0 SUCCESS, 452/600 actions, 0 API calls**, no skill failures. Both official containment goals satisfied. Video: `runs/repair-t01-v12/task01_state000/video.mp4`.

## Task 2: turn on stove, place moka pot

- v01 selected tiny base collision geometry rather than the raised rotary fin. Turn-on now grasps the fin, rotates about the actual hinge axis, and checks the official on predicate.
- v02 turned stove on but pot spout blocked whole-body grasp. Select top-knob collision geometry for the pot.
- **v03 state 0 SUCCESS, 216/600 actions, 0 API calls**, no skill failures. Video: `runs/repair-t02-v03/task02_state000/video.mp4`.

## Task 3: bowl inside bottom drawer, close drawer

- Select the forward handle bar along the drawer's actual slide axis, with matching grip orientation. Opening succeeds.
- v04 physically grasps bowl rim and places bowl inside: official containment true. Closing fails because top-down palm hits cabinet front. Testing a horizontal closed-finger push with approach retreat, preserving physical dynamics.
- v05–v08 exposed side-push reachability and wine-rack collision. A 0.45 rad forward tilt and shorter 5.5 cm pre-push offset clear both obstacles.
- **v09 state 0 SUCCESS, 333/600 actions, 0 API calls**. Video: `runs/repair-t03-v09/task03_state000/video.mp4`.

## Task 4: mugs on two plates

- Shared grip-site rotation correction and rim pinch solve both mug grasps.
- **v01 state 0 SUCCESS, 272/600 actions, 0 API calls**. Video: `runs/repair-t04-v01/task04_state000/video.mp4`.

## Task 5: book in back caddy compartment

- v01 grasps/lifts/transports but book's wide horizontal axis does not fit the narrow slot. v02 shows that a fixed 90-degree turn is insufficient because initial book yaw is oblique.
- Align the actual principal horizontal axis of the largest collision box with the long axis of the goal compartment.
- **v03 state 0 SUCCESS, 133/600 actions, 0 API calls**. Video: `runs/repair-t05-v03/task05_state000/video.mp4`.

## Task 6: mug on plate, pudding beside plate

- Shared rim grasp and corrected geometric control solve the first mug failure; pudding grasp/place also completes.
- **v01 state 0 SUCCESS, 254/600 actions, 0 API calls**. Video: `runs/repair-t06-v01/task06_state000/video.mp4`.

## Task 8: two moka pots on stove

- Existing on-state is now checked before touching the switch.
- v01 places first pot but second carried pot collides with it during transport. Raise pot transit clearance and allocate separate ±4 cm x placements on the shared stove region; pot handles no longer force fallback to the same center placement.
- **v02 state 0 SUCCESS, 297/600 actions, 0 API calls**. Video: `runs/repair-t08-v02/task08_state000/video.mp4`.

## Task 9: mug in microwave, close door

- v01 top-down placement hits microwave roof. Use staged front entry, cavity floor height and actual interior horizontal offset.
- v02–v06 explored slanted rim pinches; poor contact or rim-hooking during release dragged the mug back out. These are retained failures, not successes.
- v07–v08 use the actual protruding handle geometry, lower the upright mug outside the opening, then insert horizontally. After release and withdrawal, official containment remains true.
- v08–v10 expose wrist joint limits with door grasp yaw and small compliant closing lag. Use reachable +90-degree grasp yaw and a bounded final closing push.
- **v11 state 0 SUCCESS, 352/600 actions, 0 API calls**. Video: `runs/repair-t09-v11/task09_state000/video.mp4`.

## Unified validation

- Nine unit tests pass, including a new regression for wrist/body versus grip-site rotation and the existing first-official-success recording contract.
- Next run `fixed-repair-validation-v1`: all eight repaired tasks, dev states 0/1/2, sequentially, one immutable code hash, no network, original 600-step budget. Final per-task rates must come from this unified run, not selected successful attempts above.
- These repairs change the prior frozen implementation. The earlier interrupted paid formal run remains preserved and must not be resumed/mixed with this version.
- `fixed-repair-validation-v1`: **23/24** success. Tasks 2/3/4/5/6/8/9 each 3/3. Task 1 state 1 failed because milk was 12.1 cm from the target and fell just outside the 12 cm clearance test.
- Extend the clearance radius to 12.5 cm. `repair-t01-v13` state 1 **SUCCESS, 574/600 actions**, after explicitly moving orange juice and milk. All failures preserved.
- Final unified confirmation run: `fixed-repair-validation-v2`, all eight tasks × dev states 0/1/2, unchanged 600-step budget, network disabled. The full source/config/scripts/tests snapshot is retained in its `source.zip`.
- **FINAL: 24/24 SUCCESS, every task 3/3.** Steps by task/state 0,1,2: task 1 = 452/574/296; task 2 = 230/236/233; task 3 = 333/327/348; task 4 = 272/261/263; task 5 = 133/132/134; task 6 = 254/257/263; task 8 = 297/302/301; task 9 = 352/361/353. All final official predicates true; zero skill failures and zero API/token usage in the final run.
- All 24 dual-camera MP4 files decoded successfully, with frame count equal to policy actions plus the five recorded settling actions. Final frames match the first official success action. Video index and Chinese repair summary: `reports/fixed_repair/SUMMARY.md`.
- Independently replayed task 1 state 1: 574 steps, success, 579 frames; maximum simulator-state difference from source recording **0.0**. Nine tests passed. Vendor source integrity verification passed. No experiment processes remain after completion.
- Full repair history contains 91 recorded development attempts, including all failures and both unified validation versions. `docs/fixed_repair_attempts.jsonl` is rebuilt from saved results; the reported 24/24 uses only the final unified run, never selected successes from earlier versions.
