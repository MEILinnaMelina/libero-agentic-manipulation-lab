"""Verified lid-flap closing by a fingertip sweep around the hinge."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from roboeval.agentic_v2.executor import CLOSE_COMMAND
from roboeval.agentic_v2.skills.base import BaseSkill
from roboeval.agentic_v2.state import collect_scene_state
from roboeval.agentic_v2.types import (
    AllowedContactPolicy,
    AllowedContactRule,
    ConstraintSet,
    FailureCode,
    Pose,
    SkillName,
    SkillRequest,
    SkillResult,
)


# Wrist-site to closed-fingertip distance along the hand's approach axis.
FINGERTIP_OFFSET = 0.115
# Radius of the fingertip's arc around the hinge; the flap plate reaches
# ~0.21 m from its hinge, so 0.15 keeps the tip well inside the free edge.
SWEEP_RADII = (0.12, 0.15)
# Height above the first fingertip waypoint from which the sweep descends
# so the hand approaches the open flap from outside rather than swinging
# through the box on a straight joint path.
STAGING_HEIGHT = 0.10
SWEEP_STEP = 0.26
# RoboEval counts a flap closed when its normalized state is within 0.1 of
# 0, i.e. the hinge within ~10 degrees of its closed limit.
CLOSED_TOLERANCE = 0.10


class CloseFlapSkill(BaseSkill):
    def execute(self, request: SkillRequest) -> SkillResult:
        if request.skill is not SkillName.CLOSE_FLAP:
            raise ValueError("CloseFlapSkill received the wrong request")
        state = collect_scene_state(self.env)
        object_name = request.object_name
        box = getattr(self.env, "packing_box", None)
        if object_name not in state.objects or box is None or object_name != "packing_box":
            return self.failure(
                request, state, FailureCode.INVALID_REQUEST,
                f"{object_name!r} is not a box with hinged flaps", [],
            )
        sides = [side for side in ("left", "right") if side in request.roles]
        if not sides:
            if request.strategy in ("left", "right"):
                sides = [request.strategy]
            else:
                sides = ["right", "left"]
        reports = []
        attempts: list[dict[str, Any]] = []
        closed_now = []
        for side in sides:
            idle = "left" if side == "right" else "right"
            parked = self.park_idle_arm(idle)
            if parked is not None:
                reports.append(parked)
            flap = self._flap_for_side(box, side)
            if flap is None:
                return self.failure(
                    request, collect_scene_state(self.env), FailureCode.INVALID_REQUEST,
                    f"no hinge on the {side} arm's side", reports,
                )
            if abs(self._normalized(box, flap["index"])) < CLOSED_TOLERANCE:
                closed_now.append(side)
                continue
            ok, evidence = self._sweep(object_name, side, box, flap, reports)
            attempts.append(evidence)
            if not ok:
                code = FailureCode(evidence.get("failure") or FailureCode.POSTCONDITION_FAILED.value)
                return self.failure(
                    request, collect_scene_state(self.env), code,
                    f"{side} flap sweep failed at {evidence.get('stage')}", reports,
                    {"attempts": attempts},
                )
            closed_now.append(side)
        final = collect_scene_state(self.env)
        return SkillResult(
            request, True, f"closed flap(s) with {closed_now}", final,
            execution_reports=tuple(reports),
            diagnostics={"attempts": attempts, "box_state": [float(v) for v in np.asarray(box.get_state()).reshape(-1)]},
        )

    def _sweep(self, object_name, side, box, flap, reports):
        physics = self.env.mojo.physics
        anchor = np.asarray(physics.bind(flap["joint"].mjcf).xanchor, dtype=float)
        axis = np.asarray(physics.bind(flap["joint"].mjcf).xaxis, dtype=float)
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        plate = np.asarray(physics.bind(flap["geom"].mjcf).xpos, dtype=float)
        # The joint anchor sits at one *end* of the hinge (x = 0.386 for the
        # left flap, mid-hinge for the right); push at the plate's center
        # along the hinge instead, or one arm is sent to a corner it can
        # only reach with its elbow at the limit.
        anchor = anchor + np.dot(plate - anchor, axis) * axis
        up = np.array([0.0, 0.0, 1.0])
        radial = plate - anchor
        radial -= np.dot(radial, axis) * axis
        outward = radial - np.dot(radial, up) * up
        outward = outward / max(np.linalg.norm(outward), 1e-9)
        flap_angle = float(np.arctan2(np.dot(radial, up), np.dot(radial, outward)))
        evidence: dict[str, Any] = {"side": side, "flap_angle_start": flap_angle}

        # Fingers closed: the sweep pushes with the closed fingertip.
        policy = AllowedContactPolicy(
            rules=(
                AllowedContactRule(f"robot:{side}:finger", f"object:{object_name}", penetration_tolerance=0.05),
                AllowedContactRule(f"robot:{side}:link", f"object:{object_name}", penetration_tolerance=0.012),
                AllowedContactRule(f"robot:{side}:finger", "scene:*table*", penetration_tolerance=0.004),
            ),
            penetration_tolerance=0.008,
        )
        constraints = ConstraintSet(allowed_contacts=policy)
        closed = self.context.executor.command_gripper(side, CLOSE_COMMAND, constraints=constraints)
        reports.append(closed)
        if not closed.success:
            evidence.update({"stage": "close_fingers", "failure": (closed.failure_code or FailureCode.EXECUTION_DIVERGED).value})
            return False, evidence

        for radius in SWEEP_RADII:
            evidence["radius"] = radius
            angles = self._sweep_angles(flap_angle)
            chosen = None
            failed_stage = None
            for index, angle in enumerate(angles):
                fingertip = anchor + radius * (np.cos(angle) * outward + np.sin(angle) * up)
                # Never push below the closed flap's resting plane.
                fingertip[2] = max(fingertip[2], anchor[2] + 0.006)
                if index == 0:
                    staged = self._stage_above(
                        object_name, side, fingertip, angle, outward, up, axis, constraints, reports,
                    )
                    if staged is not None:
                        chosen = staged
                orientations = self._hand_orientations(angle, outward, up, axis)
                # Prefer the roll that has worked so far, but allow the hand
                # to flip its roll mid-sweep if that one runs into a joint
                # limit (the two arms start in identical, not mirrored,
                # configurations, so one arm may need the other roll).
                orientation_options = list(enumerate(orientations))
                if chosen is not None:
                    orientation_options.sort(key=lambda item: item[0] != chosen)
                moved = None
                for option_index, rotation in orientation_options:
                    wrist = fingertip - rotation.apply((0.0, 0.0, FINGERTIP_OFFSET))
                    target = self._pose(wrist, rotation)
                    execution, ik_result, path_result = self.move(
                        name=f"{object_name}_{side}_flap_sweep_{index}",
                        targets={side: target},
                        constraints=constraints,
                        require_holds=False,
                        candidate_count=9,
                    )
                    if execution is not None:
                        reports.append(execution)
                    if execution is not None and execution.success:
                        moved = execution
                        chosen = option_index
                        break
                    failed_stage = self._evidence(f"sweep_{index}", execution, ik_result, path_result)
                evidence.setdefault("progress", []).append(
                    {"waypoint": index, "angle": round(float(angle), 3), "state": round(self._normalized(box, flap["index"]), 3), "moved": moved is not None}
                )
                if moved is None:
                    evidence.update(failed_stage or {"stage": f"sweep_{index}", "failure": FailureCode.PATH_BLOCKED.value})
                    break
                if abs(self._normalized(box, flap["index"])) < CLOSED_TOLERANCE and angle > np.pi / 2.0:
                    break
            normalized = self._normalized(box, flap["index"])
            evidence["state_after"] = normalized
            if abs(normalized) < CLOSED_TOLERANCE:
                evidence.update({"stage": "done", "failure": None})
                self._clear(side, reports, constraints)
                return True, evidence
            # Retreat before retrying with the other radius.
            self._clear(side, reports, constraints)
        evidence.setdefault("stage", "sweep")
        evidence["failure"] = evidence.get("failure") or FailureCode.POSTCONDITION_FAILED.value
        return False, evidence

    def _stage_above(self, object_name, side, fingertip, angle, outward, up, axis, constraints, reports):
        """Move to a point above and outside the first waypoint first; returns
        the orientation option index that worked, or None."""

        # The open flap reaches ~0.21 m out from its hinge at ~20 degrees up:
        # come in beyond its free edge, drop below the edge, then slide in
        # under the plate to the first arc point.
        outside_above = fingertip + 0.18 * outward + np.array([0.0, 0.0, STAGING_HEIGHT])
        outside_below = fingertip + 0.18 * outward + np.array([0.0, 0.0, -0.02])
        for option_index, rotation in enumerate(self._hand_orientations(angle, outward, up, axis)):
            ok = True
            for stage_index, tip in enumerate((outside_above, outside_below)):
                wrist = tip - rotation.apply((0.0, 0.0, FINGERTIP_OFFSET))
                execution, _, _ = self.move(
                    name=f"{object_name}_{side}_flap_stage_{stage_index}",
                    targets={side: self._pose(wrist, rotation)},
                    constraints=constraints,
                    require_holds=False,
                    candidate_count=9,
                )
                if execution is not None:
                    reports.append(execution)
                if execution is None or not execution.success:
                    ok = False
                    break
            if ok:
                return option_index
        return None

    def _clear(self, side, reports, constraints) -> None:
        """Lift and pull back toward the arm's own side so the other arm has
        the box top to itself. The fingertip is still resting on the flap it
        just closed, so the push contact policy stays in force."""

        current = collect_scene_state(self.env).robot.arms[side].ee_pose
        sign = 1.0 if side == "left" else -1.0
        target = Pose(
            (current.position[0], current.position[1] + sign * 0.18, current.position[2] + 0.10),
            current.quaternion_wxyz,
        )
        execution, _, _ = self.move(
            name=f"flap_clear_{side}", targets={side: target},
            constraints=constraints, require_holds=False,
        )
        if execution is not None:
            reports.append(execution)
        if execution is None or not execution.success:
            retreat = self.retreat(side, distance=0.10, constraints=constraints)
            if retreat is not None:
                reports.append(retreat)

    @staticmethod
    def _sweep_angles(flap_angle: float) -> list[float]:
        start = min(flap_angle - 0.40, 0.0)
        end = np.pi - 0.05
        count = int(np.ceil((end - start) / SWEEP_STEP)) + 1
        return [float(value) for value in np.linspace(start, end, count)]

    @staticmethod
    def _hand_orientations(angle, outward, up, hinge_axis) -> list[Rotation]:
        """Hand trailing the fingertip along the arc: the fingers point along
        the direction of motion (up-and-inward at the start, inward at the
        top, down at the end) so the hand body stays in the space the flap
        has already left rather than in the flap's plane - a hand pointing
        straight down sits in the open flap's plane and cannot get under it
        (measured: -8.7 mm palm/flap penetration at the first waypoint).
        The finger-closing axis stays parallel to the hinge so the flat side
        of the closed fingers does the pushing. Upward pitch is clamped so
        the wrist never has to flip fully under the hand."""

        tangent = -np.sin(angle) * outward + np.cos(angle) * up
        pitch = float(np.arctan2(np.dot(tangent, up), np.dot(tangent, -outward)))
        pitch = min(pitch, np.deg2rad(40.0))
        approach = np.cos(pitch) * (-outward) + np.sin(pitch) * up
        approach = approach / max(np.linalg.norm(approach), 1e-9)
        result = []
        for sign in (1.0, -1.0):
            gap = sign * hinge_axis
            lateral = np.cross(gap, approach)
            result.append(Rotation.from_matrix(np.column_stack((lateral, gap, approach))))
        return result

    @staticmethod
    def _flap_for_side(box, side: str):
        """Pair each hinge with the flap plate it carries and hand the one on
        the arm's own side (smaller world y for the right arm) to that arm."""

        joints = list(getattr(box, "_joints", ()))
        geoms = [getattr(box, "flap_1", None), getattr(box, "flap_2", None)]
        if len(joints) < 2 or any(geom is None for geom in geoms):
            return None
        physics = box._mojo.physics
        entries = []
        for index, joint in enumerate(joints[:2]):
            anchor = np.asarray(physics.bind(joint.mjcf).xanchor, dtype=float)
            geom = min(
                geoms,
                key=lambda item: np.linalg.norm(np.asarray(physics.bind(item.mjcf).xpos, dtype=float) - anchor),
            )
            entries.append({"index": index, "joint": joint, "geom": geom, "anchor_y": float(anchor[1])})
        entries.sort(key=lambda item: item["anchor_y"])
        return entries[0] if side == "right" else entries[-1]

    @staticmethod
    def _normalized(box, index: int) -> float:
        state = np.asarray(box.get_state(), dtype=float).reshape(-1)
        return float(state[index]) if index < state.size else 0.0

    @staticmethod
    def _pose(position: np.ndarray, rotation: Rotation) -> Pose:
        xyzw = rotation.as_quat()
        return Pose(tuple(position), (xyzw[3], xyzw[0], xyzw[1], xyzw[2]))

    @staticmethod
    def _evidence(stage, execution, ik_result, path_result):
        if execution is not None:
            code = execution.failure_code
        elif path_result is not None:
            code = path_result.report.failure_code
        else:
            code = ik_result.report.failure_code
        return {
            "stage": stage,
            "failure": code.value if code else None,
            "ik_accepted": len(ik_result.accepted),
            "ik_rejected": len(ik_result.rejected),
            "path_attempted": path_result.attempted_paths if path_result else 0,
        }
