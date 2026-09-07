"""Geometry-derived grasp and placement candidate generation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from roboeval.agentic_v2.state import collect_scene_state, vertical_extent
from roboeval.agentic_v2.types import (
    AllowedContactPolicy,
    AllowedContactRule,
    GraspCandidate,
    Pose,
    SceneState,
)


FINGER_PAD_OFFSET = np.array([0.0, 0.0, 0.1034])
MAXIMUM_APERTURE = 0.08
POT_HANDLE_BAR_LOCAL = {
    # Centers of the outer horizontal handle-bar collision meshes.
    "left": np.array([0.0014, 0.2338, 0.041]),
    "right": np.array([0.0019, -0.2457, 0.041]),
}
# Breakfast tray (props/breakfast_tray/wood_tray.xml), body frame: the two
# long rim walls are 2.6 cm thick along body X, 10 cm tall along body Y
# (top edge at y = +0.0933), centered at x = +/-0.266. Grasp points sit
# 3 cm below the top edge so the pads bite the upper part of the wall.
TRAY_RIM_LOCAL = (
    np.array([0.2658, 0.0633, 0.0]),
    np.array([-0.2663, 0.0633, 0.0]),
)
TRAY_RIM_THICKNESS = 0.026
# Objects taller than this (along world Z at their current pose) are
# grasped near their top instead of at their centroid so the palm never
# has to descend onto the object and the fingers need not reach so deep.
TALL_OBJECT_THRESHOLD = 0.10
# The palm face sits ~0.035 m above the pad centers, so the object's top can
# be at most that far above the pads or it hits the palm (measured on the
# standing rod: palm-rod contact at a 0.035 inset, 0.3 mm deep). Keep the
# pads a little less than that below the top so the palm clears by ~7 mm.
TALL_OBJECT_TOP_INSET = 0.028
# Thin, flat objects too wide to pinch from above (books lying on a
# counter) are pinched across their thickness at an overhanging edge.
FLAT_OBJECT_MAX_THICKNESS = 0.045
EDGE_GRASP_MINIMUM_OVERHANG = 0.025
EDGE_GRASP_PAD_INSET = 0.013
# Deepest pad inset before the object's edge meets the palm (measured on
# the book: a 4.2 cm inset just grazes it).
EDGE_GRASP_DEEP_INSET = 0.042


class CandidateGenerator:
    def __init__(self, env: Any) -> None:
        self.env = env

    def grasp_candidates(
        self,
        object_name: str,
        side: str,
        state: SceneState | None = None,
    ) -> tuple[GraspCandidate, ...]:
        state = state or collect_scene_state(self.env)
        if object_name not in state.objects:
            raise ValueError(f"unknown task object {object_name!r}")
        if side not in state.robot.arms:
            raise ValueError(f"unknown arm side {side!r}")
        if state.objects[object_name].fixed:
            return ()
        if object_name == "kitchenpot":
            return self._pot_handle_candidates(side, state)
        if object_name == "tray":
            return self._tray_rim_candidates(side, state)
        if object_name.startswith("valve"):
            return self._valve_wheel_candidates(object_name, side, state)
        candidates = list(self._top_down_candidates(object_name, side, state))
        if not candidates:
            candidates.extend(self._edge_candidates(object_name, side, state))
        return tuple(sorted(candidates, key=lambda candidate: candidate.score))

    def _top_down_candidates(
        self,
        object_name: str,
        side: str,
        state: SceneState,
    ) -> tuple[GraspCandidate, ...]:
        obj = state.objects[object_name]
        center = np.asarray(obj.pose.position)
        # Aperture must be checked against the object's fixed physical size,
        # not the live world AABB, which inflates as soon as the object
        # tilts even slightly and would spuriously reject a fitting axis.
        size = np.asarray(obj.canonical_size)
        # canonical_size lives in the object's own body frame; gap_axis below
        # is a world-frame direction, so it must be rotated into the body
        # frame before comparing against size - otherwise this is only
        # correct by accident for objects that happen to be world-aligned.
        object_rotation = np.asarray(obj.pose.as_matrix())[:3, :3]
        height = vertical_extent(obj)
        grasp_center = center.copy()
        if height > TALL_OBJECT_THRESHOLD:
            grasp_center[2] = center[2] + height / 2.0 - TALL_OBJECT_TOP_INSET
        candidates: list[GraspCandidate] = []
        for index, yaw in enumerate((-np.pi / 2.0, np.pi / 2.0, 0.0, np.pi)):
            rotation = Rotation.from_euler("xyz", (np.pi, 0.0, yaw))
            gap_axis = rotation.apply((0.0, 1.0, 0.0))
            gap_axis_local = object_rotation.T @ gap_axis
            required_aperture = float(np.dot(np.abs(gap_axis_local), size) + 0.012)
            if required_aperture > MAXIMUM_APERTURE:
                continue
            approach_axis = rotation.apply((0.0, 0.0, 1.0))
            wrist = grasp_center - rotation.apply(FINGER_PAD_OFFSET)
            pregrasp = wrist - approach_axis * 0.08
            grasp_pose = self._pose(wrist, rotation)
            pregrasp_pose = self._pose(pregrasp, rotation)
            candidates.append(
                GraspCandidate(
                    name=f"{object_name}_{side}_top_{index}",
                    object_name=object_name,
                    side=side,
                    pregrasp_pose=pregrasp_pose,
                    grasp_pose=grasp_pose,
                    approach_axis=tuple(approach_axis),
                    required_aperture=required_aperture,
                    contact_policy=self._grasp_policy(object_name, side),
                    score=self._score(pregrasp_pose, state.robot.arms[side].ee_pose),
                )
            )
        return tuple(sorted(candidates, key=lambda candidate: candidate.score))

    def _edge_candidates(
        self,
        object_name: str,
        side: str,
        state: SceneState,
    ) -> tuple[GraspCandidate, ...]:
        """Horizontal-hand pinch across the thickness of a thin, flat object
        at the edge where it overhangs its support (fingers above and below
        the overhang). Only offered when no top-down axis fits and the
        support's front edge is known."""

        obj = state.objects[object_name]
        thickness = vertical_extent(obj)
        if thickness > FLAT_OBJECT_MAX_THICKNESS:
            return ()
        support_front_x = self._support_front_x()
        if support_front_x is None:
            return ()
        center = np.asarray(obj.aabb_center, dtype=float)
        size = np.asarray(obj.size, dtype=float)
        near_x = center[0] - size[0] / 2.0
        overhang = support_front_x - near_x
        if overhang < EDGE_GRASP_MINIMUM_OVERHANG:
            return ()
        inset = min(EDGE_GRASP_PAD_INSET, overhang - 0.010)
        pad_center = np.array([near_x + inset, center[1], center[2]])
        candidates: list[GraspCandidate] = []
        for index, (yaw, gap_sign) in enumerate(
            ((0.0, 1.0), (0.0, -1.0), (0.35, 1.0), (-0.35, 1.0), (0.35, -1.0), (-0.35, -1.0))
        ):
            approach = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            gap = np.array([0.0, 0.0, gap_sign])
            lateral = np.cross(gap, approach)
            rotation = Rotation.from_matrix(np.column_stack((lateral, gap, approach)))
            wrist = pad_center - rotation.apply(FINGER_PAD_OFFSET)
            pregrasp = wrist - approach * 0.08
            grasp_pose = self._pose(wrist, rotation)
            pregrasp_pose = self._pose(pregrasp, rotation)
            candidates.append(
                GraspCandidate(
                    name=f"{object_name}_{side}_edge_{index}",
                    object_name=object_name,
                    side=side,
                    pregrasp_pose=pregrasp_pose,
                    grasp_pose=grasp_pose,
                    approach_axis=tuple(approach),
                    required_aperture=float(thickness + 0.012),
                    contact_policy=self._grasp_policy(object_name, side),
                    score=self._score(pregrasp_pose, state.robot.arms[side].ee_pose),
                    edge_grasp=True,
                )
            )
        return tuple(sorted(candidates, key=lambda candidate: candidate.score))

    def edge_overhang(self, object_name: str, state: SceneState) -> float | None:
        """How far the object's near (robot-facing) edge currently overhangs
        the support's front edge, or None when no support edge is known."""

        support_front_x = self._support_front_x()
        if support_front_x is None or object_name not in state.objects:
            return None
        obj = state.objects[object_name]
        near_x = float(obj.aabb_center[0] - obj.size[0] / 2.0)
        return float(support_front_x - near_x)

    def _support_front_x(self) -> float | None:
        """World x of the front (robot-facing) edge of the book counter."""

        shelf = getattr(self.env, "book_shelf", None)
        counter = getattr(shelf, "counter", None)
        if counter is None:
            return None
        bound = self.env.mojo.physics.bind(counter.mjcf)
        aabb = np.asarray(bound.aabb, dtype=float)
        rotation = np.asarray(bound.xmat, dtype=float).reshape(3, 3)
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=float,
        )
        corners = (signs * aabb[3:] + aabb[:3]) @ rotation.T + np.asarray(bound.xpos, dtype=float)
        return float(corners[:, 0].min())

    def _tray_rim_candidates(
        self,
        side: str,
        state: SceneState,
    ) -> tuple[GraspCandidate, ...]:
        tray = state.objects["tray"]
        matrix = tray.pose.as_matrix()
        rotation_world = matrix[:3, :3]
        points = [matrix @ np.append(local, 1.0) for local in TRAY_RIM_LOCAL]
        # The left arm takes whichever long rim currently lies at larger
        # world y (its own side), the right arm the other one.
        ordered = sorted(points, key=lambda point: point[1], reverse=True)
        handle_point = ordered[0] if side == "left" else ordered[1]
        thin_axis = rotation_world @ np.array([1.0, 0.0, 0.0])
        thin_axis = thin_axis / max(np.linalg.norm(thin_axis), 1e-9)
        approach = np.array([0.0, 0.0, -1.0])
        result: list[GraspCandidate] = []
        for orientation, sign in enumerate((1.0, -1.0)):
            gap = sign * thin_axis
            lateral = np.cross(gap, approach)
            rotation = Rotation.from_matrix(np.column_stack((lateral, gap, approach)))
            wrist = handle_point[:3] - rotation.apply(FINGER_PAD_OFFSET)
            pregrasp = wrist - approach * 0.08
            grasp_pose = self._pose(wrist, rotation)
            pregrasp_pose = self._pose(pregrasp, rotation)
            result.append(
                GraspCandidate(
                    name=f"tray_{side}_rim_{orientation}",
                    object_name="tray",
                    side=side,
                    pregrasp_pose=pregrasp_pose,
                    grasp_pose=grasp_pose,
                    approach_axis=tuple(approach),
                    required_aperture=TRAY_RIM_THICKNESS + 0.012,
                    contact_policy=self._grasp_policy("tray", side),
                    score=self._score(pregrasp_pose, state.robot.arms[side].ee_pose),
                )
            )
        return tuple(sorted(result, key=lambda candidate: candidate.score))

    def _valve_wheel_candidates(
        self,
        object_name: str,
        side: str,
        state: SceneState,
    ) -> tuple[GraspCandidate, ...]:
        """Top-down pinch across the valve's horizontal handwheel (6.2 cm
        across, 2.8 cm tall, axis vertical). The pads bite the upper part of
        the wheel so the fingertips stay clear of the valve body under it."""

        wheel = self._valve_wheel_center(object_name)
        if wheel is None:
            return ()
        wheel_center, wheel_size = wheel
        grasp_center = wheel_center + np.array([0.0, 0.0, 0.006])
        required_aperture = float(max(wheel_size[0], wheel_size[1]) + 0.012)
        if required_aperture > MAXIMUM_APERTURE:
            return ()
        result: list[GraspCandidate] = []
        for index, yaw in enumerate((-np.pi / 2.0, np.pi / 2.0, 0.0, np.pi)):
            rotation = Rotation.from_euler("xyz", (np.pi, 0.0, yaw))
            approach_axis = rotation.apply((0.0, 0.0, 1.0))
            wrist = grasp_center - rotation.apply(FINGER_PAD_OFFSET)
            pregrasp = wrist - approach_axis * 0.08
            grasp_pose = self._pose(wrist, rotation)
            pregrasp_pose = self._pose(pregrasp, rotation)
            result.append(
                GraspCandidate(
                    name=f"{object_name}_{side}_wheel_{index}",
                    object_name=object_name,
                    side=side,
                    pregrasp_pose=pregrasp_pose,
                    grasp_pose=grasp_pose,
                    approach_axis=tuple(approach_axis),
                    required_aperture=required_aperture,
                    contact_policy=self._grasp_policy(object_name, side),
                    score=self._score(pregrasp_pose, state.robot.arms[side].ee_pose),
                )
            )
        return tuple(sorted(result, key=lambda candidate: candidate.score))

    def _valve_wheel_center(self, object_name: str) -> tuple[np.ndarray, np.ndarray] | None:
        valves = getattr(self.env, "valves", None)
        if not valves:
            return None
        try:
            index = int(object_name.split("_", 1)[1])
            valve = valves[index]
        except (IndexError, ValueError):
            return None
        wheel = getattr(valve, "valve", None)
        if wheel is None:
            return None
        bound = self.env.mojo.physics.bind(wheel.mjcf)
        aabb = np.asarray(bound.aabb, dtype=float)
        rotation = np.asarray(bound.xmat, dtype=float).reshape(3, 3)
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=float,
        )
        corners = (signs * aabb[3:] + aabb[:3]) @ rotation.T + np.asarray(bound.xpos, dtype=float)
        minimum, maximum = corners.min(axis=0), corners.max(axis=0)
        return (minimum + maximum) / 2.0, maximum - minimum

    def _pot_handle_candidates(
        self,
        side: str,
        state: SceneState,
    ) -> tuple[GraspCandidate, ...]:
        pot = state.objects["kitchenpot"]
        pot_matrix = pot.pose.as_matrix()
        result: list[GraspCandidate] = []
        handle_point = pot_matrix @ np.append(POT_HANDLE_BAR_LOCAL[side], 1.0)
        inward = np.array([0.0, -1.0 if side == "left" else 1.0, 0.0])
        gap = np.array([0.0, 0.0, 1.0])
        lateral = np.cross(gap, inward)
        base = np.column_stack((lateral, gap, inward))
        for orientation, matrix in enumerate(
            (base, np.column_stack((-lateral, -gap, inward)))
        ):
            rotation = Rotation.from_matrix(matrix)
            approach_axis = rotation.apply((0.0, 0.0, 1.0))
            wrist = handle_point[:3] - rotation.apply(FINGER_PAD_OFFSET)
            pregrasp = wrist - approach_axis * 0.07
            grasp_pose = self._pose(wrist, rotation)
            pregrasp_pose = self._pose(pregrasp, rotation)
            result.append(
                GraspCandidate(
                    name=f"kitchenpot_{side}_side_handle_{orientation}",
                    object_name="kitchenpot",
                    side=side,
                    pregrasp_pose=pregrasp_pose,
                    grasp_pose=grasp_pose,
                    approach_axis=tuple(approach_axis),
                    required_aperture=0.037,
                    contact_policy=self._grasp_policy("kitchenpot", side),
                    score=self._score(pregrasp_pose, state.robot.arms[side].ee_pose),
                )
            )
        return tuple(sorted(result, key=lambda candidate: candidate.score))

    @staticmethod
    def _grasp_policy(object_name: str, side: str) -> AllowedContactPolicy:
        return AllowedContactPolicy(
            rules=(
                AllowedContactRule(f"robot:{side}:finger", f"object:{object_name}"),
                # A fingertip grazing the support surface while closing on a
                # short object resting on it is normal tabletop picking, not
                # a crash (reproduced on a 4cm block at -0.1mm, blocking an
                # otherwise-correct regrasp). Tolerate it at a few mm only;
                # wrist/link-table contact stays forbidden.
                AllowedContactRule(
                    f"robot:{side}:finger", "scene:*table*", penetration_tolerance=0.004
                ),
            ),
            penetration_tolerance=0.008,
        )

    @staticmethod
    def _pose(position: np.ndarray, rotation: Rotation) -> Pose:
        xyzw = rotation.as_quat()
        return Pose(tuple(position), (xyzw[3], xyzw[0], xyzw[1], xyzw[2]))

    @staticmethod
    def _score(target: Pose, current: Pose) -> float:
        distance = float(np.linalg.norm(np.asarray(target.position) - np.asarray(current.position)))
        target_rotation = Rotation.from_matrix(target.as_matrix()[:3, :3])
        current_rotation = Rotation.from_matrix(current.as_matrix()[:3, :3])
        return distance + 0.05 * float((current_rotation.inv() * target_rotation).magnitude())
