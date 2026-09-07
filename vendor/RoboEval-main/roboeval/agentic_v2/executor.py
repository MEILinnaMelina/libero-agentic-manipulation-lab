"""Absolute joint action adapter and bounded monitored executor."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from roboeval.agentic_v2.evaluator import assess_behavior_quality, benchmark_success, subtask_progress
from roboeval.agentic_v2.monitor import ExecutionMonitor
from roboeval.agentic_v2.motion.collision_checker import CollisionChecker
from roboeval.agentic_v2.state import collect_scene_state
from roboeval.agentic_v2.types import (
    ConstraintSet,
    ExecutionReport,
    FailureCode,
    MonitorEvent,
    MotionPlan,
    Pose,
    TrajectoryPoint,
)


OPEN_COMMAND = 0.0
CLOSE_COMMAND = 1.0


class JointActionAdapter:
    """Map 14 arm joints and named grippers to RoboEval's action vector."""

    def __init__(self, env: Any) -> None:
        mode = env.action_mode
        if getattr(mode, "ee", True) or not getattr(mode, "absolute", False):
            raise ValueError("Agentic v2 requires absolute non-EE joint position mode")
        self.env = env
        self.arm_count = len(env.robot.limb_actuators)
        self.sides = [side.name.lower() for side in env.robot.grippers]
        if env.action_space.shape != (self.arm_count + len(self.sides),):
            raise ValueError("unexpected RoboEval action layout")
        last = np.asarray(env.action, dtype=float)
        self.gripper_commands = {
            side: float(last[self.arm_count + index])
            for index, side in enumerate(self.sides)
        }
        # Last arm joint targets actually sent. Position servos settle a few
        # mm below their target under gravity; any plan that re-anchors an
        # idle arm at its *measured* joints re-commands that sagged pose and
        # the arm ratchets downward plan after plan (measured: an idle hand
        # dropping 7 cm over ~250 steps until it rested on the scene).
        self.last_commanded: np.ndarray | None = None

    def set_gripper(self, side: str, command: float) -> None:
        if side not in self.gripper_commands:
            raise ValueError(f"unknown gripper side {side!r}")
        if command not in (OPEN_COMMAND, CLOSE_COMMAND):
            raise ValueError("gripper command must be 0 (open) or 1 (close)")
        self.gripper_commands[side] = float(command)

    def build(self, joint_positions: Any) -> np.ndarray:
        joints = np.asarray(joint_positions, dtype=float)
        if joints.shape != (self.arm_count,) or not np.all(np.isfinite(joints)):
            raise ValueError(f"expected {self.arm_count} finite arm joints")
        action = np.concatenate(
            (joints, [self.gripper_commands[side] for side in self.sides])
        ).astype(np.float32)
        if np.any(action < self.env.action_space.low) or np.any(action > self.env.action_space.high):
            raise ValueError("action exceeds RoboEval action bounds")
        self.last_commanded = joints.copy()
        return action

    def hold_targets(self, measured: Any) -> np.ndarray:
        """Joint targets that keep the robot where it was *told* to be."""

        measured = np.asarray(measured, dtype=float)
        if self.last_commanded is None or self.last_commanded.shape != measured.shape:
            return measured
        return self.last_commanded.copy()


class MonitoredExecutor:
    def __init__(
        self,
        env: Any,
        *,
        collision_checker: CollisionChecker | None = None,
        render: bool = False,
        frame_callback: Callable[[int, Any], None] | None = None,
        settle_steps: int = 5,
        convergence_tolerance: float = 0.03,
        maximum_convergence_steps: int = 40,
    ) -> None:
        self.env = env
        self.checker = collision_checker or CollisionChecker(env)
        self.monitor = ExecutionMonitor(self.checker)
        self.adapter = JointActionAdapter(env)
        self.render = render
        self.frame_callback = frame_callback
        self.settle_steps = int(settle_steps)
        # After the last planned point, keep commanding it until the joints
        # have actually arrived. A torque-limited joint (the 12 Nm wrist
        # pitch under a 4 kg book) lags the ramp by well under the 0.3 rad
        # tracking tolerance yet leaves the hand ~10 cm short if the plan
        # simply ends; given a moment it converges.
        self.convergence_tolerance = float(convergence_tolerance)
        self.maximum_convergence_steps = int(maximum_convergence_steps)

    def execute(
        self,
        plan: MotionPlan,
        *,
        gripper_commands: Mapping[str, float] | None = None,
        protected_objects: Mapping[str, Pose] | None = None,
        require_holds: bool = True,
        stop_condition: Callable[[Any], bool] | None = None,
        terminal_constraints: ConstraintSet | None = None,
        stop_on_success: bool = True,
    ) -> ExecutionReport:
        for side, command in (gripper_commands or {}).items():
            self.adapter.set_gripper(side, command)
        self.monitor.reset()
        events: list[MonitorEvent] = []
        executed = 0
        condition_reached = False
        info = self.env.get_info()
        final_state = collect_scene_state(self.env, info)
        points = list(plan.points)
        if not points:
            raise ValueError("cannot execute an empty motion plan")
        schedule = points + [points[-1]] * self.settle_steps
        for index, point in enumerate(schedule):
            event, info, final_state, stop_reached = self._execute_point(
                index,
                point,
                plan.constraints,
                protected_objects,
                require_holds,
                stop_condition,
                terminal_constraints,
            )
            executed += 1
            if event is not None:
                events.append(event)
                break
            if stop_reached:
                condition_reached = True
                break
            if stop_on_success and benchmark_success(final_state) >= 1.0:
                # Stopping early on task success is a deliberate end, not a
                # tracking failure - it must not trip the final-target check.
                condition_reached = True
                break
        extra = 0
        while (
            not events
            and not condition_reached
            and extra < self.maximum_convergence_steps
            and float(
                np.max(
                    np.abs(
                        np.asarray(final_state.robot.joint_positions)
                        - np.asarray(points[-1].joint_positions)
                    )
                )
            ) > self.convergence_tolerance
        ):
            event, info, final_state, stop_reached = self._execute_point(
                len(schedule) + extra,
                points[-1],
                plan.constraints,
                protected_objects,
                require_holds,
                stop_condition,
                terminal_constraints,
            )
            executed += 1
            extra += 1
            if event is not None:
                events.append(event)
                break
            if stop_reached or (stop_on_success and benchmark_success(final_state) >= 1.0):
                condition_reached = True
                break
        final_error = float(
            np.max(
                np.abs(
                    np.asarray(final_state.robot.joint_positions)
                    - np.asarray(points[-1].joint_positions)
                )
            )
        )
        if (
            not events
            and not condition_reached
            and final_error > self.monitor.config.tracking_tolerance
        ):
            events.append(
                MonitorEvent(
                    executed,
                    FailureCode.EXECUTION_DIVERGED,
                    "final joint target was not reached",
                    {"tracking_error": final_error},
                )
            )
        return ExecutionReport(
            success=not events,
            plan_name=plan.name,
            executed_points=executed,
            final_state=final_state,
            events=tuple(events),
            failure_code=events[-1].code if events else None,
            benchmark_success=benchmark_success(final_state),
            subtask_progress=subtask_progress(final_state),
            behavior_quality=assess_behavior_quality(final_state),
        )

    def _execute_point(
        self,
        index: int,
        point: TrajectoryPoint,
        constraints: ConstraintSet,
        protected_objects: Mapping[str, Pose] | None,
        require_holds: bool,
        stop_condition: Callable[[Any], bool] | None,
        terminal_constraints: ConstraintSet | None,
    ) -> tuple[MonitorEvent | None, dict[str, Any], Any, bool]:
        preflight = self.checker.check(point.joint_positions, constraints)
        if not preflight.feasible:
            state = collect_scene_state(self.env)
            event = MonitorEvent(
                index,
                preflight.failure_code or FailureCode.PATH_BLOCKED,
                f"preflight rejected trajectory sample: {preflight.message}",
                {"contacts": len(preflight.contacts)},
            )
            return event, dict(state.metrics), state, False
        action = self.adapter.build(point.joint_positions)
        observation, _, terminated, truncated, info = self.env.step(action, fast=False)
        if self.render:
            self.env.render()
        if self.frame_callback is not None:
            self.frame_callback(index, observation)
        state = collect_scene_state(self.env, info)
        stop_reached = bool(stop_condition and stop_condition(state))
        event = self.monitor.evaluate(
            step=index,
            target_joints=point.joint_positions,
            state=state,
            constraints=(terminal_constraints if stop_reached and terminal_constraints else constraints),
            protected_objects=protected_objects,
            require_holds=require_holds,
        )
        if event is None and truncated:
            event = MonitorEvent(index, FailureCode.TIMEOUT, "environment truncated the episode")
        if event is None and terminated and benchmark_success(state) < 1.0:
            event = MonitorEvent(index, FailureCode.TERMINATED, "environment terminated before success")
        return event, info, state, stop_reached

    def hold_plan(
        self,
        name: str,
        *,
        steps: int,
        constraints: ConstraintSet | None = None,
    ) -> MotionPlan:
        joints = tuple(
            self.adapter.hold_targets(collect_scene_state(self.env).robot.joint_positions)
        )
        return MotionPlan(
            name=name,
            points=tuple(
                TrajectoryPoint(joints, index / float(self.env.control_frequency))
                for index in range(max(1, int(steps)))
            ),
            constraints=constraints or ConstraintSet(),
        )

    def command_gripper(
        self,
        side: str,
        command: float,
        *,
        steps: int = 12,
        constraints: ConstraintSet | None = None,
        protected_objects: Mapping[str, Pose] | None = None,
    ) -> ExecutionReport:
        return self.command_grippers(
            {side: command},
            steps=steps,
            constraints=constraints,
            protected_objects=protected_objects,
        )

    def command_grippers(
        self,
        commands: Mapping[str, float],
        *,
        steps: int = 12,
        constraints: ConstraintSet | None = None,
        protected_objects: Mapping[str, Pose] | None = None,
    ) -> ExecutionReport:
        if not commands:
            raise ValueError("at least one gripper command is required")
        verb = "close" if all(value == CLOSE_COMMAND for value in commands.values()) else "open"
        target = "both" if set(commands) == {"left", "right"} else next(iter(commands))
        plan = self.hold_plan(
            f"{verb}_{target}_gripper",
            steps=steps,
            constraints=constraints,
        )
        return self.execute(
            plan,
            gripper_commands=commands,
            protected_objects=protected_objects,
            require_holds=False,
        )
