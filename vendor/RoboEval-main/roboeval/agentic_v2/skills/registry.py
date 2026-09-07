"""Single dispatch point from validated semantic requests to verified skills."""

from __future__ import annotations

from roboeval.agentic_v2.skills.base import SkillContext
from roboeval.agentic_v2.skills.bimanual_grasp import BimanualGraspSkill
from roboeval.agentic_v2.skills.close_flap import CloseFlapSkill
from roboeval.agentic_v2.skills.grasp import GraspSkill
from roboeval.agentic_v2.skills.handover import HandoverSkill
from roboeval.agentic_v2.skills.lift import LiftSkill
from roboeval.agentic_v2.skills.place import PlaceSkill
from roboeval.agentic_v2.skills.rotate import RotateSkill
from roboeval.agentic_v2.skills.transport import TransportSkill
from roboeval.agentic_v2.types import SkillName, SkillRequest, SkillResult


class SkillRegistry:
    """Execute only the closed set of Agentic v2 semantic skills."""

    def __init__(self, context: SkillContext) -> None:
        self.skills = {
            SkillName.GRASP: GraspSkill(context),
            SkillName.BIMANUAL_GRASP: BimanualGraspSkill(context),
            SkillName.LIFT: LiftSkill(context),
            SkillName.TRANSPORT: TransportSkill(context),
            SkillName.HANDOVER: HandoverSkill(context),
            SkillName.PLACE: PlaceSkill(context),
            SkillName.CLOSE_FLAP: CloseFlapSkill(context),
            SkillName.ROTATE: RotateSkill(context),
        }

    def execute(self, request: SkillRequest) -> SkillResult:
        if request.skill is SkillName.FINISH:
            raise ValueError("finish is handled by the task runner")
        try:
            skill = self.skills[request.skill]
        except KeyError as error:
            raise ValueError(f"unsupported semantic skill {request.skill.value!r}") from error
        return skill.execute(request)
