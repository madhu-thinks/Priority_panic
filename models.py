from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List


class PriorityPanicObservation(Observation):
    """What the agent sees — a pile of tasks and constraints."""
    tasks: List[dict] = Field(default=[], description="List of tasks to prioritize")
    available_energy: int = Field(default=5, description="Total energy agent has today")
    waiting_person: str = Field(default="", description="Someone waiting for agent response")
    level: str = Field(default="easy", description="easy, medium, or hard")


class PriorityPanicAction(Action):
    """What the agent decides to do."""
    ordered_task_ids: List[str] = Field(default=[], description="Task IDs in priority order")
    dropped_task_ids: List[str] = Field(default=[], description="Tasks agent chose to drop")
    message_to_waiting_person: str = Field(default="", description="Message to person waiting")
    reasoning: str = Field(default="", description="Why agent made these choices")