from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List


class PriorityPanicObservation(Observation):
    """What the agent sees — a pile of tasks and constraints."""
    tasks: List[dict] = Field(default=[], description="List of tasks to prioritize (includes age)")
    available_energy: int = Field(default=5, description="Total energy agent has for this step")
    waiting_person: str = Field(default="", description="Someone waiting for agent response")
    level: str = Field(default="easy", description="easy, medium, or hard")
    current_step: int = Field(default=1, description="Current time step of the episode")
    max_steps: int = Field(default=15, description="Maximum number of steps in the episode")


class PriorityPanicAction(Action):
    """What the agent decides to do."""
    ordered_task_ids: List[str] = Field(default=[], description="Tasks to execute this step, up to energy limit")
    dropped_task_ids: List[str] = Field(default=[], description="Tasks agent chose to drop permanently")
    message_to_waiting_person: str = Field(default="", description="Message to person waiting")
    reasoning: str = Field(default="", description="Why agent made these choices for this step")