# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Priority Panic Environment Models — India AI Hackathon '26 Edition."""

from typing import List, Dict, Any, Optional
from pydantic import Field, ConfigDict
from openenv.core.env_server.types import Action, Observation

class PriorityPanicObservation(Observation):
    """
    The state of the world as seen by the Agent.
    Contains tasks, resource constraints, and environmental context.
    """
    
    tasks: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Active tasks: Includes 'id', 'priority', 'energy', and 'age'."
    )
    available_energy: int = Field(
        default=5, 
        description="Total energy units available to spend this step."
    )
    waiting_person: str = Field(
        default="", 
        description="Contextual stakeholder waiting for a status update."
    )
    level: str = Field(
        default="easy", 
        description="Current difficulty setting (Easy, Medium, or Hard)."
    )
    
    # Required for OpenEnv 
    done: bool = Field(
        default=False, 
        description="Flag indicating if the 9-step episode has concluded."
    )
    reward: float = Field(
        default=0.0, 
        description="The immediate scalar reward from the previous step (0.0 to 1.0)."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional debugging or environment-specific data."
    )

    # CRITICAL: Prevents Pydantic from crashing if the server sends unexpected fields.
    model_config = ConfigDict(extra='ignore')


class PriorityPanicAction(Action):
    """
    The decision made by the Agent.
    Dictates task processing and communication strategy.
    """
    
    ordered_task_ids: List[str] = Field(
        default_factory=list, 
        description="IDs of tasks to execute this step, in order of execution."
    )
    dropped_task_ids: List[str] = Field(
        default_factory=list, 
        description="IDs of tasks the agent is explicitly ignoring to save energy."
    )
    message_to_waiting_person: str = Field(
        default="", 
        description="A natural language status update for the stakeholder."
    )
    reasoning: str = Field(
        default="", 
        description="The clinical logic/VPE math justifying this specific action."
    )

    # CRITICAL: Ensures compatibility with varying server payload structures.
    model_config = ConfigDict(extra='ignore')