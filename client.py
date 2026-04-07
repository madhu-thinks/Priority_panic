# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Priority Panic Environment Client — High-Efficiency Logic Sync."""

import os
from typing import Dict, Any
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

# Internal imports - ensures alignment with your local Pydantic schemas
try:
    from .models import PriorityPanicAction, PriorityPanicObservation
except ImportError:
    from models import PriorityPanicAction, PriorityPanicObservation

class PriorityPanicEnv(
    EnvClient[PriorityPanicAction, PriorityPanicObservation, State]
):
    """
    Client for the Priority Panic Environment.
    Facilitates communication between the AI Agent and the Environment Server.
    Optimized for 9-step efficiency runs.
    """

    def _step_payload(self, action: PriorityPanicAction) -> Dict[str, Any]:
        """
        Serializes the Action object into a JSON-compatible dictionary.
        Includes fields for reasoning and task management to push AI transparency.
        """
        return {
            "ordered_task_ids": action.ordered_task_ids,
            "dropped_task_ids": getattr(action, "dropped_task_ids", []),
            "message_to_waiting_person": getattr(action, "message_to_waiting_person", ""),
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PriorityPanicObservation]:
        """
        Parses the server's JSON response into a StepResult.
        Fixes:
        1. Gracefully handles missing 'observation' keys.
        2. Aligns reward/done flags between the payload and the observation object.
        3. Removes redundant fields that trigger Pydantic validation crashes.
        """
        # Deep extract observation data
        obs_data = payload.get("observation", {})
        
        # Unify reward and done status across the response
        reward = payload.get("reward", obs_data.get("reward", 0.0))
        done = payload.get("done", obs_data.get("done", False))

        # Map to Pydantic Observation Model
        observation = PriorityPanicObservation(
            tasks=obs_data.get("tasks", []),
            available_energy=obs_data.get("available_energy", 0),
            waiting_person=obs_data.get("waiting_person", ""),
            level=obs_data.get("level", "easy"),
            done=done,
            reward=reward,
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Extracts session metadata to maintain episode continuity.
        """
        return State(
            episode_id=payload.get("episode_id") or payload.get("session_id"),
            step_count=payload.get("step_count", 0),
        )

# --- Clean Implementation Notes --- #
# 1. 'getattr' is used for dropped_task_ids to prevent crashes if your 
#    models.py hasn't been updated with those specific fields yet.
# 2. The _parse_result logic is now "Top-Down", checking the root payload 
#    first for reward/done, then falling back to the observation dictionary.