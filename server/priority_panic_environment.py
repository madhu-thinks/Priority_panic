import sys
import os
import json
import textwrap
import httpx
from typing import List, Dict, Any, Optional

# Add parent directory to path to find models.py if running from server/
# This ensures "from models import ..." always works
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PriorityPanicObservation, PriorityPanicAction
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

class PriorityPanicEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS = 15

    def __init__(self):
        import uuid # Move here as it was removed from top
        self._uuid_lib = uuid
        self._episode_id = str(uuid.uuid4())
        self._action_history = []
        self._state_history = []
        self._current_grader_feedback = "Initial step."
        self._reset_internal()

    def reset(self, level: str = "easy", **kwargs) -> PriorityPanicObservation:
        """Initialize the environment for a new episode."""
        self._episode_id = str(self._uuid_lib.uuid4())
        self._action_history = []
        self._state_history = []
        self._reset_internal(level=level)
        return self._get_observation()

    async def reset_async(self, **kwargs) -> PriorityPanicObservation:
        """Required for OpenEnv WebSocket server compatibility."""
        return self.reset(**kwargs)

    def state(self) -> State:
        """Return the current episode state."""
        return State(
            episode_id=self._episode_id,
            step_count=self.step_count,
        )

    def _reset_internal(self, level="easy"):
        self.step_count = 0
        self._level = level
        self._available_energy = 5 if level == "easy" else 7
        self._current_tasks = self._get_base_tasks(level)
        self._streak = 0
        self._prev_step_reward = 0.0
        self._social_debt = 0.0  # Tracks annoyance of stakeholders

    def _get_base_tasks(self, level: str):
        loadouts = {
            "easy": [
                {"id": "T1", "priority": "high", "energy": 2, "age": 0},
                {"id": "T3", "priority": "high", "energy": 1, "age": 0},
                {"id": "T5", "priority": "medium", "energy": 2, "age": 0},
            ],
            "medium": [
                {"id": "T1", "priority": "high", "energy": 3, "age": 0},
                {"id": "T2", "priority": "high", "energy": 2, "age": 0},
                {"id": "T8", "priority": "low", "energy": 2, "age": 0},
            ],
            "hard": [
                {"id": "T1", "priority": "high", "energy": 4, "age": 0},
                {"id": "T6", "priority": "high", "energy": 3, "age": 0},
                {"id": "S0", "priority": "high", "energy": 2, "age": 0},
            ]
        }
        return loadouts.get(level, loadouts["easy"])

    def step(self, action: PriorityPanicAction, **kwargs) -> PriorityPanicObservation:
        """Execute one step in the environment."""
        self.step_count += 1
        return self._step_internal(action)

    async def step_async(self, action: PriorityPanicAction, **kwargs) -> PriorityPanicObservation:
        """Required for OpenEnv WebSocket server compatibility."""
        # 1. Execute the step logic (sync)
        obs = self.step(action, **kwargs)
        
        # 2. Record history for the grader
        self._action_history.append(action.dict())
        self._state_history.append(obs.dict())
        
        # 3. Periodic Grader (Modulo 3)
        if self.step_count > 0 and self.step_count % 3 == 0:
            await self._run_llm_grader()
            # The next observation will carry the new reward and feedback
            obs.reward = self._prev_step_reward
            obs.metadata["grader_feedback"] = self._current_grader_feedback
            
        return obs

    async def _run_llm_grader(self):
        """Calls Qwen2.5 to evaluate the recent 3 steps against the rubric."""
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            self._current_grader_feedback = "Grader offline: No HF_TOKEN."
            return

        # Prepare context (last 3 steps)
        history_window = {
            "states": self._state_history[-3:],
            "actions": self._action_history[-3:]
        }
        
        prompt = self._get_grader_prompt(history_window)
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://router.huggingface.co/v1/chat/completions",
                    headers={"Authorization": f"Bearer {hf_token}"},
                    json={
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "messages": [
                            {"role": "system", "content": "You are an Elite AI Grader for the Priority Panic RL environment."},
                            {"role": "user", "content": prompt}
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.1
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    result = json.loads(data["choices"][0]["message"]["content"])
                    
                    # Compute rubric-based reward: (Decision + Efficiency + Impact) / 3
                    d = float(result.get("decision", 0.5))
                    e = float(result.get("efficiency", 0.5))
                    i = float(result.get("impact", 0.5))
                    
                    self._prev_step_reward = max(0.01, min(1.0, (d + e + i) / 3.0))
                    self._current_grader_feedback = result.get("reasoning", "Evaluated by Qwen.")
                else:
                    self._current_grader_feedback = f"Grader error: API returned {response.status_code}"
        except Exception as e:
            self._current_grader_feedback = f"Grader Exception: {str(e)}"

    def _get_grader_prompt(self, window: dict) -> str:
        return textwrap.dedent(f"""
            Analyze the following 3 steps of a Priority Panic agent's trajectory.
            
            HISTORY DATA:
            {json.dumps(window, indent=2)}
            
            RUBRIC (Score each 0.0 to 1.0):
            1. Decision Quality: Did the agent prioritize 'high' and 'panic' tasks correctly?
            2. Efficiency: Did the agent maximize energy usage per step?
            3. Impact: Did the agent reduce system stress (panic tasks) and maintain social debt?
            
            Return ONLY a JSON object:
            {{
                "decision": float,
                "efficiency": float,
                "impact": float,
                "reasoning": "string (brief justification)"
            }}
        """).strip()

    def _step_internal(self, action: PriorityPanicAction) -> PriorityPanicObservation:
        ids_to_process = action.ordered_task_ids or []
        
        energy_used = 0
        completed_ids = []
        
        # 1. Energy Validation Logic
        for tid in ids_to_process:
            task = next((t for t in self._current_tasks if t["id"] == tid), None)
            if task and (energy_used + task["energy"] <= self._available_energy):
                energy_used += task["energy"]
                completed_ids.append(tid)

        # 2. Update State
        if len(completed_ids) > 0:
            self._streak += 1
        else:
            self._streak = 0

        # Remove completed and age the rest
        self._current_tasks = [t for t in self._current_tasks if t["id"] not in completed_ids]
        for t in self._current_tasks:
            t["age"] += 1

        # 3. Social Logic (Stakeholder Interaction)
        message = getattr(action, "message_to_waiting_person", "")
        if not message or len(str(message)) < 10:
            self._social_debt += 0.15  # Stakeholder gets annoyed by silence
        else:
            self._social_debt = max(0, self._social_debt - 0.1) # Stakeholder satisfied

        # 4. Dynamic Task Injection (High Pressure)
        # In Hard mode, inject tasks even faster
        injection_steps = [2, 4, 6] if self._level == "hard" else [2, 5]
        if self.step_count in injection_steps:
            self._current_tasks.append({
                "id": f"S{self.step_count}", 
                "priority": "high" if self._level != "easy" else "medium",
                "energy": 3, "age": 0
            })

        # 5. REWARD CALCULATION (The "Push" Logic)
        step_reward = self._calculate_reward(completed_ids, energy_used)
        
        #  if current reward > previous reward
        if step_reward > self._prev_step_reward:
            step_reward += 0.05 
        elif step_reward < self._prev_step_reward and step_reward > 0:
            step_reward -= 0.02 # "Momentum Loss" penalty
            
        self._prev_step_reward = step_reward
        
        done = self.step_count >= self.MAX_STEPS or (not self._current_tasks and self.step_count > 5)
        
        return self._get_observation(reward=step_reward, done=done)

    def _get_observation(self, reward: float = 0.0, done: bool = False) -> PriorityPanicObservation:
        """Constructs an observation object based on internal state."""
        return PriorityPanicObservation(
            tasks=self._current_tasks,
            available_energy=self._available_energy,
            waiting_person="Mentor" if self.step_count > 4 else "Colleague",
            level=self._level,
            done=done,
            reward=reward,
            metadata={"streak": self._streak}
        )

    def _calculate_reward(self, completed_ids, energy_used: int) -> float:
        """
        SHARP REWARD SHAPING (Hackathon Optimized):
        - Partial Progress: Gain based on energy units spent (0.05 / unit).
        - Completion Gain: 0.3 per task.
        - Social Penalty: -0.1 per unit of social debt.
        - Panic Penalty: Exponential aging penalty.
        """
        if not completed_ids and not self._current_tasks:
            return 0.5 # Maintenance reward

        # 1. Partial Progress & Completion
        gain = (len(completed_ids) * 0.3) + (energy_used * 0.05)
        
        # 2. Streak bonus
        streak_bonus = (self._streak ** 2) * 0.02 
        
        # 3. Panic Penalty (Aging)
        panic_penalty = 0.0
        for t in self._current_tasks:
            p_multiplier = 2.0 if t["priority"] == "high" else 1.0
            panic_penalty += (0.1 * (t["age"] ** 1.5)) * p_multiplier

        # 4. Social Penalty
        social_penalty = self._social_debt * 0.1

        # Final score calculation
        raw_score = gain + streak_bonus - panic_penalty - social_penalty
        
        # Clamp between 0 and 1.0
        return max(0.0, min(1.0, raw_score))