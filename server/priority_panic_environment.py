import math
from uuid import uuid4
# Assuming these interfaces exist in your openenv setup
# from openenv.core.env_server.interfaces import Environment
# from openenv.core.env_server.types import State

class PriorityPanicEnvironment: # Inherit from Environment as per your setup
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS = 9  # Reduced as per our efficiency optimization

    def __init__(self):
        self._reset_internal()

    def _reset_internal(self, level="easy"):
        self.step_count = 0
        self._level = level
        self._available_energy = 5 if level == "easy" else 7
        self._current_tasks = self._get_base_tasks(level)
        self._streak = 0
        self._prev_step_reward = 0.0  # Track last reward to enforce improvement

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

    def step(self, action):
        self.step_count += 1
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

        # 3. Dynamic Task Injection (High Pressure)
        if self.step_count in [2, 5]: # Inject sooner for a 9-step limit
            self._current_tasks.append({
                "id": f"S{self.step_count}", 
                "priority": "high",
                "energy": 3, "age": 0
            })

        # 4. REWARD CALCULATION (The "Push" Logic)
        step_reward = self._calculate_reward(completed_ids)
        
        # ENFORCED IMPROVEMENT: Small bonus if current reward > previous reward
        if step_reward > self._prev_step_reward:
            step_reward += 0.05 
        elif step_reward < self._prev_step_reward and step_reward > 0:
            step_reward -= 0.02 # Slight "Momentum Loss" penalty
            
        self._prev_step_reward = step_reward
        
        done = self.step_count >= self.MAX_STEPS or (not self._current_tasks and self.step_count > 5)
        
        return self._get_observation(reward=step_reward, done=done)

    def _calculate_reward(self, completed_ids) -> float:
        """
        SHARP REWARD SHAPING:
        - High Penalty for High Priority aging.
        - Higher weight for finishing tasks.
        - Removed the +2.0 offset to make 0.0 actually feel like a failure.
        """
        if not completed_ids and not self._current_tasks:
            return 0.5 # Maintenance reward

        # Base Gain: 0.4 per task (Stronger signal)
        gain = len(completed_ids) * 0.4
        
        # Streak bonus (Exponential to push for long streaks)
        streak_bonus = (self._streak ** 2) * 0.02 
        
        # Panic Penalty (Exponential aging penalty)
        penalty = 0.0
        for t in self._current_tasks:
            # Exponentially increase penalty as high-priority tasks age
            p_multiplier = 2.0 if t["priority"] == "high" else 1.0
            penalty += (0.1 * (t["age"] ** 1.5)) * p_multiplier

        # Final score calculation without the "softening" offset
        raw_score = gain + streak_bonus - penalty
        
        # Clamp between 0 and 1.0
        return max(0.0, min(1.0, raw_score))