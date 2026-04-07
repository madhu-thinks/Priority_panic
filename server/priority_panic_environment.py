import math
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PriorityPanicAction, PriorityPanicObservation
except ImportError:
    from models import PriorityPanicAction, PriorityPanicObservation


class PriorityPanicEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_tasks = []
        self._level = "easy"
        self._available_energy = 5
        self._max_steps = 15
        self._cumulative_reward = 0.0

    def _get_tasks(self, level: str):
        """Return tasks based on difficulty level."""
        tasks = []
        if level == "easy":
            tasks = [
                {"id": "T1", "name": "Submit assignment", "deadline": "today", "priority": "high", "energy": 2, "depends_on": ""},
                {"id": "T2", "name": "Reply to friend", "deadline": "this_week", "priority": "low", "energy": 1, "depends_on": ""},
                {"id": "T3", "name": "Eat lunch", "deadline": "today", "priority": "high", "energy": 1, "depends_on": ""},
                {"id": "T4", "name": "Watch YouTube", "deadline": "this_week", "priority": "low", "energy": 1, "depends_on": ""},
                {"id": "T5", "name": "Read extra chapter", "deadline": "tomorrow", "priority": "medium", "energy": 2, "depends_on": ""},
            ]
        elif level == "medium":
            tasks = [
                {"id": "T1", "name": "Fix critical bug in project", "deadline": "today", "priority": "high", "energy": 3, "depends_on": ""},
                {"id": "T2", "name": "Attend team meeting", "deadline": "today", "priority": "high", "energy": 1, "depends_on": ""},
                {"id": "T3", "name": "Write project report", "deadline": "tomorrow", "priority": "medium", "energy": 2, "depends_on": "T1"},
                {"id": "T4", "name": "Reply to recruiter email", "deadline": "tomorrow", "priority": "medium", "energy": 1, "depends_on": ""},
                {"id": "T5", "name": "Clean up desktop files", "deadline": "this_week", "priority": "low", "energy": 1, "depends_on": ""},
                {"id": "T6", "name": "Watch optional lecture", "deadline": "this_week", "priority": "low", "energy": 2, "depends_on": ""},
                {"id": "T7", "name": "Prepare presentation slides", "deadline": "tomorrow", "priority": "high", "energy": 3, "depends_on": ""},
            ]
        elif level == "hard":
            tasks = [
                {"id": "T1", "name": "Submit hackathon project", "deadline": "today", "priority": "high", "energy": 3, "depends_on": ""},
                {"id": "T2", "name": "Debug deployment error", "deadline": "today", "priority": "high", "energy": 2, "depends_on": "T1"},
                {"id": "T3", "name": "Reply to mentor who is waiting", "deadline": "today", "priority": "high", "energy": 1, "depends_on": ""},
                {"id": "T4", "name": "Study for exam tomorrow", "deadline": "today", "priority": "high", "energy": 3, "depends_on": ""},
                {"id": "T5", "name": "Help teammate with their task", "deadline": "today", "priority": "medium", "energy": 2, "depends_on": "T2"},
                {"id": "T6", "name": "Eat and sleep properly", "deadline": "today", "priority": "high", "energy": 1, "depends_on": ""},
                {"id": "T7", "name": "Reply to 10 pending messages", "deadline": "this_week", "priority": "low", "energy": 1, "depends_on": ""},
                {"id": "T8", "name": "Update LinkedIn profile", "deadline": "this_week", "priority": "low", "energy": 1, "depends_on": ""},
            ]
        
        # Initialize age for all tasks
        for t in tasks:
            t["age"] = 0
            
        return tasks

    def reset(self, level: str = "easy") -> PriorityPanicObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._level = level
        self._available_energy = 5 if level == "easy" else 6 if level == "medium" else 7
        self._current_tasks = self._get_tasks(level)
        self._cumulative_reward = 0.0

        waiting = ""
        if level == "hard":
            waiting = "Your mentor Priya is waiting for a project update since yesterday."

        return PriorityPanicObservation(
            tasks=self._current_tasks,
            available_energy=self._available_energy,
            waiting_person=waiting,
            level=self._level,
            current_step=1,
            max_steps=self._max_steps,
            done=False,
            reward=0.0,
        )

    def step(self, action: PriorityPanicAction) -> PriorityPanicObservation:
        self._state.step_count += 1
        step_reward = 0.0
        
        # 1. Base Multipliers based on difficulty
        if self._level == "easy":
            baseline_completion = 0.5
            panic_base = 0.05
        elif self._level == "medium":
            baseline_completion = 1.0
            panic_base = 0.1
        else: # hard
            baseline_completion = 2.0
            panic_base = 0.2

        # 2. Binary Completion Logic
        # Agent executes tasks in order until energy is exhausted
        remaining_energy = self._available_energy
        executed_tasks_this_step = []
        
        task_dict = {t["id"]: t for t in self._current_tasks}
        
        for tid in action.ordered_task_ids:
            if tid in task_dict:
                task = task_dict[tid]
                # Check Dependencies
                if task.get("depends_on") and task["depends_on"] in task_dict:
                    # Dependency not completed yet (still in current_tasks)
                    step_reward -= 0.5 # Penalty for violating dependency
                    continue

                if remaining_energy >= task["energy"]:
                    # Execute Task!
                    remaining_energy -= task["energy"]
                    executed_tasks_this_step.append(tid)
                    
                    # Reward for completion (scales with priority)
                    multiplier = 1.5 if task["priority"] == "high" else (1.0 if task["priority"] == "medium" else 0.5)
                    step_reward += baseline_completion * multiplier
                    
                    del task_dict[tid]
                else:
                    # Not enough energy -> Cannot partially complete (Binary completion rule)
                    # We just skip it, it stays in the backlog
                    pass
                    
        # 3. Handle Dropped Tasks Permanently
        for tid in action.dropped_task_ids:
            if tid in task_dict:
                task = task_dict[tid]
                # Slight penalty for dropping high priority, reward for dropping low priority
                if task["priority"] == "high":
                    step_reward -= baseline_completion * 1.5
                elif task["priority"] == "low":
                    step_reward += baseline_completion * 0.2
                del task_dict[tid]

        # Reconstruct current tasks list
        self._current_tasks = list(task_dict.values())
        
        # 4. Spawning Dynamic Tasks (Specific Step Intervals)
        if self._state.step_count == 3:
            new_t = {"id": "T_S3", "name": "Urgent Email from Boss", "deadline": "today", "priority": "high", "energy": 1, "depends_on": "", "age": 0}
            self._current_tasks.append(new_t)
        if self._state.step_count == 7 and self._level in ["medium", "hard"]:
            new_t = {"id": "T_S7", "name": "Unexpected Server Outage", "deadline": "today", "priority": "high", "energy": 2, "depends_on": "", "age": 0}
            self._current_tasks.append(new_t)
        if self._state.step_count == 10 and self._level == "hard":
            new_t = {"id": "T_S10", "name": "URGENT HR Request", "deadline": "today", "priority": "medium", "energy": 1, "depends_on": "", "age": 0}
            self._current_tasks.append(new_t)

        # 5. Exponential Panic Logic
        panic_penalty_total = 0.0
        for t in self._current_tasks:
            t["age"] += 1
            # Age penalty only matters if they were high/medium priority. Delaying low priority is fine.
            if t["priority"] in ["high", "medium"]:
                mult = 1.0 if t["priority"] == "high" else 0.5
                panic_penalty_total += (panic_base * mult) * math.exp(t["age"] * 0.2)
                
        step_reward -= panic_penalty_total
        
        # Message checks
        if self._level == "hard" and "mentor" in action.message_to_waiting_person.lower():
            if len(action.message_to_waiting_person.split()) >= 10:
                step_reward += 0.5

        # Accumulate
        self._cumulative_reward += step_reward

        # Done state - Episode ends ONLY when max_steps is reached
        is_done = self._state.step_count >= self._max_steps

        # Note: waiting_person resets if handled, but we'll leave it static for the observation if requested.
        waiting = "Your mentor Priya is waiting for a project update since yesterday." if (self._level == "hard" and not action.message_to_waiting_person) else ""

        return PriorityPanicObservation(
            tasks=self._current_tasks,
            available_energy=self._available_energy,
            waiting_person=waiting,
            level=self._level,
            current_step=self._state.step_count + 1,
            max_steps=self._max_steps,
            done=is_done,
            reward=round(step_reward, 3),
        )

    def state(self) -> State:
        """Return current episode state."""
        return self._state