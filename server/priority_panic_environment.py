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

    def _get_tasks(self, level: str):
        """Return tasks based on difficulty level."""

        if level == "easy":
            return [
                {"id": "T1", "name": "Submit assignment", "deadline": "today", "priority": "high", "energy": 2, "depends_on": ""},
                {"id": "T2", "name": "Reply to friend", "deadline": "this_week", "priority": "low", "energy": 1, "depends_on": ""},
                {"id": "T3", "name": "Eat lunch", "deadline": "today", "priority": "high", "energy": 1, "depends_on": ""},
                {"id": "T4", "name": "Watch YouTube", "deadline": "this_week", "priority": "low", "energy": 1, "depends_on": ""},
                {"id": "T5", "name": "Read extra chapter", "deadline": "tomorrow", "priority": "medium", "energy": 2, "depends_on": ""},
            ]

        elif level == "medium":
            return [
                {"id": "T1", "name": "Fix critical bug in project", "deadline": "today", "priority": "high", "energy": 3, "depends_on": ""},
                {"id": "T2", "name": "Attend team meeting", "deadline": "today", "priority": "high", "energy": 1, "depends_on": ""},
                {"id": "T3", "name": "Write project report", "deadline": "tomorrow", "priority": "medium", "energy": 2, "depends_on": "T1"},
                {"id": "T4", "name": "Reply to recruiter email", "deadline": "tomorrow", "priority": "medium", "energy": 1, "depends_on": ""},
                {"id": "T5", "name": "Clean up desktop files", "deadline": "this_week", "priority": "low", "energy": 1, "depends_on": ""},
                {"id": "T6", "name": "Watch optional lecture", "deadline": "this_week", "priority": "low", "energy": 2, "depends_on": ""},
                {"id": "T7", "name": "Prepare presentation slides", "deadline": "tomorrow", "priority": "high", "energy": 3, "depends_on": ""},
            ]

        elif level == "hard":
            return [
                {"id": "T1", "name": "Submit hackathon project", "deadline": "today", "priority": "high", "energy": 3, "depends_on": ""},
                {"id": "T2", "name": "Debug deployment error", "deadline": "today", "priority": "high", "energy": 2, "depends_on": "T1"},
                {"id": "T3", "name": "Reply to mentor who is waiting", "deadline": "today", "priority": "high", "energy": 1, "depends_on": ""},
                {"id": "T4", "name": "Study for exam tomorrow", "deadline": "today", "priority": "high", "energy": 3, "depends_on": ""},
                {"id": "T5", "name": "Help teammate with their task", "deadline": "today", "priority": "medium", "energy": 2, "depends_on": "T2"},
                {"id": "T6", "name": "Eat and sleep properly", "deadline": "today", "priority": "high", "energy": 1, "depends_on": ""},
                {"id": "T7", "name": "Reply to 10 pending messages", "deadline": "this_week", "priority": "low", "energy": 1, "depends_on": ""},
                {"id": "T8", "name": "Update LinkedIn profile", "deadline": "this_week", "priority": "low", "energy": 1, "depends_on": ""},
            ]

    def reset(self, level: str = "easy") -> PriorityPanicObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._level = level
        self._available_energy = 5 if level == "easy" else 6 if level == "medium" else 7
        self._current_tasks = self._get_tasks(level)

        waiting = ""
        if level == "hard":
            waiting = "Your mentor Priya is waiting for a project update since yesterday."

        return PriorityPanicObservation(
            tasks=self._current_tasks,
            available_energy=self._available_energy,
            waiting_person=waiting,
            level=self._level,
            done=False,
            reward=0.0,
        )

    def step(self, action: PriorityPanicAction) -> PriorityPanicObservation:
        self._state.step_count += 1

        if self._level == "easy":
            reward = self._grade_easy(action)
        elif self._level == "medium":
            reward = self._grade_medium(action)
        else:
            reward = self._grade_hard(action)

        return PriorityPanicObservation(
            tasks=self._current_tasks,
            available_energy=self._available_energy,
            waiting_person="",
            level=self._level,
            done=True,
            reward=reward,
        )

    def _grade_easy(self, action: PriorityPanicAction) -> float:
        """Grade easy level — correct ordering of 5 tasks."""
        score = 0.0
        ordered = action.ordered_task_ids

        if not ordered:
            return 0.0

        # T1 and T3 are today+high — must be first two
        urgent = {"T1", "T3"}
        first_two = set(ordered[:2])
        urgent_score = len(urgent & first_two) / len(urgent)
        score += urgent_score * 0.5

        # T2 and T4 are low priority — must be last
        low = {"T2", "T4"}
        last_two = set(ordered[-2:]) if len(ordered) >= 2 else set()
        low_score = len(low & last_two) / len(low)
        score += low_score * 0.3

        # T5 medium — should be in middle
        if len(ordered) >= 3 and "T5" in ordered[1:-1]:
            score += 0.2

        return round(score, 2)

    def _grade_medium(self, action: PriorityPanicAction) -> float:
        """Grade medium level — choose 4 from 7, respect dependency."""
        score = 0.0
        ordered = action.ordered_task_ids
        dropped = action.dropped_task_ids

        if not ordered:
            return 0.0

        # Must keep all today+high tasks: T1, T2, T7
        must_keep = {"T1", "T2", "T7"}
        kept = set(ordered)
        kept_score = len(must_keep & kept) / len(must_keep)
        score += kept_score * 0.4

        # T3 depends on T1 — T1 must come before T3
        if "T1" in ordered and "T3" in ordered:
            if ordered.index("T1") < ordered.index("T3"):
                score += 0.3

        # Low priority tasks should be dropped
        should_drop = {"T5", "T6"}
        dropped_set = set(dropped)
        drop_score = len(should_drop & dropped_set) / len(should_drop)
        score += drop_score * 0.3

        return round(score, 2)

    def _grade_hard(self, action: PriorityPanicAction) -> float:
        """Grade hard level — complex constraints, dependencies, communication."""
        score = 0.0
        ordered = action.ordered_task_ids
        dropped = action.dropped_task_ids
        message = action.message_to_waiting_person

        if not ordered:
            return 0.0

        # T1 must come before T2 (dependency)
        if "T1" in ordered and "T2" in ordered:
            if ordered.index("T1") < ordered.index("T2"):
                score += 0.25

        # T2 must come before T5 (dependency)
        if "T2" in ordered and "T5" in ordered:
            if ordered.index("T2") < ordered.index("T5"):
                score += 0.15

        # T6 (eat/sleep) must be kept — self care is non negotiable
        if "T6" in ordered:
            score += 0.2

        # Low priority T7 T8 should be dropped
        should_drop = {"T7", "T8"}
        dropped_set = set(dropped)
        drop_score = len(should_drop & dropped_set) / len(should_drop)
        score += drop_score * 0.2

        # Message to waiting person must exist and be meaningful
        if message and len(message) > 20:
            score += 0.2

        return round(score, 2)

    def state(self) -> State:
        """Return current episode state (episode_id, step_count).
        
        Required by OpenEnv spec.
        """
        return self._state