---
title: Priority Panic
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
license: other
short_description: 'OpenEnv: Task prioritization under extreme pressure.'
tags:
  - openenv
  - reinforcement-learning
  - pytorch
---

# 🚨 Priority Panic: OpenEnv Decision-Making Environment

AI systems today are highly capable of generating responses but often struggle in **real-world, dynamic decision-making scenarios**. **Priority Panic** is a Reinforcement Learning (RL) environment designed to bridge this gap, forcing agents to operate within evolving states and mounting consequences.

---

## 🧠 Motivation: Real-World Utility (30% Weight)

Most RL benchmarks are static. **Priority Panic** simulates a universal human challenge: **The Deadline Crush**.  
Inspired by actual hackathon conditions, it models:
- **Exponential Aging**: Every step a high-priority task is ignored, its negative impact grows non-linearly.
- **Social Debt (Stakeholder Sentiment)**: AI agents must communicate. Ignoring stakeholders (waiting persons) leads to trust depletion and severe score penalties.
- **Panic Spawns**: Real-life isn't predictable. Interruptions happen at the worst times, forcing the agent to re-evaluate its entire plan.

---

## 📊 Action & Observation Spaces

### Observation Space
Captured via `PriorityPanicObservation`.

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `tasks` | `List[Dict]` | Active tasks with `id`, `priority`, `energy`, and `age`. | `[{"id": "T1", "energy": 3, ...}]` |
| `available_energy` | `int` | Total units you can spend this step (max 5 or 7). | `5` |
| `waiting_person` | `str` | Name of the stakeholder expecting an update. | `"Mentor"` |
| `level` | `str` | Difficulty: `easy`, `medium`, or `hard`. | `"hard"` |

### Action Space
Captured via `PriorityPanicAction`.

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `ordered_task_ids` | `List[str]` | Order of operations to execute within energy limits. | `["T1", "S2"]` |
| `message_to_waiting_person` | `str` | Status update for the stakeholder to reduce social debt. | `"Working on T1 now."` |
| `reasoning` | `str` | Clinical logic justification for the agent's decision. | `"T1 aging fast; VPE focus."` |

---

## 🧠 Intelligent Grader (LLM-as-a-Judge)
Priority Panic V2 introduces a sophisticated **Reinforcement Learning-based reward mechanism** combined with an **LLM-as-a-Judge** approach.

- **Periodic Grading**: Every 3 steps, the environment calls **Qwen2.5-7B-Instruct** to evaluate the agent's recent trajectory.
- **Multidimensional Rubric**: The grader evaluates:
    - **Decision Quality**: Was prioritization correct? (Score 0-1)
    - **Efficiency**: Was energy usage maximized per step? (Score 0-1)
    - **Impact**: Was system stress (panic) and social debt reduced? (Score 0-1)
- **Continuous Reward Signal**: Between grading steps, a rule-based system provides partial rewards (+0.05 per energy unit) to maintain a dense learning signal.
- **Long-Horizon Trajectories (15 Steps)**: Increased episode length ensures agents must plan for future task aging and dynamic spawns.

---

## 🚀 Quick Start & Baseline

### Local Development
```bash
# Install dependencies
pip install -e .

# Start the server
python -m priority_panic.server.app

# Run the mandatory benchmark
python inference.py
```

### Baseline Reproduced (Qwen2.5-72B)
| Task | Cumulative Score | Reliability |
| :--- | :--- | :--- |
| 🟢 Easy | 0.85 | ✅ High |
| 🟡 Medium | 0.72 | ✅ Stable |
| 🔴 Hard | 0.54 | ✅ Challenging |

---

## 🛠️ OpenEnv Spec Compliance
- ✅ **Standard API**: Implements `reset()`, `step()`, and `state()`.
- ✅ **Typed Models**: Full Pydantic validation for Action/Observation.
- ✅ **Mandatory Logging**: `inference.py` adheres to `[START]`, `[STEP]`, `[END]` formats.
- ✅ **Dockerized**: Multi-stage `Dockerfile` provided for zero-friction deployment.