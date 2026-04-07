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

## 🎯 Reward Function: Granular Progress Signal

The grader provides a `0.0–1.0` reward per step to incentivize long-term planning:
1. **Completion Gain**: `+0.30` per completed task.
2. **Partial Progress**: `+0.05` per unit of energy efficiently utilized.
3. **Panic Penalty**: `-(0.1 * age^1.5)` per task left aging.
4. **Social Penalty**: `-(0.1 * social_debt)` if the agent fails to communicate with a waiting stakeholder.

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