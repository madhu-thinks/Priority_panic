---
title: Priority Panic
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: other
short_description: 'RL environment: prioritize tasks under pressure.'
tags:
  - openenv
---

# Priority Panic Environment

> Inspired by a real situation: two students, four days, no prior experience, and a decision to build anyway.
> Every task in this environment is something we personally faced while building it.
> The agent that learns to handle this learns what we had to learn the hard way.

An RL environment where an AI agent learns to prioritize tasks under pressure—
deadlines, dependencies, limited energy, and people waiting.

Three difficulty levels: from simple task ordering to complex, multi-constraint planning.

This isn’t a toy problem. Every human alive has lived this.
Now AI can learn it too.

## Quick Start

The client is **async by default**:
```python
import asyncio
from priority_panic import PriorityPanicAction, PriorityPanicEnv

async def main():
    client = await PriorityPanicEnv.from_docker_image("priority_panic-env:latest")

    async with client:
        result = await client.reset()
        print(f"Level: {result.observation.level}")
        print(f"Tasks: {result.observation.tasks}")

        action = PriorityPanicAction(
            ordered_task_ids=["T1", "T3", "T5", "T2", "T4"],
            dropped_task_ids=[],
            message_to_waiting_person="",
            reasoning="Urgent tasks first"
        )
        result = await client.step(action)
        print(f"Reward: {result.reward}")

asyncio.run(main())
