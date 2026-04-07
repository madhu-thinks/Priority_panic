"""
Inference Script — Priority Panic Environment

This script benchmarks AI models on the Priority Panic environment.
It uses the OpenAI client to interface with LLM APIs.

Required environment variables:
    - API_BASE_URL: LLM API endpoint (default: HuggingFace Router)
    - MODEL_NAME: Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    - HF_TOKEN: HuggingFace API token (or API_KEY)
    - LOCAL_IMAGE_NAME: (optional) Docker image for local testing
    - HF_SPACE_URL: (optional) HuggingFace Space URL for deployed environment
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Dict, List, Optional

from openai import OpenAI

from priority_panic import PriorityPanicAction, PriorityPanicEnv

# Environment Configuration
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "priority_panic"
MAX_STEPS = 15
TEMPERATURE = 0.7
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 5.0  # arbitrary threshold for average rewards


def validate_config() -> bool:
    """Validate required environment variables before execution."""
    errors = []
    
    if not API_KEY:
        errors.append("ERROR: HF_TOKEN or API_KEY not set")
    if not IMAGE_NAME and not os.getenv("HF_SPACE_URL"):
        errors.append("ERROR: Neither LOCAL_IMAGE_NAME nor HF_SPACE_URL set")
    if not API_BASE_URL:
        errors.append("ERROR: API_BASE_URL not set")
    if not MODEL_NAME:
        errors.append("ERROR: MODEL_NAME not set")
    
    if errors:
        print("[START] task=validation status=failed", flush=True)
        for err in errors:
            print(f"[DEBUG] {err}", flush=True)
        print("[END] success=false steps=0 score=0.0 rewards=", flush=True)
        return False
    
    return True

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert task prioritization agent operating in a MULTI-STEP environment.
    Each step, you receive a list of current tasks, their ages, and a bandwidth (energy) budget.
    Your goal is to maximize your cumulative reward over the episode.

    STRICT RULES:
    1. ENERGY BUDGET (PER STEP): The sum of energy costs for ordered_task_ids must NOT exceed available_energy for THIS step.
       Tasks that you execute will be binary completed and removed.
    2. DEPENDENCIES: If task B depends_on task A, A must be completed BEFORE B.
    3. THE PANIC MECHANIC (AGE PENALTY): The "age" of a task increases every step it is not completed.
       Uncompleted High & Medium priority tasks incur an exponentially growing penalty based on their age every single step. Complete them FAST!
    4. PERMANENT DROPPING: Tasks placed in dropped_task_ids are PERMANENTLY deleted. Be very careful dropping High tasks.
    5. COMMUNICATION: If there is a waiting_person, you MUST write a meaningful message (at least 10 words) explaining the delay.
    6. PERSISTENCE: Tasks you don't complete or drop today WILL persist to the next step. Only use ordered_task_ids for what you can actually finish TODAY given your energy bandwidth.

    Respond ONLY in this exact JSON format, no extra text:
    {
        "ordered_task_ids": ["T1", "T3"],
        "dropped_task_ids": ["T7"],
        "message_to_waiting_person": "Hi mentor, working on the hackathon project.",
        "reasoning": "Executed high priority tasks to prevent age panic. Carried over remaining tasks to next step."
    }
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log step execution in hackathon-required format.
    
    Format: [STEP] step=<int> action=<str> reward=<float> done=<bool> error=<str|null>
    """
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode completion in hackathon-required format.
    
    Format: [END] success=<bool> steps=<int> score=<float> rewards=<csv_floats>
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def get_model_action(client: OpenAI, observation: Dict) -> Dict:
    """Request task prioritization decision from LLM model.
    
    Args:
        client: OpenAI client configured for the API endpoint
        observation: Environment observation containing tasks and constraints
        
    Returns:
        Parsed action dict with keys: ordered_task_ids, dropped_task_ids,
        message_to_waiting_person, reasoning
    """
    level = observation.get('level', 'easy')
    energy = observation.get('available_energy', 5)
    waiting = observation.get('waiting_person') or 'None'
    tasks = observation.get('tasks', [])

    # Build a clean task listing for the prompt
    task_lines = []
    total_energy_all = 0
    for t in tasks:
        tid = t.get('id', 'unknown')
        name = t.get('name', 'Unnamed task')
        deadline = t.get('deadline', 'none')
        priority = t.get('priority', 'medium')
        energy_cost = t.get('energy', 0)
        age = t.get('age', 0)
        dep_id = t.get('depends_on', '')
        
        dep = f" | depends_on: {dep_id}" if dep_id else ""
        task_lines.append(
            f"  {tid}: {name} | deadline: {deadline} | priority: {priority} | energy: {energy_cost} | age: {age}{dep}"
        )
        total_energy_all += energy_cost

    task_str = "\n".join(task_lines)
    energy_warning = "WARNING: Total energy of all tasks exceeds budget. You MUST drop some tasks." if total_energy_all > energy else "Energy is sufficient if you are selective."

    user_prompt = textwrap.dedent(f"""
        STEP: {observation.get('current_step', 1)} / {observation.get('max_steps', 15)}
        LEVEL: {level}
        ENERGY BUDGET: {energy} (do not exceed this with your ordered tasks)
        {energy_warning}

        TASKS:
{task_str}

        WAITING PERSON: {waiting}

        Your job:
        - ordered_task_ids: task IDs you will DO, in correct priority/dependency order
        - dropped_task_ids: task IDs you are deliberately skipping (low priority / energy overflow)
        - message_to_waiting_person: REQUIRED if waiting person is not None (min 25 words)
        - reasoning: explain your choices

        Remember: energy sum of ordered tasks must be <= {energy}.
        Remember: if task X has depends_on=Y, Y must come before X in ordered_task_ids.

        Respond in JSON only.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Parse JSON, handling markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        parsed = json.loads(text)
        return parsed

    except json.JSONDecodeError as exc:
        print(f"[DEBUG] JSON parsing failed: {exc}. Response: {text[:100]}...", flush=True)
        # Fallback action on parse failure
        return {
            "ordered_task_ids": ["T1", "T2", "T3"],
            "dropped_task_ids": [],
            "message_to_waiting_person": "I will update you soon.",
            "reasoning": "Fallback action due to parse error."
        }
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {type(exc).__name__}: {exc}", flush=True)
        # Fallback action on API failure
        return {
            "ordered_task_ids": ["T1", "T2", "T3"],
            "dropped_task_ids": [],
            "message_to_waiting_person": "I will update you soon.",
            "reasoning": "Fallback action due to API error."
        }


async def run_level(client: OpenAI, env, level: str) -> float:
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    
    # Latency tracking
    task_spawn_steps = {} # tid -> step_count
    latencies = []

    log_start(task=level, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(level=level)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
            
            # Track spawns
            for t in obs.tasks:
                if t['id'] not in task_spawn_steps:
                    task_spawn_steps[t['id']] = step

            parsed = get_model_action(client, {
                "level": obs.level,
                "available_energy": obs.available_energy,
                "waiting_person": obs.waiting_person,
                "tasks": obs.tasks,
                "current_step": obs.current_step,
                "max_steps": obs.max_steps,
            })

            action = PriorityPanicAction(
                ordered_task_ids=parsed.get("ordered_task_ids", []),
                dropped_task_ids=parsed.get("dropped_task_ids", []),
                message_to_waiting_person=parsed.get("message_to_waiting_person", ""),
                reasoning=parsed.get("reasoning", ""),
            )
            
            # Before stepping, keep track of tasks currently in observation
            remaining_before = {t['id'] for t in obs.tasks}

            result = await env.step(action)
            obs = result.observation
            
            # Track completions/removals
            remaining_after = {t['id'] for t in obs.tasks}
            completed_or_dropped = remaining_before - remaining_after
            for tid in completed_or_dropped:
                if tid in task_spawn_steps:
                    latency = step - task_spawn_steps[tid] + 1
                    latencies.append(latency)
                    # We don't want to recount it if it's somehow re-spawned (unlikely in this env)
                    del task_spawn_steps[tid]

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=str(parsed.get("ordered_task_ids", [])),
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        score = sum(rewards)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        print(f"[DEBUG] Level {level} | Average Task Latency: {avg_latency:.2f} steps | Cumulative Reward: {score:.3f}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    """Main inference loop: validate config, connect to environment, run benchmarks."""
    # Validate configuration before starting
    if not validate_config():
        sys.exit(1)
    
    print(f"[DEBUG] Config validated. Using model: {MODEL_NAME}", flush=True)
    print(f"[DEBUG] API endpoint: {API_BASE_URL}", flush=True)
    
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[DEBUG] Failed to initialize OpenAI client: {exc}", flush=True)
        print("[END] success=false steps=0 score=0.0 rewards=", flush=True)
        return

    if IMAGE_NAME:
        print(f"[DEBUG] Connecting to Docker image: {IMAGE_NAME}", flush=True)
        try:
            env = await PriorityPanicEnv.from_docker_image(IMAGE_NAME)
        except Exception as exc:
            print(f"[DEBUG] Docker connection failed: {exc}", flush=True)
            print("[END] success=false steps=0 score=0.0 rewards=", flush=True)
            return
    else:
        hf_space = os.getenv("HF_SPACE_URL", "")
        if hf_space:
            print(f"[DEBUG] Connecting to HF Space: {hf_space}", flush=True)
            env = PriorityPanicEnv(base_url=hf_space)
        else:
            print("[DEBUG] ERROR: No IMAGE_NAME or HF_SPACE_URL provided", flush=True)
            print("[END] success=false steps=0 score=0.0 rewards=", flush=True)
            return

    async with env:
        levels = ["easy", "medium", "hard"]
        all_scores = []

        for level in levels:
            score = await run_level(client, env, level)
            all_scores.append(score)
            print(f"[DEBUG] Level {level} complete. Score: {score:.3f}", flush=True)

        final_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(f"[DEBUG] Final average score: {final_score:.3f}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[DEBUG] Interrupted by user", flush=True)
        sys.exit(1)
    except Exception as exc:
        print(f"[DEBUG] Unexpected error: {type(exc).__name__}: {exc}", flush=True)
        print("[END] success=false steps=0 score=0.0 rewards=", flush=True)
        sys.exit(1)