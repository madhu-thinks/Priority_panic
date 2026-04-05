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
MAX_STEPS = 3
TEMPERATURE = 0.7
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5


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
    You are a task prioritization agent.
    You will receive a list of tasks, each with an id, name, deadline, priority, energy cost, and dependencies.
    You must decide:
    1. ordered_task_ids: list of task IDs in the order you will do them
    2. dropped_task_ids: list of task IDs you are dropping (if energy is limited)
    3. message_to_waiting_person: a message to anyone waiting for your response (if applicable)
    4. reasoning: brief explanation of your decisions

    Respond ONLY in this exact JSON format, nothing else:
    {
        "ordered_task_ids": ["T1", "T3"],
        "dropped_task_ids": ["T4"],
        "message_to_waiting_person": "I will get back to you by tonight.",
        "reasoning": "Handled urgent tasks first."
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
    user_prompt = textwrap.dedent(f"""
        Current level: {observation.get('level')}
        Available energy: {observation.get('available_energy')}
        Waiting person: {observation.get('waiting_person') or 'None'}
        Tasks:
        {observation.get('tasks')}

        Make your prioritization decision now.
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

    log_start(task=level, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            parsed = get_model_action(client, {
                "level": obs.level,
                "available_energy": obs.available_energy,
                "waiting_person": obs.waiting_person,
                "tasks": obs.tasks,
            })

            action = PriorityPanicAction(
                ordered_task_ids=parsed.get("ordered_task_ids", []),
                dropped_task_ids=parsed.get("dropped_task_ids", []),
                message_to_waiting_person=parsed.get("message_to_waiting_person", ""),
                reasoning=parsed.get("reasoning", ""),
            )

            result = await env.step(action)
            obs = result.observation

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

        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
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