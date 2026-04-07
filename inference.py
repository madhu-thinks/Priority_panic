import asyncio
import json
import os
import textwrap
from dotenv import load_dotenv
from typing import Dict, List
from openai import OpenAI
from priority_panic import PriorityPanicAction, PriorityPanicEnv

# --- Configuration --- #
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"
API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
HF_SPACE_URL = os.getenv("HF_SPACE_URL") or "https://madhubuilds-priority-panic.hf.space"
# --- DEBUG PRINT (Add this temporarily to see what's happening) ---
if not API_KEY:
    print("[DEBUG] HF_TOKEN is missing from .env or .env is not found!")
if not HF_SPACE_URL:
    print("[DEBUG] HF_SPACE_URL is missing from .env!")
# --- Optimized System Prompt --- #
SYSTEM_PROMPT = textwrap.dedent("""
    You are an Elite Efficiency AI. Your performance is graded on 'Value per Energy' (VPE).
    
    GOALS:
    1. CRITICAL PRIORITIZATION: T1 and S3 tasks are "Panic" triggers. Solve them first or lose the streak.
    2. MAXIMIZE DENSITY: Do not leave energy unused. If 5 energy remains, find a 5-energy task.
    3. MINIMAL LATENCY: You only have 8 steps. Every idle step is a massive failure.
    
    LOGIC:
    - Sort by (Priority Weight / Energy Cost).
    - Maintain the Success Streak at all costs to compound rewards.

    RESPONSE FORMAT (STRICT JSON):
    {
        "ordered_task_ids": ["ID1", "ID2"],
        "reasoning": "Briefly state the VPE logic used."
    }
""").strip()

async def run_level(client: OpenAI, env: PriorityPanicEnv, level: str) -> float:
    rewards = []
    
    # [MANDATORY] Start logging
    print(f"[START] task={level} env=priority_panic model={MODEL_NAME}", flush=True)
    
    result = await env.reset(level=level)
    obs = result.observation
    
    MAX_STEPS = 8
    success = False
    
    for step in range(1, MAX_STEPS + 1):
        task_list = obs.tasks if obs.tasks else "Monitoring..."
        
        user_prompt = (
            f"STEP: {step}/{MAX_STEPS} | ENERGY: {obs.available_energy}\n"
            f"TASKS: {task_list}"
        )
        
        error_msg = "null"
        action_str = "None"
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0, 
                response_format={"type": "json_object"} 
            )
            
            parsed = json.loads(completion.choices[0].message.content)
            ordered_ids = [str(tid) for tid in parsed.get("ordered_task_ids", [])]
            action_str = f"process({ordered_ids})"
            
            action = PriorityPanicAction(
                ordered_task_ids=ordered_ids,
                reasoning=parsed.get("reasoning", "Optimizing VPE.")
            )
        except Exception as e:
            error_msg = str(e)
            action = PriorityPanicAction(ordered_task_ids=[], reasoning="Error fallback.")

        result = await env.step(action)
        obs = result.observation
        rewards.append(result.reward)
        
        # [MANDATORY] Step logging
        print(f"[STEP] step={step} action={action_str} reward={result.reward:.2f} done={str(result.done).lower()} error={error_msg}", flush=True)
        
        if result.done:
            success = True
            break

    # [MANDATORY] End logging
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    score = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"[END] success={str(success).lower()} steps={len(rewards)} score={score:.2f} rewards={rewards_str}", flush=True)
    
    return score

async def main():
    if not API_KEY or not HF_SPACE_URL:
        print("[ERROR] Environment variables API_KEY and HF_SPACE_URL are required.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = PriorityPanicEnv(base_url=HF_SPACE_URL)

    async with env:
        levels = ["easy", "medium", "hard"]
        scores = []
        
        for level in levels:
            scores.append(await run_level(client, env, level))
        
        print("\n" + "="*35)
        print(f"AGGREGATED EFFICIENCY: {sum(scores)/len(scores):.4f}")
        print("="*35)

if __name__ == "__main__":
    asyncio.run(main())