# Priority Panic: Hackathon Submission Guide

This guide walks you through submitting Priority Panic to the **Scaler × Meta × HuggingFace × Unsloth Hackathon**.

## Pre-Submission Checklist (MANDATORY)

All items must pass or your submission will be **DISQUALIFIED**.

- [ ] **HF Space deploys** — Automated ping returns 200 + responds to reset()
- [ ] **OpenEnv spec compliance** — Validates openenv.yaml, models, step()/reset()/state() endpoints
- [ ] **Dockerfile builds** — `docker build .` succeeds without errors
- [ ] **Baseline reproduces** — `python inference.py` completes without error in < 20 min
- [ ] **3+ tasks with graders** — Easy (5), Medium (7), Hard (8) tasks all grade correctly
- [ ] **Structured logs** — Output follows [START]/[STEP]/[END] format exactly
- [ ] **Environment variables** — API_BASE_URL, MODEL_NAME, HF_TOKEN all configurable

## Step 1: Local Testing

### 1.1 Setup Environment
```bash
# Install dependencies
pip install -e .

# Create .env file (copy from .env.example)
cp .env.example .env

# Edit .env and add your HuggingFace token
# HF_TOKEN=hf_YOUR_TOKEN_HERE
```

### 1.2 Test Docker Build
```bash
# Build image (this must succeed)
docker build -t priority_panic-env:latest -f server/Dockerfile .

# Should output: "Successfully tagged priority_panic-env:latest"
```

### 1.3 Test Inference Script
```bash
# Source your .env file
export $(cat .env | grep -v '^#' | xargs)

# Run inference (should complete in < 20 minutes)
time python inference.py

# Expected output:
# [START] task=easy env=priority_panic model=Qwen/Qwen2.5-72B-Instruct
# [STEP] step=1 action=[...] reward=0.XX done=false error=null
# ...
# [END] success=true steps=3 score=0.XXX rewards=0.XX,0.XX,0.XX
# [DEBUG] Level easy complete. Score: 0.XXX
# ...
```

### 1.4 Test Environment Servers Locally
```bash
# Terminal 1: Start server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Test endpoints
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"level": "easy"}'

# Should return:
# {"tasks": [...], "available_energy": 5, "waiting_person": "", "level": "easy", ...}
```

## Step 2: Deploy to HuggingFace Spaces

### 2.1 Create HuggingFace Repository

1. Go to https://huggingface.co/new
2. Create a new Space:
   - **Name**: `priority-panic-env` (or your preferred name)
   - **License**: `openrail`
   - **Space SDK**: `Docker`
   - **Private**: Yes (recommended for development)

3. In Space settings → README, verify it includes:
   ```yaml
   ---
   title: Priority Panic Environment Server
   emoji: 🎲
   colorFrom: pink
   colorTo: yellow
   sdk: docker
   pinned: false
   app_port: 8000
   tags:
     - openenv
   ---
   ```

### 2.2 Push Your Code

```bash
# Install HuggingFace CLI if not already installed
pip install huggingface_hub

# Login to HuggingFace
huggingface-cli login
# (paste your token when prompted)

# Initialize git (if not already done)
git init
git add .
git commit -m "Initial Priority Panic submission"

# Add HuggingFace remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/priority-panic-env

# Push to HuggingFace
git push hf main
```

### 2.3 Monitor Deployment

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/priority-panic-env`
2. Watch the "Logs" tab for build progress
3. Once it says "Running", the Space is deployed

### 2.4 Test Deployed Space

```bash
# Verify the Space is responding
curl https://YOUR_USERNAME-priority-panic-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{}'

# Should return 200 with observation JSON
```

## Step 3: Test Inference Against Deployed Space

```bash
export HF_SPACE_URL="https://YOUR_USERNAME-priority-panic-env.hf.space"
export HF_TOKEN="hf_YOUR_TOKEN"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

python inference.py
```

## Step 4: Pre-Submission Validation

The hackathon provides a validation script. Run it:

```bash
# Run pre-submission validator
python scripts/validate_submission.py \
  --space-url https://YOUR_USERNAME-priority-panic-env.hf.space \
  --inference-dir . \
  --timeout 1200  # 20 minutes

# Should output:
# ✓ Space responds to ping
# ✓ reset() returns valid observation
# ✓ step() executes and returns reward
# ✓ Dockerfile builds successfully
# ✓ inference.py completes < 20min
# ✓ Structured logs correct format
# ✓ All environment variables present
```

## Step 5: Submit on Hackathon Platform

1. Go to https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard
2. Click **"Submit Assessment"** (Round 1)
3. Fill in the form:
   - **Environment Name**: Priority Panic
   - **HuggingFace Space URL**: https://YOUR_USERNAME-priority-panic-env.hf.space
   - **GitHub Repository**: (Link to your code)
   - **Description**: Brief description of your environment
4. Click **"Submit"**
5. **Deadline**: 8 April 2026, 11:59 PM IST ⏰

## Troubleshooting

### Docker Build Fails
```bash
# Check Dockerfile syntax
docker build --no-cache -t priority_panic-env .

# See detailed error output
docker build -t priority_panic-env . 2>&1 | tail -50
```

### Space Not Responding
- Check HuggingFace Space "Logs" tab for build errors
- Try restarting the Space: Settings → Restart Space
- Verify all dependencies in requirements.txt

### Inference Script Timeout
```bash
# Run with debug output
export DEBUG=1
python inference.py

# Check if model is available
curl -I https://router.huggingface.co/v1/models/Qwen/Qwen2.5-72B-Instruct
```

### Inference Script Fails Parsing
```bash
# Test JSON parsing
python -c "
import json
response = '{\"ordered_task_ids\": [\"T1\"], \"dropped_task_ids\": []}'
print(json.loads(response))
"
```

## Environment Variables Reference

| Variable | Required | Default | Example |
|----------|----------|---------|---------|
| `HF_TOKEN` | Yes | - | `hf_AJnz...` |
| `API_BASE_URL` | Yes | `https://router.huggingface.co/v1` | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Yes | `Qwen/Qwen2.5-72B-Instruct` | `meta-llama/Llama-2-70b-chat-hf` |
| `LOCAL_IMAGE_NAME` | No | - | `priority_panic-env:latest` |
| `HF_SPACE_URL` | No | - | `https://username-priority-panic.hf.space` |

## Key Files for Submission

```
priority_panic/
├── .env.example              # Environment template (required for setup)
├── .gitignore                # Git ignore file
├── README.md                 # Full documentation (required)
├── DEPLOYMENT.md             # This file
├── openenv.yaml              # Environment manifest (required)
├── pyproject.toml            # Python dependencies (required)
├── inference.py              # Benchmark script (required)
├── models.py                 # Data models (required)
├── client.py                 # Client implementation (required)
├── server/
│   ├── app.py                # FastAPI server (required)
│   ├── priority_panic_environment.py  # Environment logic (required)
│   ├── Dockerfile            # Container definition (required)
│   └── requirements.txt       # Server dependencies (required)
└── __init__.py               # Package init
```

## Success Criteria

Your submission will be evaluated on:

1. **Does it run?** — No syntax errors, all dependencies install
2. **Does it respond?** — /reset, /step, /state endpoints work
3. **Is it correctly graded?** — Rewards are between 0-1, logic is sound
4. **Is it realistic?** — Task prioritization is a real-world problem
5. **Is it complete?** — All 3 difficulty levels implemented
6. **Can I reproduce it?** — inference.py gives consistent results
7. **Is it packaged well?** — Docker builds, README is clear, code is clean

Good luck! 🚀
