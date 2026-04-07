# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the Priority Panic Environment.
Optimized for Hugging Face Spaces and OpenEnv 'India AI Hackathon 26'.
"""

import os
import sys
import uvicorn

# 1. Path Standardization
# This ensures that the root directory is in the python path,
# making 'models' and 'server' importable as top-level modules.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv-core is missing. Run: pip install openenv-core"
    ) from e

# 2. Robust Module Imports
# We attempt top-level imports first (standard for Docker/Production),
# then fall back to relative imports for local testing script execution.
try:
    from models import PriorityPanicAction, PriorityPanicObservation
    from server.priority_panic_environment import PriorityPanicEnvironment
except (ImportError, ModuleNotFoundError):
    try:
        from ..models import PriorityPanicAction, PriorityPanicObservation
        from .priority_panic_environment import PriorityPanicEnvironment
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[CRITICAL] could not find models or environment: {e}")
        raise

# 3. App Initialization
# We set max_concurrent_envs=1 to prevent memory issues on Free Tier HF Spaces.
app = create_app(
    PriorityPanicEnvironment,
    PriorityPanicAction,
    PriorityPanicObservation,
    env_name="priority_panic",
    max_concurrent_envs=1,
)

def main():
    """
    Entry point for the environment server.
    Note: Port 7860 is the MANDATORY port for Hugging Face Spaces.
    """
    print("--- Starting Priority Panic Server ---")
    print("Interface: http://0.0.0.0:7860")
    
    uvicorn.run(
        "server.app:app", 
        host="0.0.0.0", 
        port=7860, 
        log_level="info",
        reload=False  # Set to True only during local dev
    )

if __name__ == "__main__":
    main()