# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Priority Panic Environment.

This module creates an HTTP server that exposes the PriorityPanicEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os
import sys

# Standardize pathing: This ensures the 'server' and 'models' folders are visible
# no matter where the command is run from.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with 'pip install openenv-core' or 'uv sync'"
    ) from e

# Robust imports: Handles both local execution and Docker/HF Spaces
try:
    from models import PriorityPanicAction, PriorityPanicObservation
    from server.priority_panic_environment import PriorityPanicEnvironment
except (ImportError, ModuleNotFoundError):
    from ..models import PriorityPanicAction, PriorityPanicObservation
    from .priority_panic_environment import PriorityPanicEnvironment

# Create the app instance
app = create_app(
    PriorityPanicEnvironment,
    PriorityPanicAction,
    PriorityPanicObservation,
    env_name="priority_panic",
    max_concurrent_envs=1,
)

def main():
    """
    Standard entry point for openenv validate.
    This must be callable with ZERO arguments.
    """
    import uvicorn
    # Port 7860 is the required standard for Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()