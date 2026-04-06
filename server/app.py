# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import sys

# FIX: Add the project root to sys.path so 'models' and 'server' can be found 
# regardless of where uvicorn is started.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

# FIX: Use standard imports now that root_dir is in sys.path
try:
    from models import PriorityPanicAction, PriorityPanicObservation
    from server.priority_panic_environment import PriorityPanicEnvironment
except ImportError:
    # Fallback for specific local execution contexts
    from .models import PriorityPanicAction, PriorityPanicObservation
    from .priority_panic_environment import PriorityPanicEnvironment

# Create the app with web interface and README integration
app = create_app(
    PriorityPanicEnvironment,
    PriorityPanicAction,
    PriorityPanicObservation,
    env_name="priority_panic",
    max_concurrent_envs=1, 
)

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)