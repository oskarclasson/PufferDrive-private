#!/bin/sh
# Starts the policy server with the container arguments in the background, then AD in the foreground.
set -e
/opt/policy/bin/python -m pufferlib.ocean.cosim.simian.server "$@" &
exec /app/ad
