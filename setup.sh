#!/usr/bin/env bash
# Clone the upstream repos at the pinned commits and apply the local changes.
set -euo pipefail
cd "$(dirname "$0")"

LINGBOT_COMMIT=7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb
ROBOTWIN_COMMIT=2eeec322d95799f537cbfe5f291a8220d965ccb8

if [ ! -d lingbot-va ]; then
  git clone https://github.com/Robbyant/lingbot-va.git
  git -C lingbot-va checkout "$LINGBOT_COMMIT"
  git -C lingbot-va apply ../patches/lingbot-va.patch
fi

if [ ! -d RoboTwin ]; then
  git clone https://github.com/RoboTwin-Platform/RoboTwin.git
  git -C RoboTwin checkout "$ROBOTWIN_COMMIT"
fi
cp configs/robotwin_task_config/*.yml RoboTwin/task_config/

echo "Done. Checkpoints, RoboTwin assets and cuRobo still need to be installed (see README)."
