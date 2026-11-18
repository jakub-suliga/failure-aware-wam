"""Repository locations. Importing LingBot-VA modules needs its wan_va/ directory on sys.path (upstream layout)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINGBOT = ROOT / "lingbot-va"
ROBOTWIN = ROOT / "RoboTwin"
CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
WRIST_CAMS = ["cam_left_wrist", "cam_right_wrist"]


def add_lingbot_to_path():
    p = str(LINGBOT / "wan_va")
    if p not in sys.path:
        sys.path.insert(0, p)


def free_port() -> int:
    """A TCP port that is free right now (fixed pid-based ports collided between jobs sharing a node)."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
