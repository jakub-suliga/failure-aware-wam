"""Hidden calibration offsets (dataset `hid`, plan of 2026-11-18, section 3).

Per episode an invisible offset `o` is drawn from the episode seed, so both attempts of a seed (failed policy rollout
and expert fix) share it. It distorts the EXECUTION of every end-effector action, like a wrong hand-eye calibration:
  trans  executed position = commanded + o (world frame, both arms), z ~ U[-10, 10] cm, x/y ~ U[-5, 5] cm, |o| >= 1 cm
  gain   executed z = z0 + g * (commanded z - z0), g ~ U[0.7, 1.3] (z0 = arm height at the episode start)
  yaw    executed gripper orientation rotated about the world z axis by U[-25, 25] deg
With probability p_zero (default 0.2) o = 0: the context is then a success and nothing must change.

LingBot-VA outputs EE poses relative to the start pose and gets its OWN actions back as state, never measured
proprioception, so its inputs are in the "believed" frame by construction. The expert fix is executed through the same
offset with compensated targets: compensate() turns the expert's true targets into the believed actions the policy
would have to output (believed = inverse(o)(true)); those are what gets recorded and trained on.

Task config keys (configs/robotwin_task_config/hid_*.yml): hid_offset: trans|gain|yaw, hid_p_zero: 0.2.
The offset is stored in episode.h5 (attrs hid_type, hid_o) for training labels and analysis only -- never an input.
"""
import hashlib

import numpy as np
import transforms3d as t3d

TYPES = ("trans", "gain", "yaw")


def draw(seed, kind, p_zero=0.2):
    """Deterministic in (seed, kind): the same offset for every attempt on this seed."""
    h = int(hashlib.sha1(f"hid/{kind}/{int(seed)}".encode()).hexdigest()[:12], 16)
    rng = np.random.default_rng(h)
    if rng.random() < p_zero:
        return dict(type=kind, o=[0.0, 0.0, 0.0] if kind == "trans" else [1.0] if kind == "gain" else [0.0])
    while True:
        if kind == "trans":
            o = [rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(-0.10, 0.10)]
            if np.linalg.norm(o) >= 0.01:
                return dict(type=kind, o=[float(x) for x in o])
        elif kind == "gain":
            g = rng.uniform(0.7, 1.3)
            if abs(g - 1) >= 0.03:
                return dict(type=kind, o=[float(g)])
        elif kind == "yaw":
            a = rng.uniform(-25, 25)
            if abs(a) >= 2:
                return dict(type=kind, o=[float(a)])
        else:
            raise ValueError(kind)


def is_zero(h):
    return h is None or (h["type"] == "gain" and abs(h["o"][0] - 1) < 1e-9) or \
        (h["type"] != "gain" and not np.any(np.asarray(h["o"])))


def _arm(pose8, h, z0, inverse):
    """pose8 = x y z qw qx qy qz gripper (absolute, world)."""
    p = np.array(pose8, dtype=np.float64)
    t, o = h["type"], h["o"]
    if t == "trans":
        p[:3] += -np.asarray(o) if inverse else np.asarray(o)
    elif t == "gain":
        g = o[0]
        p[2] = z0 + (p[2] - z0) / g if inverse else z0 + g * (p[2] - z0)
    elif t == "yaw":
        a = np.deg2rad(-o[0] if inverse else o[0])
        q = t3d.quaternions.qmult(t3d.quaternions.axangle2quat([0, 0, 1], a), p[3:7])
        p[3:7] = q / np.linalg.norm(q)
    return p


def apply(abs16, h, init16, inverse=False):
    """Commanded absolute action (16: l xyz wxyz grip, r ...) -> executed (or executed -> believed with inverse)."""
    if is_zero(h):
        return np.asarray(abs16, dtype=np.float64)
    a = np.asarray(abs16, dtype=np.float64)
    return np.concatenate([_arm(a[:8], h, init16[2], inverse), _arm(a[8:], h, init16[10], inverse)])


def compensate(env, rel_actions, init16):
    """Expert's relative TRUE targets -> relative BELIEVED actions (what the policy must output under o)."""
    h = getattr(env, "_hid", None)
    if is_zero(h) or not len(rel_actions):
        return rel_actions
    from fawam.robotwin_hooks import to_relative, to_absolute
    return np.stack([to_relative(apply(to_absolute(r, init16), h, init16, inverse=True), init16) for r in rel_actions])


def install(args):
    """Patch RoboTwin: draw o in setup_demo (from the seed), add it to every EE action in take_action."""
    kind = args.get("hid_offset")
    if not kind:
        return False
    assert kind in TYPES, kind
    p_zero = float(args.get("hid_p_zero", 0.2))
    from envs._base_task import Base_Task
    from fawam.robotwin_hooks import ExpertTrajectory

    orig_setup, orig_take = Base_Task.setup_demo, Base_Task.take_action

    def setup_demo(self, *a, **k):
        out = orig_setup(self, *a, **k)
        self._hid = draw(k.get("seed", 0), kind, p_zero)
        self._hid_init = None  # start pose, taken at the first action (after the robot settled)
        return out

    def take_action(self, action, action_type="qpos"):
        h = getattr(self, "_hid", None)
        if action_type == "ee" and not is_zero(h):
            if self._hid_init is None:
                self._hid_init = ExpertTrajectory.current_pose(self)
            action = apply(action, h, self._hid_init)
        return orig_take(self, action, action_type=action_type)

    Base_Task.setup_demo, Base_Task.take_action = setup_demo, take_action
    print(f"hid_offset: {kind}, p_zero={p_zero}")
    return True
