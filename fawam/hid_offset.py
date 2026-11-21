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

TYPES = ("trans", "gain", "yaw", "trans_z", "lat", "fixed")


def draw(seed, kind, p_zero=0.2, fixed=None, variant=0):
    """Deterministic in (seed, kind): the same offset for every attempt on this seed.
    p20/p21 (plan v2, 2026-11-21): kind "fixed" -> `fixed` = dict(type=trans|gain, o=[...]) for every seed (D2 grid);
    kind "lat" (S-B) -> lateral offset y = s * U[3, 5] cm with s = +1 for even seeds, -1 for odd (both signs
    balanced), plus x = s_x * U[3, 5] cm on half of the seeds (s_x again balanced), z = 0; never zero."""
    if kind == "fixed":
        return dict(type=fixed["type"], o=[float(x) for x in fixed["o"]])
    if kind == "lat":
        # variant 1 (p21 counter-examples): SAME scene (seed), opposite sign, independent magnitude
        h = int(hashlib.sha1(f"hid/lat/{int(seed)}{'/v%d' % variant if variant else ''}".encode()).hexdigest()[:12], 16)
        rng = np.random.default_rng(h)
        s = (1.0 if int(seed) % 2 == 0 else -1.0) * (-1.0 if variant % 2 else 1.0)
        y = s * rng.uniform(0.03, 0.05)
        x = 0.0
        if rng.random() < 0.5:
            x = (1.0 if rng.random() < 0.5 else -1.0) * rng.uniform(0.03, 0.05)
        return dict(type="trans", o=[float(x), float(y), 0.0])
    h = int(hashlib.sha1(f"hid/{kind}/{int(seed)}".encode()).hexdigest()[:12], 16)
    rng = np.random.default_rng(h)
    if rng.random() < p_zero:
        return dict(type=kind, o=[0.0, 0.0, 0.0] if kind in ("trans", "trans_z") else [1.0] if kind == "gain" else [0.0])
    while True:
        if kind == "trans":
            o = [rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(-0.10, 0.10)]
            if np.linalg.norm(o) >= 0.01:
                return dict(type=kind, o=[float(x) for x in o])
        elif kind == "trans_z":  # p15 (2026-11-20): height-only offset, the hid analogue of the table-height failure
            z = rng.uniform(-0.10, 0.10)
            if abs(z) >= 0.01:
                return dict(type="trans", o=[0.0, 0.0, float(z)])
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
    """Patch RoboTwin: draw o in _init_task_env_ (from the seed), add it to every EE action in take_action.
    (setup_demo is defined per task subclass; every one calls Base_Task._init_task_env_(**kwargs) with the seed.)"""
    kind = args.get("hid_offset")
    if not kind:
        return False
    assert kind in TYPES, kind
    p_zero = float(args.get("hid_p_zero", 0.2))
    fixed = dict(type=args.get("hid_type", "trans"), o=list(args.get("hid_o", [0, 0, 0]))) if kind == "fixed" else None
    from envs._base_task import Base_Task
    from fawam.robotwin_hooks import ExpertTrajectory

    orig_setup, orig_take = Base_Task._init_task_env_, Base_Task.take_action

    def init_task_env(self, *a, **k):
        out = orig_setup(self, *a, **k)
        self._hid = draw(k.get("seed", 0), kind, p_zero, fixed, int(args.get("hid_variant", 0)))
        self._hid_init = None  # start pose, taken at the first action (after the robot settled)
        return out

    def take_action(self, action, action_type="qpos"):
        h = getattr(self, "_hid", None)
        if action_type == "ee" and not is_zero(h):
            if self._hid_init is None:
                self._hid_init = ExpertTrajectory.current_pose(self)
            action = apply(action, h, self._hid_init)
        return orig_take(self, action, action_type=action_type)

    Base_Task._init_task_env_, Base_Task.take_action = init_task_env, take_action
    print(f"hid_offset: {kind}, p_zero={p_zero}")
    return True
