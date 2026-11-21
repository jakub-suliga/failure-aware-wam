"""Simulator disturbances that RoboTwin's task config does not offer (configs/robotwin_task_config/dist_*.yml).

robot_init_noise: <std in rad>   Gaussian noise on the home posture of all arm joints of both arms, clipped to
    +-2.25 std, drawn per episode from the episode seed. Same protocol as the "Robot" dimension of RoboTwin 2.0-Plus
    (arXiv 2603.22078: std 0.1 rad, clip 0.225 rad), where LingBot-VA drops from 92 % to 36 % success.

object_pose_noise_xy: <std in m>, object_pose_noise_yaw_deg: <max deg>   Gaussian x/y noise and a uniform yaw about
    the vertical axis on every pose the task samples for its objects (envs.utils.rand_pose). RoboTwin 2.0-Plus "Layout /
    target pose": 2 cm and +-15 deg.

object_mass_scale: [lo, hi]   every task object's mass is multiplied by a factor drawn uniformly from [lo, hi] (per
    object, per episode): heavy objects slip out of the grasp or are dropped while lifting (p7d, "dropped" failures).

gripper_stiffness_scale: <s>   drive stiffness and damping of both grippers times s (< 1 = weaker grip force): objects
    slip during lift/transport (p7d).

install(args) is called once by the evaluation client after it has read the task config; without the key it does
nothing, so every existing config behaves exactly as before.
"""
import numpy as np


def _install_object_noise(xy, yaw_deg):
    import sys
    import sapien
    import transforms3d as t3d
    import importlib
    rca = importlib.import_module("envs.utils.rand_create_actor")  # envs.utils also exports a function of that name
    orig = rca.rand_pose

    def rand_pose(*a, **k):
        p = orig(*a, **k)
        pos = np.array(p.p, dtype=np.float64)
        pos[:2] += np.random.normal(0, xy, 2) if xy > 0 else 0
        q = np.array(p.q, dtype=np.float64)
        if yaw_deg > 0:
            q = t3d.quaternions.qmult(t3d.quaternions.axangle2quat([0, 0, 1], np.deg2rad(np.random.uniform(-yaw_deg, yaw_deg))), q)
        return sapien.Pose(pos, q)

    for name, mod in list(sys.modules.items()):  # task modules imported later copy it from envs.utils
        if name.startswith("envs") and getattr(mod, "rand_pose", None) is orig:
            mod.rand_pose = rand_pose
    print(f"sim_disturb: object pose noise xy={xy} m, yaw=+-{yaw_deg} deg")


def _install_mass_scale(lo, hi):
    from envs.utils import actor_utils as au

    def wrap(cls):
        orig = cls.set_mass

        def set_mass(self, mass, *a, **k):
            return orig(self, float(mass) * float(np.random.uniform(lo, hi)), *a, **k)

        cls.set_mass = set_mass

    wrap(au.Actor)
    if "set_mass" in au.ArticulationActor.__dict__:
        wrap(au.ArticulationActor)
    print(f"sim_disturb: object mass x U[{lo}, {hi}]")


def _install_object_scale(lo, hi):
    """Every object loaded from a model_data json (create_actor / create_glb / urdf) is scaled by a factor drawn
    uniformly from [lo, hi] (per object, from the task's seeded np.random). The scale is changed in model_data itself,
    so collision, visual mesh AND the functional/contact points the expert planner uses stay consistent."""
    import json as _json
    import envs.utils.create_actor as ca

    class _ScaledJson:
        def __getattr__(self, k):
            return getattr(_json, k)

        @staticmethod
        def load(f, *a, **k):
            d = _json.load(f, *a, **k)
            if isinstance(d, dict) and "scale" in d:
                s = float(np.random.uniform(lo, hi))
                sc = d["scale"]
                d["scale"] = [float(x) * s for x in sc] if isinstance(sc, (list, tuple)) else float(sc) * s
                d["_fawam_size_factor"] = s
            return d

    ca.json = _ScaledJson()
    print(f"sim_disturb: object size x U[{lo}, {hi}]")


def _install_gripper_scale(scale):
    from envs.robot.robot import Robot
    orig = Robot.init_joints

    def init_joints(self):
        orig(self)
        for side in ("left", "right"):
            for joint in getattr(self, f"{side}_gripper"):
                joint[0].set_drive_property(stiffness=getattr(self, f"{side}_gripper_stiffness") * scale,
                                            damping=getattr(self, f"{side}_gripper_damping") * scale)

    Robot.init_joints = init_joints
    print(f"sim_disturb: gripper stiffness/damping x {scale}")


def _install_table_signed(lo, hi, variant=0):
    """S-A (plan v2, 2026-11-21): table height offset +-U[lo, hi] per seed; sign + for even seeds, - for odd (balanced),
    magnitude from a hash of the seed. Stored as env._table_bias (label / analysis only, never a model input)."""
    import hashlib
    from envs._base_task import Base_Task
    orig = Base_Task._init_task_env_

    def init_task_env(self, *a, **k):
        seed = int(k.get("seed", 0))
        rng = np.random.default_rng(int(hashlib.sha1(f"table/{seed}{'/v%d' % variant if variant else ''}".encode())
                                        .hexdigest()[:12], 16))
        sign = (1.0 if seed % 2 == 0 else -1.0) * (-1.0 if variant % 2 else 1.0)  # variant 1: same scene, other sign
        bias = sign * float(rng.uniform(lo, hi))
        k["table_height_bias"] = bias
        self._table_bias = bias
        return orig(self, *a, **k)

    Base_Task._init_task_env_ = init_task_env
    print(f"sim_disturb: table height +-U[{lo}, {hi}] (sign by seed parity)")


def install(args):
    extra = False
    ts = args.get("table_height_signed")
    if ts:
        _install_table_signed(float(ts[0]), float(ts[1]), int(args.get("table_signed_variant", 0)))
        extra = True
    osz = args.get("object_size_scale")
    if osz:
        _install_object_scale(float(osz[0]), float(osz[1]))
        extra = True
    ms = args.get("object_mass_scale")
    if ms:
        _install_mass_scale(float(ms[0]), float(ms[1]))
        extra = True
    gs = float(args.get("gripper_stiffness_scale", 0) or 0)
    if gs > 0:
        _install_gripper_scale(gs)
        extra = True
    xy, yaw = float(args.get("object_pose_noise_xy", 0) or 0), float(args.get("object_pose_noise_yaw_deg", 0) or 0)
    if xy > 0 or yaw > 0:
        _install_object_noise(xy, yaw)
    std = float(args.get("robot_init_noise", 0) or 0)
    if std <= 0:
        return extra or xy > 0 or yaw > 0
    from envs.robot.robot import Robot

    def move_to_homestate(self):
        clip = 2.25 * std
        for entity, joints, home in ((self.left_entity, self.left_arm_joints, self.left_homestate),
                                     (self.right_entity, self.right_arm_joints, self.right_homestate)):
            target = [float(h) + float(np.clip(np.random.normal(0, std), -clip, clip)) for h in home]
            for j, t in zip(joints, target):
                j.set_drive_target(t)
            try:  # put the arm there at once instead of letting the drive settle during the first steps
                active, q = entity.get_active_joints(), entity.get_qpos()
                for j, t in zip(joints, target):
                    q[active.index(j)] = t
                entity.set_qpos(q)
            except Exception as e:
                print(f"sim_disturb: set_qpos failed ({e!r}); the drive moves the arm to the noisy posture")

    Robot.move_to_homestate = move_to_homestate
    print(f"sim_disturb: robot_init_noise std={std} rad")
    return True
