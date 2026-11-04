"""Expert replay without the model server: capture the planner trajectory during play_once, convert it to the policy's
relative action format, convert back exactly like the eval client (add_init_pose) and execute it with
take_action(ee). The replay must solve the task (robotwin env, run from lingbot-va/):

  ROBOTWIN_ROOT=../RoboTwin PYTHONPATH=.. python -m fawam.tests.test_expert_replay pick_diverse_bottles table_h_m05 3
"""
import sys

import numpy as np

import evaluation.robotwin.eval_polict_client_openpi as C  # chdirs into RoboTwin
from fawam.robotwin_hooks import ExpertTrajectory, expert_chunk


def make_env(task, config):
    import yaml
    with open(f"./task_config/{config}.yml") as f:
        args = yaml.safe_load(f)
    args.update(task_name=task, task_config=config, ckpt_setting=0, save_root="/tmp/fawam_test", eval_mode=True)
    with open(C.os.path.join(C.CONFIGS_PATH, "_embodiment_config.yml")) as f:
        emb = yaml.safe_load(f)
    with open(C.CONFIGS_PATH + "_camera_config.yml") as f:
        cam = yaml.safe_load(f)
    args["head_camera_h"] = cam[args["camera"]["head_camera_type"]]["h"]
    args["head_camera_w"] = cam[args["camera"]["head_camera_type"]]["w"]
    rf = emb[args["embodiment"][0]]["file_path"]
    args.update(left_robot_file=rf, right_robot_file=rf, dual_arm_embodied=True,
                left_embodiment_config=C.get_embodiment_config(rf), right_embodiment_config=C.get_embodiment_config(rf))
    args["eval_video_log"] = False
    return C.class_decorator(task), args


def main(task="pick_diverse_bottles", config="table_h_m05", n=3):
    env, args = make_env(task, config)
    seed, ok, tried = 10000, 0, 0
    while tried < int(n):
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
            with ExpertTrajectory(env) as ex:
                env.play_once()
            planned_ok = env.plan_success and env.check_success()
            env.close_env()
        except Exception as e:  # unstable seed
            print(f"seed {seed}: setup failed ({e})")
            seed += 1
            continue
        if not planned_ok:
            seed += 1
            continue
        env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        obs = env.get_obs()
        init = np.array(obs['endpose']['left_endpose'] + [obs['endpose']['left_gripper']] +
                        obs['endpose']['right_endpose'] + [obs['endpose']['right_gripper']], dtype=np.float64)
        rel = ex.relative_actions(init)
        # same chunking as the client: first chunk executes frame 1 only, later chunks 2 frames x 16 actions
        step, first = 0, True
        while env.take_action_cnt < env.step_lim and not env.eval_success and step < len(rel) + 32:
            chunk = expert_chunk(rel, step, 2, 16, first=first)
            for i in range(1 if first else 0, 2):
                for j in range(16):
                    a = C.add_init_pose(chunk[:, i, j], init)
                    a = np.concatenate([a[:3], a[3:7] / np.linalg.norm(a[3:7]), a[7:11],
                                        a[11:15] / np.linalg.norm(a[11:15]), a[15:16]])
                    env.take_action(a, action_type='ee')
                    step += 1
            first = False
        succ = bool(env.eval_success)
        print(f"seed {seed}: expert frames {len(ex.poses)}, replay steps {env.take_action_cnt}, success {succ}")
        ok += succ
        tried += 1
        env.close_env()
        seed += 1
    print(f"expert replay success {ok}/{tried}")
    return ok, tried


if __name__ == "__main__":
    main(*sys.argv[1:])
