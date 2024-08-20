"""Replay the trajectory stored in HDF5.
The replayed trajectory can use different observation modes and control modes.
We support translating actions from certain controllers to a limited number of controllers.
The script is only tested for Panda, and may include some Panda-specific hardcode.
"""

import argparse
import multiprocessing as mp
import os
from copy import deepcopy
from typing import Union

import gym as gym
import h5py
import numpy as np
import sapien
from tqdm.auto import tqdm
from transforms3d.quaternions import quat2axangle

import mani_skill.envs
from mani_skill.agents.controllers import *
from mani_skill.agents.controllers.base_controller import CombinedController
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.trajectory.merge_trajectory import merge_h5
from mani_skill.utils import common, gym_utils, io_utils, wrappers
from mani_skill.utils.structs.link import Link

import sys
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_PATH)
os.environ["CUSTOM_ASSET_DIR"] = os.path.join(PROJECT_PATH, "custom_assets")
from custom_tasks import PosPushTEnv
from replay_buffer import ReplayBuffer




import h5py
import numpy as np


from mani_skill.utils.structs.types import Array


def _get_dict_len(x):
    if isinstance(x, dict) or isinstance(x, h5py.Group):
        for k in x.keys():
            return _get_dict_len(x[k])
    else:
        return len(x)


def index_dict(x, i):
    res = dict()
    if isinstance(x, dict) or isinstance(x, h5py.Group):
        for k in x.keys():
            res[k] = index_dict(x[k], i)
        return res
    else:
        return x[i]


def dict_to_list_of_dicts(x):
    result = []
    N = _get_dict_len(x)
    for i in tqdm(range(N), desc=f"Converting dict {x} to list of dicts"):
        result.append(index_dict(x, i))
    return result



def main(args, proc_id: int = 0, num_procs=1, pbar=None):
    pbar = tqdm(position=proc_id, leave=None, unit="step", dynamic_ncols=True)

    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(args.replay_buffer_output, mode='a')

    # Load HDF5 containing trajectories
    traj_path = args.traj_path
    ori_h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)

    env_info = json_data["env_info"]
    obs_mode = args.obs_mode
    control_mode = args.control_mode
    if pbar is not None:
        pbar.set_postfix(
            {
                "control_mode": control_mode,
                "obs_mode": obs_mode
            }
        )
    # Prepare for recording
    episodes = json_data["episodes"][: args.count]
    n_ep = len(episodes)
    inds = np.arange(n_ep)
    inds = np.array_split(inds, num_procs)[proc_id]

    # Replay
    for ind in inds:
        ep = episodes[ind]
        episode_id = ep["episode_id"]
        traj_id = f"traj_{episode_id}"
        if pbar is not None:
            pbar.set_description(f"Replaying {traj_id}")

        if traj_id not in ori_h5_file:
            tqdm.write(f"{traj_id} does not exist in {traj_path}")
            continue

        # Original actions to replay
        ori_actions = ori_h5_file[traj_id]["actions"][:]
        orig_obs = dict_to_list_of_dicts(ori_h5_file['traj_0']['obs'])
        n = len(ori_actions)
        if pbar is not None:
            pbar.reset(total=n)

        ep_datas = []
        for t, a in enumerate(ori_actions):
            base_camera_img = orig_obs[t]['sensor_data']['base_camera']['rgb']
            base_camera_segmentation = orig_obs[t]['sensor_data']['base_camera']['segmentation']
            hand_camera_img = orig_obs[t]['sensor_data']['hand_camera']['rgb']
            hand_camera_segmentation = orig_obs[t]['sensor_data']['hand_camera']['segmentation']
            robot_tcp_pose = orig_obs[t]['extra']['tcp_pose'] # dim=7, (x,y,z,quat)
            tee_object_pose = orig_obs[t]['extra']['obj_pose'] # dim=7, (x,y,z,quat)
            goal_pose = orig_obs[t]['extra']['goal_pos'] # dim=7, (x,y,z,quat)
            # pd_ee_delta_pose mode, action dim=6, (dx,dy,dz,droll,dpitch,dyaw)
            action = a
            if pbar is not None:
                pbar.update()
            data = {
                    'base_camera_img': base_camera_img,
                    'base_camera_segmentation': base_camera_segmentation,
                    'hand_camera_img': hand_camera_img,
                    'hand_camera_segmentation': hand_camera_segmentation,
                    'action': action,
                    'robot_tcp_pose': robot_tcp_pose,
                    'tee_object_pose': tee_object_pose,
                    'goal_pose': goal_pose
                }
            ep_datas.append(data)
        # save episode buffer to replay buffer (on disk)
        data_dict = dict()
        for key in ep_datas[0].keys():
            data_dict[key] = np.stack([x[key] for x in ep_datas])
        replay_buffer.add_episode(data_dict, compressors='disk')

    ori_h5_file.close()

    if pbar is not None:
        pbar.close()

    return replay_buffer


if __name__ == "__main__":
    # OmegaConf
    from omegaconf import OmegaConf
    args = OmegaConf.create({
        "traj_path": "/nvmessd/yinzi/pusht_3dsim/demos/PosPushT-v1/motionplanning/20240817_100620.h5",
        "obs_mode": "rgb",
        "control_mode": "pd_ee_delta_pose",
        "target_control_mode": None,
        "sim_backend": None,
        "shader": None,
        "reward_mode": "dense",
        "render_mode": "rgb_array",
        "save_traj": True,
        "save_video": False,
        "video_fps": 30,
        "record_rewards": False,
        "count": 1,
        "max_retry": 0,
        "use_first_env_state": False,
        "use_env_states": False,
        "discard_timeout": False,
        "allow_failure": False,
        "verbose": False,
        "vis": False,
        "num_procs": 1,
        "replay_buffer_output": "/nvmessd/yinzi/pusht_3dsim/diff_out/20240817_100620"
    })
    main(args)
