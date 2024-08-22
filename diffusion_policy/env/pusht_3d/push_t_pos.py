import sys
import os
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_PATH)
CUSTOM_ASSET_DIR = os.path.join(PROJECT_PATH, "custom_assets")
os.environ["CUSTOM_ASSET_DIR"] = CUSTOM_ASSET_DIR

from typing import Any, Dict, Union, Optional, Sequence, Tuple
from mani_skill.envs.tasks import PushTEnv
from mani_skill.envs.tasks.tabletop.push_t import WhiteTableSceneBuilder
from mani_skill.utils.structs.types import Array
from mani_skill.utils.registration import register_env
from panda_stick_wrist_camera import WristCameraPandaStick
from mani_skill.agents.robots import PandaStick
from mani_skill.utils.logging_utils import logger
import torch
import sapien
import numpy as np
from gym import spaces
from gym.vector.utils.spaces import batch_space
from functools import cached_property

from transforms3d.quaternions import quat2axangle
from transforms3d.quaternions import qinverse, qmult


def compact_axis_angle_from_quaternion(quat: np.ndarray) -> np.ndarray:
    theta, omega = quat2axangle(quat)
    # - 2 * np.pi to make the angle symmetrical around 0
    if omega > np.pi:
        omega = omega - 2 * np.pi
    return omega * theta


def get_dtype_bounds(dtype: np.dtype):
    """Gets the min and max values of a given numpy type"""
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.bool_):
        return 0, 1
    else:
        raise TypeError(dtype)


def convert_observation_to_space(observation, prefix="", unbatched=False):
    """Convert observation to OpenAI gym observation space (recursively).
    Modified from `gym.envs.mujoco_env`
    """
    if isinstance(observation, (dict)):
        # CATUION: Explicitly create a list of key-value tuples
        # Otherwise, spaces.Dict will sort keys if a dict is provided
        space = spaces.Dict(
            [
                (
                    k,
                    convert_observation_to_space(
                        v, prefix + "/" + k, unbatched=unbatched
                    ),
                )
                for k, v in observation.items()
            ]
        )
    elif isinstance(observation, np.ndarray):
        if unbatched:
            shape = observation.shape[1:]
        else:
            shape = observation.shape
        dtype = observation.dtype
        low, high = get_dtype_bounds(dtype)
        if np.issubdtype(dtype, np.floating):
            low, high = -np.inf, np.inf
        space = spaces.Box(low, high, shape=shape, dtype=dtype)
    elif isinstance(observation, (float, np.float32, np.float64)):
        logger.debug(f"The observation ({prefix}) is a (float) scalar")
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=np.float32)
    elif isinstance(observation, (int, np.int32, np.int64)):
        logger.debug(f"The observation ({prefix}) is a (integer) scalar")
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=int)
    elif isinstance(observation, (bool, np.bool_)):
        logger.debug(f"The observation ({prefix}) is a (bool) scalar")
        space = spaces.Box(0, 1, shape=[1], dtype=np.bool_)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

def _to_numpy(array: Union[Array, Sequence]) -> np.ndarray:
    if isinstance(array, (dict)):
        return {k: _to_numpy(v) for k, v in array.items()}
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    if (
        isinstance(array, np.ndarray)
        or isinstance(array, bool)
        or isinstance(array, str)
        or isinstance(array, float)
        or isinstance(array, int)
    ):
        return array
    else:
        return np.array(array)


def to_numpy(array: Union[Array, Sequence], dtype=None) -> np.ndarray:
    array = _to_numpy(array)
    if dtype is not None:
        return array.astype(dtype)
    return array



class MyWhiteTableSceneBuilder(WhiteTableSceneBuilder):
    def initialize(self, env_idx: torch.Tensor):
        super().initialize(env_idx)
        b = len(env_idx)
        if self.env.robot_uids == "wrist_camera_panda_stick":
            qpos = np.array([0.662,0.212,0.086,-2.685,-.115,2.898,1.673,])
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
    

@register_env("PosPushT-v1", max_episode_steps=100)
class PosPushTEnv(PushTEnv):
    SUPPORTED_ROBOTS = ["wrist_camera_panda_stick"]
    agent: Union[WristCameraPandaStick]
    #T block design choices
    T_mass = 1.0
    T_dynamic_friction = 1
    T_static_friction = 1

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # reward for overlap of the tees
        reward = self.pseudo_render_intersection()
        return reward
    
    def __init__(self, *args, robot_uids="wrist_camera_panda_stick", robot_init_qpos_noise=0.02,**kwargs):
        super(PosPushTEnv, self).__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=self.action_space.dtype)

    @cached_property
    def single_observation_space(self):
        return convert_observation_to_space(to_numpy(self._init_raw_obs), unbatched=True)

    @cached_property
    def observation_space(self):
        return batch_space(self.single_observation_space, n=self.num_envs)
    
    def _load_scene(self, options: dict):
        # have to put these parmaeters to device - defined before we had access to device
        # load scene is a convienent place for this one time operation
        self.ee_starting_pos2D = self.ee_starting_pos2D.to(self.device)
        self.ee_starting_pos3D = self.ee_starting_pos3D.to(self.device)

        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = MyWhiteTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # returns 3d cad of create_tee - center of mass at (0,0,0)
        # cad Tee is upside down (both 3D tee and target)
        TARGET_RED = np.array([194, 19, 22, 255]) / 255 # same as mani_skill.utils.building.actors.common - goal target
        def create_tee(name="tee", target=False, base_color=TARGET_RED):
            # dimensions of boxes that make tee 
            # box2 is same as box1, except (3/4) the lenght, and rotated 90 degrees
            # these dimensions are an exact replica of the 3D tee model given by diffusion policy: https://cad.onshape.com/documents/f1140134e38f6ed6902648d5/w/a78cf81827600e4ff4058d03/e/f35f57fb7589f72e05c76caf
            box1_half_w = 0.2/2
            box1_half_h = 0.05/2
            half_thickness = 0.04/2 if not target else 1e-4

            # we have to center tee at its com so rotations are applied to com
            # vertical block is (3/4) size of horizontal block, so
            # center of mass is (1*com_horiz + (3/4)*com_vert) / (1+(3/4))
            # # center of mass is (1*(0,0)) + (3/4)*(0,(.025+.15)/2)) / (1+(3/4)) = (0,0.0375)
            com_y = 0.0375
            
            builder = self.scene.create_actor_builder()
            first_block_pose = sapien.Pose([0., 0.-com_y, 0.])
            first_block_size = [box1_half_w, box1_half_h, half_thickness]
            if not target:
                builder._mass = self.T_mass
                tee_material = sapien.pysapien.physx.PhysxMaterial(
                    static_friction=self.T_dynamic_friction, 
                    dynamic_friction=self.T_static_friction, 
                    restitution=0
                )
                builder.add_box_collision(pose=first_block_pose, half_size=first_block_size, material=tee_material)
                #builder.add_box_collision(pose=first_block_pose, half_size=first_block_size)
            builder.add_box_visual(pose=first_block_pose, half_size=first_block_size, material=sapien.render.RenderMaterial(
                base_color=base_color,
            ),)

            # for the second block (vertical part), we translate y by 4*(box1_half_h)-com_y to align flush with horizontal block
            # note that the cad model tee made here is upside down
            second_block_pose = sapien.Pose([0., 4*(box1_half_h)-com_y, 0.])
            second_block_size = [box1_half_h, (3/4)*(box1_half_w), half_thickness]
            if not target:
                builder.add_box_collision(pose=second_block_pose, half_size=second_block_size,material=tee_material)
                #builder.add_box_collision(pose=second_block_pose, half_size=second_block_size)
            builder.add_box_visual(pose=second_block_pose, half_size=second_block_size, material=sapien.render.RenderMaterial(
                base_color=base_color,
            ),)
            if not target:
                return builder.build(name=name)
            else: return builder.build_kinematic(name=name)

        self.tee = create_tee(name="Tee", target=False)
        self.goal_tee = create_tee(name="goal_Tee", target=True, base_color=np.array([128,128,128,255])/255)

        # adding end-effector end-episode goal position
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_visual(
            radius=0.02,
            half_length=1e-4,
            material=sapien.render.RenderMaterial(base_color=np.array([128, 128, 128, 255]) / 255),
        )
        self.ee_goal_pos = builder.build_kinematic(name="goal_ee")

        # Rest of function is setting up for Custom 2D "Pseudo-Rendering" function below
        res = 64
        uv_half_width = 0.15
        self.uv_half_width = uv_half_width
        self.res = res
        oned_grid = (torch.arange(res, dtype=torch.float32).view(1,res).repeat(res,1) - (res/2))
        self.uv_grid = (torch.cat([oned_grid.unsqueeze(0), (-1*oned_grid.T).unsqueeze(0)], dim=0) + 0.5) / ((res/2)/uv_half_width)
        self.uv_grid = self.uv_grid.to(self.device)
        self.homo_uv = torch.cat([self.uv_grid, torch.ones_like(self.uv_grid[0]).unsqueeze(0)], dim=0)
        
        # tee render
        # tee is made of two different boxes, and then translated by center of mass
        self.center_of_mass = (0,0.0375) #in frame of upside tee with center of horizontal box (add cetner of mass to get to real tee frame)
        box1 = torch.tensor([[-0.1, 0.025], [0.1, 0.025], [-0.1, -0.025], [0.1, -0.025]]) 
        box2 = torch.tensor([[-0.025, 0.175], [0.025, 0.175], [-0.025, 0.025], [0.025, 0.025]])
        box1[:, 1] -= self.center_of_mass[1]
        box2[:, 1] -= self.center_of_mass[1]

        #convert tee boxes to indices
        box1 *= ((res/2)/uv_half_width)
        box1 += (res/2)

        box2 *= ((res/2)/uv_half_width)
        box2 += (res/2)

        box1 = box1.long()
        box2 = box2.long()

        self.tee_render = torch.zeros(res,res)
        # image map has flipped x and y, set values in transpose to undo
        self.tee_render.T[box1[0,0]:box1[1,0], box1[2,1]:box1[0,1]] = 1
        self.tee_render.T[box2[0,0]:box2[1,0], box2[2,1]:box2[0,1]] = 1
        # image map y is flipped of xy plane, flip to unflip
        self.tee_render = self.tee_render.flip(0).to(self.device)
        
        goal_fake_quat = torch.tensor([(torch.tensor([self.goal_z_rot])/2).cos(),0,0,0.0]).unsqueeze(0)
        zrot = self.quat_to_zrot(goal_fake_quat).squeeze(0) # 3x3 rot matrix for goal to world transform
        goal_trans = torch.eye(3)
        goal_trans[:2,:2] = zrot[:2,:2]
        goal_trans[0:2, 2] = self.goal_offset
        self.world_to_goal_trans = torch.linalg.inv(goal_trans).to(self.device) # this is just a 3x3 matrix (2d homogenious transform)


    def _get_obs_extra(self, info: Dict):
        # ee position is super useful for pandastick robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose
        )
        # state based gets info on goal position and t full pose - necessary to learn task
        obs.update(
            goal_pos=self.goal_tee.pose.p,
            obj_pose=self.tee.pose.raw_pose,
        )
        return obs
    
    def reset(self, seed=None, options=None):
        if seed is None and options is None:
            obs,reconfigure_dict = super().reset(seed=seed, options=options)
            self.init_tcp_pose = self.agent.tcp.pose.raw_pose[0]
            return obs
        else:
            return super().reset(seed=seed, options=options)
    
    def seed(self, seed):
        self._set_episode_rng(seed)

    def get_obs(self, info: Optional[Dict] = None):
        obs = super().get_obs(info)
        if info is None:
            base_camera_img = obs['sensor_data']['base_camera']['rgb']
            hand_camera_img = obs['sensor_data']['hand_camera']['rgb']
            robot_tcp_pose = obs['extra']['tcp_pose'] # dim=7, (x,y,z,quat)

            agent_pose = robot_tcp_pose
            image1 = base_camera_img.permute(3,0,1,2) / 255  # (N, 128, 128, 3) -> (N, 3, 128, 128)
            image1 = image1.numpy()
            image2 = hand_camera_img.permute(3,0,1,2) / 255  # (N, 128, 128, 3) -> (N, 3, 128, 128)
            image2 = image2.numpy()

            agent_pos = agent_pose[:, :2]  # N, 2
            agent_pos = agent_pos.numpy()

            new_obs ={
                    'image1': image1,  # N, 3, 128, 128
                    'image2': image2,  # N, 3, 128, 128
                    'agent_pos': agent_pos,  # N, 7
                }
            return new_obs
        else:
            return obs
    
    def step(self, action, is_delta=False):
        assert action.shape == (2,), f"Action shape is {action.shape}, expected (2,)"
        if is_delta:
            obs,reward,terminated,_,info = super(PosPushTEnv, self).step(action)
        else:
            # action: dim=2, target_ee_pos_x, target_ee_pos_y
            current_tcp_pose = self.agent.tcp.pose.raw_pose[0]
            action_delta = action - current_tcp_pose[:2].numpy()
            step_action = np.zeros(6)
            step_action[:2] = action_delta
            step_action[2] = self.init_tcp_pose[2] - current_tcp_pose[2]

            # compensation_quaternion = qmult(self.init_tcp_pose[3:], qinverse(current_tcp_pose[3:]))
            compensation_quaternion = qmult(current_tcp_pose[3:], qinverse(self.init_tcp_pose[3:]))
            delta_axis_angle = compact_axis_angle_from_quaternion(compensation_quaternion)
            step_action[3:] = delta_axis_angle
            obs,reward,terminated,_,info = super(PosPushTEnv, self).step(step_action)

        base_camera_img = obs['sensor_data']['base_camera']['rgb']
        hand_camera_img = obs['sensor_data']['hand_camera']['rgb']
        robot_tcp_pose = obs['extra']['tcp_pose'] # dim=7, (x,y,z,quat)


        agent_pose = robot_tcp_pose
        image1 = base_camera_img.permute(3,0,1,2) / 255  # (N, 128, 128, 3) -> (N, 3, 128, 128)
        image1 = image1.numpy()
        image2 = hand_camera_img.permute(3,0,1,2) / 255  # (N, 128, 128, 3) -> (N, 3, 128, 128)
        image2 = image2.numpy()

        agent_pos = agent_pose[:, :2]  # N, 2
        agent_pos = agent_pos.numpy()

        new_obs ={
            'image1': image1,  # N, 3, 128, 128
            'image2': image2,  # N, 3, 128, 128
            'agent_pos': agent_pos,  # N, 7
        }
        return new_obs,reward,terminated,info
    
    def render(self, mode="rgb_array"):
        if self.render_mode != mode:
            self.render_mode = mode
        img_tensor = super().render()
        img = img_tensor[0].numpy()
        return img

if __name__ == "__main__":
    env = PosPushTEnv()
    obs = env.reset()
    print(obs)
    action = np.array([0.1, 0.1])
    obs, reward, done, info = env.step(action)
    print(obs, reward, done, info)
    env.close()