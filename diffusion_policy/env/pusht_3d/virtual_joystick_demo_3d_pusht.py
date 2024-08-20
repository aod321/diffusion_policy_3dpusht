#%%
import os
import sys
import numpy as np
import os.path as osp
import gym as gym
from mani_skill.utils.wrappers.record import RecordEpisode
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PATH)
os.environ["CUSTOM_ASSET_DIR"] = os.path.join(PROJECT_PATH,"pusht_3d","custom_assets")
from mani_skill.utils import common, gym_utils
from mani_skill.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
    tile_images,
)
from pusht_3d import PosPushTEnv
from pusht_3d.replay_buffer import ReplayBuffer
import time

# %%

obs_mode = "rgb"
render_mode = "rgb_array"
reward_mode = "dense"
shader = "default"
sim_backend = "auto"
record_dir = "demos"
traj_name = None
save_video = True
env_id = "PosPushT-v1"
replay_buffer_output = "./replay_buffer_output"
max_episode_steps = 100
control_mode = "pd_ee_delta_pose"
max_pos_speed = 0.25


#%%
env = PosPushTEnv(
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        reward_mode=reward_mode,
        shader_dir=shader,
        sim_backend=sim_backend
    )

env.max_episode_steps = max_episode_steps
viewer = env.render_human()
replay_buffer = ReplayBuffer.create_from_path(replay_buffer_output, mode='a')
# %%
import zmq
# ZeroMQ setup
context = zmq.Context()
socket = context.socket(zmq.SUB)
# Connect to the server using the fixed port
server_address = "tcp://localhost:5555"  # Change 'localhost' to the server's IP if not on the same machine
socket.connect(server_address)
# Subscribe to all topics (empty string means subscribe to everything)
socket.setsockopt_string(zmq.SUBSCRIBE, '')
#%%
frequency = 30  # Define the desired frequency in Hz
time_step = 1.0 / frequency  # Calculate the time step duration in seconds
while True:

    episode = list()
    # record in seed order, starting with 0
    seed = replay_buffer.n_episodes
    print(f'starting seed {seed}')
    # set seed for env
    env.seed(seed)

    # reset env and get observations (including info and render for recording)
    obs = env.reset()
    target_tcp_pose = env.agent.tcp.pose.raw_pose
    
    # loop state
    retry = False
    pause = False
    done = False
    while not done:
        start_time = time.time()  # Start time for this loop iteration

        viewer = env.render_human()
        delta_ee_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        action_taken = False
        
        if viewer.window.key_down("r"):  # end episode
            retry=True
        
        if viewer.window.key_down("p"):  # pause
            pause = not pause

        if viewer.window.key_down("q"):  # quit
            env.close()
            exit(0)
            
        # handle control flow
        if retry:
            break

        if pause:
            continue

        # Receive and decode the joystick message
        message = socket.recv_string()
        x, y = map(float, message.split(","))
        # if x == 0 and y == 0:
            # action_taken = False
        # else:
            # action_taken = True

        # if action_taken:
        print(f"Received message: {x}, {y}")
        dpos = np.array([y, x], dtype=np.float32) * (max_pos_speed / frequency)
        target_tcp_pose[0][:2] -= dpos[:2]
        action = target_tcp_pose[0][:2]

        data = {
            'image1': obs['image1'],
            'image2': obs['image2'],
            'agent_pos': obs['agent_pos'],
            'action': action.numpy()
        }
        episode.append(data)
        obs, reward, done, info = env.step(action)
        print(f"reward: {reward}")

        # Control loop frequency
        elapsed_time = time.time() - start_time
        if elapsed_time < time_step:
            time.sleep(time_step - elapsed_time)  # Sleep to maintain frequency


# %%
