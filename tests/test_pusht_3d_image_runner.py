#%%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from diffusion_policy.env_runner.pusht_3d_image_runner import Pusht3DImageRunner

def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = './diffusion_policy/config/task/pusht_3d_image.yaml'
    cfg = OmegaConf.load(cfg_path)
    cfg['n_obs_steps'] = 1
    cfg['n_action_steps'] = 1
    cfg['past_action_visible'] = False
    runner_cfg = cfg['env_runner']
    runner_cfg['n_train'] = 1
    runner_cfg['n_test'] = 1
    del runner_cfg['_target_']
    runner = Pusht3DImageRunner(
        **runner_cfg, 
        output_dir='/tmp/test_push_3d_image_runner')

    # import pdb; pdb.set_trace()

    self = runner
    env = self.env
    # env.seed(seeds=self.env_seeds)
    obs = env.reset()
    for i in range(10):
        _ = env.step(env.action_space.sample())

    imgs = env.render()
    print(imgs)
    import mediapy as wv
    wv.show_video(wv.read_video(imgs))

if __name__ == '__main__':
    test()

# %%
