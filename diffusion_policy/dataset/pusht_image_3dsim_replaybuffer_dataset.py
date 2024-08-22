#%%
from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class PushtImageDatasetReplayBuffer(BaseImageDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None
                 ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['image1', 'image2',
                             'agent_pos', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        action = self.replay_buffer['action'][:, :2]
        agent_pos = self.replay_buffer['agent_pos'][:, 0, :2]
        
        data = {
            'action': action,
            'agent_pos': agent_pos
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image1'] = get_image_range_normalizer()
        normalizer['image2'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pose = sample['agent_pos'].astype(np.float32)  # (N, 7), (x, y, z, qw, qx, qy, qz)
        image1 = sample['image1'][:,:,0]  # (N, 3, 128, 128)
        image2 = sample['image2'][:,:,0]  # (N, 3, 128, 128)

        agent_pos = agent_pose[:,0, :2]  # N, 2

        action = sample['action'][:, :2] # N, 2

        data = {
            'obs': {
                'image1': image1,  # N, 3, 128, 128
                'image2': image2,  # N, 3, 128, 128
                'agent_pos': agent_pos,  # N, 7
            },
            'action': action.astype(np.float32)  # N, 2
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

# %%
if __name__ == "__main__":
    import os
    zarr_path = "./replay_buffer_output/pusht_3d_yinzi_20240821.zarr.zip"
    dataset = PushtImageDatasetReplayBuffer(zarr_path, horizon=16)
    print(dataset[0])
