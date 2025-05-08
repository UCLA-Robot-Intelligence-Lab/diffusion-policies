import os
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict

from shared.env.base_env import BaseLowdimRunner
from shared.utils.dict_util import dict_apply

class PushBlockLowdimRunner(BaseLowdimRunner):
    def __init__(
        self,
        output_dir,
        n_train=6,
        n_train_vis=2,
        train_start_seed=0,
        n_test=50,
        n_test_vis=4,
        test_start_seed=100000,
        max_steps=300,
        n_obs_steps=2,
        n_action_steps=8,
        n_latency_steps=0,
        fps=10,
        past_action=False,
        n_envs=None,
    ):
        """
        Args:
            output_dir: Output directory for saving results
            n_train: Number of training episodes to run
            n_train_vis: Number of training episodes to visualize
            train_start_seed: Starting seed for training episodes
            n_test: Number of test episodes to run
            n_test_vis: Number of test episodes to visualize
            test_start_seed: Starting seed for test episodes
            max_steps: Maximum number of steps per episode
            n_obs_steps: Number of observation steps
            n_action_steps: Number of action steps
            n_latency_steps: Number of latency steps between observation and action
            fps: Frames per second for visualization
            past_action: Whether to include past actions in observation
            n_envs: Number of parallel environments
        """
        super().__init__()
        
        self.output_dir = output_dir
        self.n_train = n_train
        self.n_train_vis = n_train_vis
        self.train_start_seed = train_start_seed
        self.n_test = n_test 
        self.n_test_vis = n_test_vis
        self.test_start_seed = test_start_seed
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.fps = fps
        self.past_action = past_action
        self.n_envs = n_envs

    def run(self, policy):
        """
        Run policy on the environment
        
        Args:
            policy: Policy to run
            
        Returns:
            Dictionary with run statistics
        """
        # Temporarily set policy to eval mode
        training = policy.training
        policy.eval()
        
        results = self._run_eval(
            policy=policy,
            n_episodes=self.n_train,
            n_episodes_vis=self.n_train_vis,
            start_seed=self.train_start_seed,
            vis_prefix='train'
        )
        
        test_results = self._run_eval(
            policy=policy,
            n_episodes=self.n_test,
            n_episodes_vis=self.n_test_vis,
            start_seed=self.test_start_seed,
            vis_prefix='test'
        )
        
        # Add test results to the main results
        for k, v in test_results.items():
            results[f'test_{k}'] = v
            
        # Reset policy to its original mode
        policy.train(training)
        
        return results

    def _run_eval(
        self,
        policy,
        n_episodes,
        n_episodes_vis,
        start_seed,
        vis_prefix
    ):
        """
        Run evaluation with the given policy
        
        Args:
            policy: Policy to evaluate
            n_episodes: Number of episodes to run
            n_episodes_vis: Number of episodes to visualize
            start_seed: Starting seed for episodes
            vis_prefix: Prefix for visualization files
            
        Returns:
            Dictionary with evaluation statistics
        """
        # Initialize metrics and visualization
        all_metrics = defaultdict(list)
        vis_save_dir = os.path.join(self.output_dir, 'video')
        os.makedirs(vis_save_dir, exist_ok=True)
        
        # Helper function for executing the policy
        def run_episode(seed, visualize=False):
            from shared.env.pushblock.pushblock_env import PushBlockEnv
            
            env = PushBlockEnv(seed=seed)
            obs = env.reset()
            
            # Observation history buffer
            obs_deque = []
            
            # Performance metrics
            episode_metrics = defaultdict(list)
            episode_max_coverage = 0
            
            # For visualization
            if visualize:
                frames = []
            
            # Initialize history
            for _ in range(self.n_obs_steps):
                obs_dict = {
                    # Make sure the robot state has the correct shape (6 dimensions)
                    'robot_eef_pose': np.array(obs, dtype=np.float32).reshape(-1)
                }
                obs_deque.append(obs_dict)
            
            # Main simulation loop
            for step_idx in range(self.max_steps):
                # Create batch observation (add batch dimension)
                obs_seq = {
                    'robot_eef_pose': np.stack([o['robot_eef_pose'] for o in obs_deque], axis=0)
                }
                batch_obs = {}
                for k, v in obs_seq.items():
                    # Add batch dimension
                    batch_obs[k] = v[np.newaxis, ...]
                
                # Debug print observation shape
                print(f"Env Runner - Observation shape: {batch_obs['robot_eef_pose'].shape}")
                
                # Get action from policy
                with torch.no_grad():
                    batch_obs_torch = dict_apply(batch_obs, torch.from_numpy)
                    
                    # Ensure shape matches training expectations [B, T, D] where D=6
                    if batch_obs_torch['robot_eef_pose'].dim() > 3:
                        # Reshape if needed
                        batch_obs_torch['robot_eef_pose'] = batch_obs_torch['robot_eef_pose'].view(
                            batch_obs_torch['robot_eef_pose'].shape[0], -1, 6
                        )
                    
                    action_dict = policy.predict_action({'obs': batch_obs_torch})
                    action = action_dict['action'][0, 0].cpu().numpy()
                
                # Step the environment
                obs, reward, done, info = env.step(action)
                
                # Update observation history
                obs_dict = {
                    'robot_eef_pose': np.array(obs, dtype=np.float32).reshape(-1)
                }
                obs_deque.append(obs_dict)
                if len(obs_deque) > self.n_obs_steps:
                    obs_deque.pop(0)
                
                # Update metrics
                for k, v in info.items():
                    episode_metrics[k].append(v)
                
                # Track coverage
                if 'coverage' in info and info['coverage'] is not None:
                    episode_max_coverage = max(episode_max_coverage, info['coverage'])
                
                # For visualization
                if visualize:
                    frame = env.render(mode='rgb_array')
                    frames.append(frame)
                
                if done:
                    break
            
            # Compute summary metrics
            metrics = dict()
            metrics['max_coverage'] = episode_max_coverage
            
            for k, v in episode_metrics.items():
                if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    metrics[f'{k}_mean'] = np.mean(v)
                    
            # Create and save video if visualize is enabled
            if visualize and len(frames) > 0:
                video_path = os.path.join(vis_save_dir, f"{vis_prefix}_seed{seed}.mp4")
                height, width, _ = frames[0].shape
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    self.fps,
                    (width, height),
                )
                
                for frame in frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
                
                # Skip wandb logging during testing
                if 'WANDB_MODE' not in os.environ or os.environ['WANDB_MODE'] != 'disabled':
                    try:
                        wandb.log({
                            f"{vis_prefix}_video_seed{seed}": wandb.Video(
                                video_path, fps=self.fps, format="mp4"
                            )
                        })
                    except Exception as e:
                        print(f"Wandb logging failed: {e}")
            
            return metrics
            
        # Run episodes
        for episode_idx in tqdm(range(n_episodes), desc=f'Eval {vis_prefix}'):
            seed = start_seed + episode_idx
            visualize = episode_idx < n_episodes_vis
            
            metrics = run_episode(seed, visualize)
            
            # Collect metrics
            for k, v in metrics.items():
                all_metrics[k].append(v)
                
        # Average metrics
        results = {}
        for k, v in all_metrics.items():
            if len(v) > 0:
                results[f'mean_{k}'] = np.mean(v)
                results[f'std_{k}'] = np.std(v)
                results[f'min_{k}'] = np.min(v)
                results[f'max_{k}'] = np.max(v)
        
        # For convenience, populate a human-readable score
        if 'mean_max_coverage' in results:
            results['mean_score'] = results['mean_max_coverage']
            
        return results

def test():
    from shared.models.common.normalizer import LinearNormalizer
    from diffusion_policy.unet.unet_lowdim_policy import DiffusionUnetLowdimPolicy
    from shared.models.unet.conditional_unet1d import ConditionalUnet1D
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    
    # Disable wandb for testing
    os.environ['WANDB_MODE'] = 'disabled'
    
    # Create a simple random policy for testing
    class RandomPolicy:
        def __init__(self):
            self.training = False
            
        def eval(self):
            self.training = False
            
        def train(self, mode=True):
            self.training = mode
            
        def predict_action(self, obs_dict):
            # Get batch size from observation
            batch_size = obs_dict['obs']['robot_eef_pose'].shape[0]
            # Return random action
            return {
                'action': torch.randn(batch_size, 1, 7)
            }
    
    # Initialize runner
    runner = PushBlockLowdimRunner(
        output_dir='./output',
        n_train=1,
        n_train_vis=1,
        n_test=1,
        n_test_vis=1,
        max_steps=10
    )
    
    # Create and run policy
    policy = RandomPolicy()
    results = runner.run(policy)
    
    print("Results:", results)

if __name__ == "__main__":
    test() 