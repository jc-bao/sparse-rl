import isaacgym

import torch
import os
import pickle
import hydra
import wandb
from attrdict import AttrDict
from omegaconf import OmegaConf
import gym 
from sparse_rl.agent import ddpg_agent
from sparse_rl.utils import SubprocVecEnv, get_env_params

@hydra.main(config_name='main', config_path='config')
def launch(cfg = None):
	# 1. make env
	import os, sys
	sys.path.append('/home/reed/rl/srl/envs')
	import franka_cube
	env = IsaacWrapper(gym.make('FrankaPNP-v0', num_envs=cfg.num_workers, num_cameras=0, headless=False, auto_reset=False))
	p = env.env_params
	env_params = {
		'gripper': p.shared_dim,
		'goal': p.goal_dim,
		'action': p.action_dim,
		'action_max': 1.0,
		'object': p.seperate_dim,
		'n_objects': p.num_goals,
		'max_timesteps': env.cfg.max_steps, 
		'compute_reward': p.compute_reward
	}
	# sys.path.append("/home/reed/rl/panda-isaac")
	# from panda_isaac.panda_push import PandaPushEnv
	# from panda_isaac.base_config import BaseConfig
	# class PushConfig(BaseConfig):
	#     class env(BaseConfig.env):
	#         seed = 42
	#         # num_envs = 1024
	#         num_envs = cfg.num_workers 
	#         # num_observations = 3 * 224 * 224 + 12
	#         num_observations = (3 + 15) * 2
	#         num_actions = 4
	#         max_episode_length = 100
	#     class obs(BaseConfig.obs):
	#         type = "state"
	#         state_history_length = 2
	#     class control(BaseConfig.control):
	#         decimal = 6
	#         controller = "ik"
	#     class reward(BaseConfig.reward):
	#         type = "sparse"
	# env = IsaacWrapper(PandaPushEnv(PushConfig, headless=False))
	# env_params = {
	#     'gripper': 12,
	#     'goal': 3,
	#     'action': 4,
	#     'action_max': 1.0,
	#     'object': 3,
	#     'n_objects': 1,
	#     'max_timesteps': 50, 
	#     'compute_reward': None
	# }

	# env_params = get_env_params(cfg.env_name, cfg.env_kwargs)
	# env = SubprocVecEnv([make_env for i in range(cfg.num_workers)])

	# 2. start wandb
	ckpt_data = None
	if cfg.wandb:
		# mode1: init 
		if cfg.wid is None:
			wandb.init(project=cfg.project, name=cfg.name, dir=hydra.utils.get_original_cwd())
		else:
			print('[DEBUG] load data from remote')
			file = wandb.restore('models/checkpoint.pkl', f'jc-bao/{cfg.project}/{cfg.wid}')
			with open(file.name, "rb") as f:
				ckpt_data = pickle.load(f)
				print('[DEBUG] load done')
			if cfg.name == ckpt_data['wandb_run_name']:
				# mode2: resume and continue
				print('[DEBUG] resume old run')
				wandb.init(project=cfg.project, id=cfg.wid, resume="allow", dir=hydra.utils.get_original_cwd())
			else:
				# mode3: resume and new
				print('[DEBUG] start new run')
				wandb.init(project=cfg.project, name=cfg.name, resume="allow", dir=hydra.utils.get_original_cwd())
		wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
		wandb.save(".hydra/*")
		cfg = AttrDict(wandb.config)
	elif not cfg.wid is None:
		print('[WARN] wid was set but wandb is not enabled')
	else:
		cfg = AttrDict(cfg)
	
	# 3. run
	ddpg_trainer = ddpg_agent(cfg, env, env_params, ckpt_data=ckpt_data)
	ddpg_trainer.learn()

class IsaacWrapper2:
	def __init__(self, env):
		self.env = env

	def step(self, action):
		action = torch.tensor(action, device='cuda:0')
		obs_old, reward, done, info = self.env.step(action)
		obs = {
			'gripper_arr': obs_old[:,21:33].cpu().numpy(), 
			'object_arr': obs_old[:,18:21].unsqueeze(1).cpu().numpy(),
			'desired_goal_arr': obs_old[:,33:36].unsqueeze(1).cpu().numpy(),
			'achieved_goal_arr': obs_old[:,18:21].unsqueeze(1).cpu().numpy(),
		}
		return obs, reward-1, done, info

	def reset(self):
		self.env.reset_idx(torch.arange(self.env.num_envs,device='cuda:0'))
		return self.step(torch.zeros(self.env.num_envs, 4))[0]

	def render(self):
		pass
class IsaacWrapper:
	def __init__(self, env):
		self.env = env
		self.env_params = self.env.env_params()
		self.cfg = self.env.cfg
		self.num_envs = self.env.cfg.num_envs

	def step(self, action):
		action = torch.tensor(action, device='cuda:0')
		obs_old, reward, done, info = self.env.step(action)
		obs_dict = self.env.obs_parser(obs_old)
		obs = {
			'gripper_arr': obs_dict.shared.cpu().numpy(), 
			'object_arr': obs_dict.seperate.unsqueeze(1).cpu().numpy(),
			'desired_goal_arr': obs_dict.g.unsqueeze(1).cpu().numpy(),
			'achieved_goal_arr': obs_dict.ag.unsqueeze(1).cpu().numpy(),
		}
		return obs, reward, done, info
	
	def reset(self):
		obs_old = self.env.reset()
		obs_dict = self.env.obs_parser(obs_old)
		obs = {
			'gripper_arr': obs_dict.shared.cpu().numpy(), 
			'object_arr': obs_dict.seperate.unsqueeze(1).cpu().numpy(),
			'desired_goal_arr': obs_dict.g.unsqueeze(1).cpu().numpy(),
			'achieved_goal_arr': obs_dict.ag.unsqueeze(1).cpu().numpy(),
		}
		return obs

	def render(self):
		pass

if __name__ == '__main__':
	launch()