import pickle
import torch
import os
import time
from datetime import datetime
import numpy as np
import hydra
import wandb
from tqdm import tqdm
from attrdict import AttrDict
from mpi4py import MPI

import sparse_rl
from sparse_rl.replay_buffer import replay_buffer
from sparse_rl.utils import ArrayNormalizer, sync_networks, sync_grads
from sparse_rl.relabel import her_sampler

def linear_sched(x0, x1, y0, y1, x):
	m = (y1 - y0) / (x1 - x0)
	return m * (x - x1) + y1


"""
ddpg with HER
"""


class ddpg_agent: 
	def __init__(self, args, env, env_params, ckpt_data=None):
		self.args = args
		if self.args.cuda and not torch.cuda.is_available():
			print('[WARN] cuda not available.')
			self.args.cuda = False
		self.env = env
		self.env_params = env_params
		# MPI 
		self.comm = MPI.COMM_WORLD
		# her sampler
		self.her_module = her_sampler(
			self.args.replay_strategy, self.args.replay_k, env_params['compute_reward'])
		if ckpt_data is None:
			self.current_epoch = 0
			self.tot_samples = 0
			self.best_success_rate = 0

			# create the network
			self.actor_network = hydra.utils.instantiate(
				args.actor, env_params)
			self.critic_network = hydra.utils.instantiate(
				args.critic, env_params)
			# sync network
			sync_networks(self.actor_network)
			sync_networks(self.critic_network)
			# build up the target network
			self.actor_target_network = hydra.utils.instantiate(
				args.actor, env_params)
			self.critic_target_network = hydra.utils.instantiate(
				args.critic, env_params)
			# load the weights into the target networks
			self.actor_target_network.load_state_dict(
				self.actor_network.state_dict())
			self.critic_target_network.load_state_dict(
				self.critic_network.state_dict())
			# if use gpu
			if self.args.cuda:
				self.actor_network.cuda()
				self.critic_network.cuda()
				self.actor_target_network.cuda().eval()
				self.critic_target_network.cuda().eval()
			self.actor_optim = torch.optim.Adam(
				self.actor_network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
			self.critic_optim = torch.optim.Adam(
				self.critic_network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
			self.actor_sched, self.critic_sched = None, None
			if args.warmup_actor > 0:
				self.actor_sched = torch.optim.lr_scheduler.LambdaLR(
					self.actor_optim, lambda t: min((t+1) / args.warmup_actor, 1))
			if args.warmup_critic > 0:
				self.critic_sched = torch.optim.lr_scheduler.LambdaLR(
					self.critic_optim, lambda t: min((t+1) / args.warmup_critic, 1))
			# create the replay buffer
			self.buffer = replay_buffer(
				self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
			# create the normalizer
			self.x_norm = ArrayNormalizer(
				self.env_params, default_clip_range=self.args.clip_range)
			# create the dict for store the model
		else:
			for k, v in ckpt_data.items():
				setattr(self, k, v)
			self.buffer = replay_buffer(
				self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
		self.init_data = None
		if args.init_trajs:
			print(f"loading initial trajectories from {args.init_trajs}.")
			data = np.load(args.init_trajs)
			self.init_data = [data["grip"], data["obj"],
												data["ag"], data["g"], data["action"]]
		if MPI.COMM_WORLD.Get_rank() == 0:
			if self.args.wandb:
				self.model_dir = os.path.join(wandb.run.dir, "models")
			else:
				self.model_dir = os.path.join('saved_models/', "models")
			if not os.path.exists(self.model_dir):
				os.makedirs(self.model_dir, exist_ok=True)
		self.curri_params = {}
		for k, v in self.args.curri.items():
			self.curri_params[k] = v['start']
		self.useless_steps = 0

	def learn(self):
		"""
		train the network

		"""
		if self.current_epoch == 0 and self.init_data is not None:
			self.buffer.store_episode(self.init_data)
			self._update_normalizer(self.init_data)
			for _ in range(self.args.n_init_steps):
				for _ in range(self.args.n_batches):
					metrics = self._update_network()
				# soft update
				self._soft_update_target_network(
					self.actor_target_network, self.actor_network)
				self._soft_update_target_network(
					self.critic_target_network, self.critic_network)
		for epoch in tqdm(range(self.current_epoch, self.args.n_epochs)):
			if self.args.exp_schedule:
				start_slope, end_slope, final_ratio = self.args.exp_schedule
				assert final_ratio < 1 and start_slope < end_slope
				exp_ratio = np.clip(linear_sched(
					start_slope, end_slope, 1, final_ratio, self.current_epoch), final_ratio, 1)
				noise_eps = self.args.noise_eps * exp_ratio
				random_eps = self.args.random_eps * exp_ratio
			else:
				noise_eps, random_eps = self.args.noise_eps, self.args.random_eps
			start = time.time()
			for _ in range(self.args.n_cycles):
				# shape (rollout_id, time, goal_id, 3)
				mb_grip, mb_obj, mb_ag, mb_g, mb_actions = [], [], [], [], []
				collected_rollout = 0
				while collected_rollout < self.args.num_rollouts * self.args.num_workers:
					# reset the rollouts (time,env,data)
					ep_grip, ep_obj, ep_ag, ep_g, ep_actions = [], [], [], [], []
					# reset the environment
					observation = self.env.reset(self.curri_params)
					# start to collect samples
					for t in range(self.env_params['max_timesteps']):
						grip, obj = observation['gripper_arr'], observation['object_arr']
						g = observation['desired_goal_arr']
						with torch.no_grad():
							inputs = self._preproc_inputs(grip, obj, g)
							pi = self.actor_network(*inputs)
							action = self._select_actions(
								pi, noise_eps, random_eps)
						# feed the actions into the environment
						observation_new, _, _, info = self.env.step(action)
						# append rollouts
						ep_grip.append(grip)
						ep_obj.append(obj.copy())
						ep_ag.append(observation['achieved_goal_arr'].copy())
						ep_g.append(g.copy())
						ep_actions.append(action.copy())
						# re-assign the observation
						observation = observation_new
					ep_grip.append(observation['gripper_arr'].copy())
					ep_obj.append(observation['object_arr'].copy())
					ep_ag.append(observation['achieved_goal_arr'].copy())
					dropout = np.array([d['dropout'] for d in info], dtype=bool)
					self.useless_steps += (sum(dropout) * self.env_params['max_timesteps'])
					mb_grip.append(np.stack(ep_grip, 1)[~dropout])
					mb_obj.append(np.stack(ep_obj, 1)[~dropout])
					mb_ag.append(np.stack(ep_ag, 1)[~dropout])
					mb_g.append(np.stack(ep_g, 1)[~dropout])
					mb_actions.append(np.stack(ep_actions, 1)[~dropout])
					collected_rollout += sum(~dropout)
				# convert them into arrays
				mb_grip = np.concatenate(mb_grip, 0)
				mb_obj = np.concatenate(mb_obj, 0)
				mb_ag = np.concatenate(mb_ag, 0)
				mb_g = np.concatenate(mb_g, 0)
				mb_actions = np.concatenate(mb_actions, 0)
				self.tot_samples += mb_actions.shape[0] * mb_actions.shape[1]
				# store the episodes
				self.buffer.store_episode(
					[mb_grip, mb_obj, mb_ag, mb_g, mb_actions])
				self._update_normalizer(
					[mb_grip, mb_obj, mb_ag, mb_g, mb_actions])
				if self.tot_samples > self.args.min_samples:
					for _ in range(self.args.n_batches):
						# train the network
						metrics = self._update_network()
					# soft update
					self._soft_update_target_network(
						self.actor_target_network, self.actor_network)
					self._soft_update_target_network(
						self.critic_target_network, self.critic_network)
					if self.args.wandb and MPI.COMM_WORLD.Get_rank() == 0:
						wandb.log(metrics, step=self.tot_samples)
			# start to do the evaluation
			eval_return = self._eval_agent(
				render=(self.current_epoch % self.args.render_interval) == 0 and self.current_epoch > 0)
			print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(
				datetime.now(), self.current_epoch, eval_return.succ))
			# start curriculum
			for k, v in self.args.curri.items():
				if eval_return.rew_final > v['bar'] and self.curri_params[k] < v['end']:
					self.curri_params[k] += v['step']
			print('Curri_params:', self.curri_params)
			print(f'Epoch time: {time.time() - start}')
			if self.args.wandb and MPI.COMM_WORLD.Get_rank() == 0:
				wandb.log({
					**self.curri_params, 
					'eval/success_rate': eval_return.succ,
					'eval/rew_mean': eval_return.rew_mean,
					'eval/rew_final': eval_return.rew_final,
					'epoch': self.current_epoch,
					'exploration/random_eps': random_eps,
					'exploration/noise_eps': noise_eps,
					'exploration/useless_rate': self.useless_steps/self.tot_samples
				}, step=self.tot_samples)
			self.current_epoch = epoch
			if MPI.COMM_WORLD.Get_rank() == 0:
				save_data = [self.x_norm, self.actor_network, self.critic_network]
				torch.save(save_data, os.path.join(self.model_dir, "latest.pt"))
				if eval_return.succ >= self.best_success_rate or self.current_epoch == 0:
					self.best_success_rate = eval_return.succ
					torch.save(save_data, os.path.join(self.model_dir, "best.pt"))
				self.save_checkpoint()

	def save_checkpoint(self):
		data = {
			"actor_network": self.actor_network,
			"critic_network": self.critic_network,
			"actor_target_network": self.actor_target_network,
			"critic_target_network": self.critic_target_network,
			"current_epoch": self.current_epoch,
			"tot_samples": self.tot_samples,
			"best_success_rate": self.best_success_rate,
			"actor_optim": self.actor_optim,
			"critic_optim": self.critic_optim,
			"actor_sched": self.actor_sched,
			"critic_sched": self.critic_sched,
			"x_norm": self.x_norm,
			"wandb_run_id": wandb.run.id if self.args.wandb else None,
			"wandb_run_name": wandb.run.name if self.args.wandb else None,
			# "buffer": self.buffer,
		}
		with open(os.path.join(self.model_dir, "checkpoint.pkl"), "wb") as f:
			pickle.dump(data, f)

	# pre_process the inputs
	def _preproc_inputs(self, grip, obj, g):
		# concatenate the stuffs
		outs = self.x_norm.normalize(grip, obj, g)
		outs = [torch.tensor(x, dtype=torch.float32) for x in outs]
		if self.args.cuda:
			outs = [x.cuda() for x in outs]
		return outs

	# this function will choose action for the agent and do the exploration
	def _select_actions(self, pi, noise_eps, random_eps):
		action = pi.cpu().numpy()
		# add the gaussian
		action += noise_eps * \
			self.env_params['action_max'] * np.random.randn(*action.shape)
		action = np.clip(
			action, -self.env_params['action_max'], self.env_params['action_max'])
		# random actions...
		random_actions = np.random.uniform(
			low=-self.env_params['action_max'],
			high=self.env_params['action_max'],
			size=action.shape,
		)
		# choose if use the random actions
		action += np.random.binomial(1, random_eps,
																 (action.shape[0], 1)) * (random_actions - action)
		return action

	# update the normalizer
	def _update_normalizer(self, episode_batch):
		mb_grip, mb_obj, mb_ag, mb_g, mb_actions = episode_batch
		# get the number of normalization transitions
		num_transitions = mb_actions.shape[1]
		# create the new buffer to store them
		final_t = np.zeros(mb_grip.shape[0], dtype=np.int64)
		final_t[:] = num_transitions
		buffer_temp = {
			'obj': mb_obj,
			'ag': mb_ag,
			'g': mb_g,
			'actions': mb_actions,
			'obs_next': mb_obj[:, 1:, :],
			'ag_next': mb_ag[:, 1:, :],
			'gripper': mb_grip,
			'gripper_next': mb_grip[:, 1:, :],
			'final_t': final_t,
		}
		transitions = self.her_module.sample_her_transitions(
			buffer_temp, num_transitions)
		grip, obj, g = transitions['gripper'], transitions['obj'], transitions['g']
		# pre process the obs and g
		transitions['gripper'], transitions['obj'], transitions['g'] = self._preproc_og(
			grip, obj, g)
		# update
		self.x_norm.update(transitions['gripper'],
											 transitions['obj'], transitions['g'])
		# recompute the stats
		self.x_norm.recompute_stats()

	def _preproc_og(self, grip, obj, g):
		grip = np.clip(grip, -self.args.clip_obs, self.args.clip_obs)
		obj = np.clip(obj, -self.args.clip_obs, self.args.clip_obs)
		g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
		return grip, obj, g

	# soft update
	def _soft_update_target_network(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(
				(1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

	# update the network
	def _update_network(self):
		# sample the episodes
		transitions = self.buffer.sample(self.args.batch_size)
		# pre-process the observation and goal
		grip, grip_next, obj, obj_next, g = transitions['gripper'], transitions[
			'gripper_next'], transitions['obj'], transitions['obj_next'], transitions['g']
		transitions['gripper'], transitions['obs'], transitions['g'] = self._preproc_og(
			grip, obj, g)
		transitions['gripper_next'], transitions['obs_next'], transitions['g_next'] = self._preproc_og(
			grip_next, obj_next, g)
		# start to do the update
		inputs_norm = self.x_norm.normalize(
			transitions['gripper'], transitions['obj'], transitions['g'])
		inputs_next_norm = self.x_norm.normalize(
			transitions['gripper_next'], transitions['obj_next'], transitions['g_next'])
		# transfer them into the tensor
		inputs_norm_tensor = [torch.tensor(
			x, dtype=torch.float32) for x in inputs_norm]
		inputs_next_norm_tensor = [torch.tensor(
			x, dtype=torch.float32) for x in inputs_next_norm]
		actions_tensor = torch.tensor(
			transitions['actions'], dtype=torch.float32)
		r_tensor = self.args.r_scale * \
			torch.tensor(transitions['r'], dtype=torch.float32)
		if self.args.cuda:
			inputs_norm_tensor = [x.cuda() for x in inputs_norm_tensor]
			inputs_next_norm_tensor = [x.cuda()
																 for x in inputs_next_norm_tensor]
			actions_tensor = actions_tensor.cuda()
			r_tensor = r_tensor.cuda()
		# calculate the target Q value function
		with torch.no_grad():
			# do the normalization
			# concatenate the stuffs
			actions_next = self.actor_target_network(*inputs_next_norm_tensor)
			q_next_value = self.critic_target_network(
				*inputs_next_norm_tensor, actions_next)
			q_next_value = q_next_value.detach()
			target_q_value = r_tensor + self.args.gamma * q_next_value
			target_q_value = target_q_value.detach()
			# clip the q value
			clip_return = 1 / (1 - self.args.gamma)
			target_q_value = torch.clamp(target_q_value, -clip_return, 0)
		# the q loss
		real_q_value = self.critic_network(*inputs_norm_tensor, actions_tensor)
		critic_loss = (target_q_value - real_q_value).pow(2).mean()
		# the actor loss
		actions_real = self.actor_network(*inputs_norm_tensor)
		self.critic_network.eval()
		actor_loss = - \
			self.critic_network(*inputs_norm_tensor, actions_real).mean()
		self.critic_network.train()
		actor_loss += self.args.action_l2 * \
			(actions_real / self.env_params['action_max']).pow(2).mean()
		# start to update the network
		self.actor_optim.zero_grad()
		actor_loss.backward()
		sync_grads(self.actor_network)
		self.actor_optim.step()
		# update the critic_network
		self.critic_optim.zero_grad()
		critic_loss.backward()
		sync_grads(self.critic_network)
		self.critic_optim.step()
		metrics = {
			'loss/actor': actor_loss.detach().cpu().item(),
			'loss/critic': critic_loss.detach().cpu().item(),
		}
		if self.actor_sched is not None:
			self.actor_sched.step()
			metrics['lr/actor'] = self.actor_sched.get_last_lr()[0]
		if self.critic_sched is not None:
			self.critic_sched.step()
			metrics['lr/critic'] = self.critic_sched.get_last_lr()[0]
		return metrics

	# do the evaluation
	def _eval_agent(self, render=False):
		self.actor_network.eval()
		results, returns, final_rew = [], [], []
		observation = self.env.reset()
		if MPI.COMM_WORLD.Get_rank() == 0:
			video = np.array([])
		for _ in range(self.args.n_test_eps):
			ret = np.zeros(self.args.num_workers)
			for t in range(self.env_params['max_timesteps']):
				grip, obj = observation['gripper_arr'], observation['object_arr']
				g = observation['desired_goal_arr']
				with torch.no_grad():
					input_tensors = self._preproc_inputs(grip, obj, g)
					pi = self.actor_network(*input_tensors)
					actions = pi.detach().cpu().numpy()
				observation_new, rew, done, info = self.env.step(actions)
				ret += rew
				observation = observation_new
				if render and len(results) <= 32 and MPI.COMM_WORLD.Get_rank() == 0:  # TODO make it not hard code
					frame = np.array(self.env.render(mode='rgb_array'))
					frame = np.moveaxis(frame, -1, 1)
					if video.shape[0] == 0:
						# (time, num_env, 4, r, g, b)
						video = np.array([frame])
					else:
						video = np.concatenate((video, [frame]), axis=0)
			for idx in range(self.args.num_workers):
				results.append(info[idx]['is_success'])
				returns.append(ret[idx])
				final_rew.append(rew[idx])
		if render and self.args.wandb and MPI.COMM_WORLD.Get_rank() == 0:
			video = np.moveaxis(video, 0, 1)  # (num_env, time, 4, r, g, b)
			video = np.concatenate(video, axis=0)  # (num_env*time, 4, r, g, b)
			wandb.log({"video": wandb.Video(
				np.array(video), fps=30, format="mp4")})
			del video
		success_rate = np.mean(results)
		ret = np.mean(returns)
		rew_final=np.mean(final_rew)
		global_success_rate = MPI.COMM_WORLD.allreduce(success_rate, op=MPI.SUM)/MPI.COMM_WORLD.Get_size()
		global_ret = MPI.COMM_WORLD.allreduce(ret, op=MPI.SUM)/MPI.COMM_WORLD.Get_size()
		global_rew_final = MPI.COMM_WORLD.allreduce(rew_final, op=MPI.SUM)/MPI.COMM_WORLD.Get_size()
		self.actor_network.train()
		return AttrDict(
			succ=global_success_rate,
			rew_mean=global_ret,
			rew_final=global_rew_final
		)