import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        # additional infos
        self.replace_rate = []
        self.total_sample_num = 1
        self.relabel_num = 0
        self.random_num = 0
        self.nochange_num = 0
        self.not_relabel_unmoved = False # TODO add to arguments
        self.random_unmoved_rate = 1.0

    def sample_her_transitions_old(self, episode_batch, batch_size_in_transitions):
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = int(batch_size_in_transitions)
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        # t_samples = np.random.randint(T, size=batch_size)
        final_t = episode_batch['final_t'][episode_idxs]
        t_samples = np.random.randint(final_t)
        transitions = {}
        for key in episode_batch.keys():
            if key != 'final_t':
                transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (final_t - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        info = {'gripper_arr': transitions['gripper_next']}
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], info), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {}
        for key in episode_batch.keys():
            if key != 'final_t':
                transitions[key] = episode_batch[key][episode_idxs, t_samples].copy() 
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)[0]
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        if self.not_relabel_unmoved:
            sample_size, num_obj, goal_dim = future_ag.shape
            old_ag = transitions['ag'][her_indexes]
            old_goal = transitions['g'][her_indexes]
            if_done = np.linalg.norm(old_ag - old_goal, axis=-1) < 0.05
            if_moved = np.linalg.norm(future_ag - old_ag, axis=-1) > 0.0005
            relabel_musk = np.logical_and((np.logical_not(if_done)), if_moved).reshape(sample_size, num_obj,-1)
            random_musk = np.logical_and((np.logical_not(if_done)), np.logical_not(if_moved)).reshape(sample_size, num_obj,-1)
            nochange_musk = if_done.reshape(sample_size, num_obj,-1)
            # record parameters
            self.total_sample_num += relabel_musk.size
            self.relabel_num += np.sum(relabel_musk)
            self.random_num += np.sum(random_musk)
            self.nochange_num += np.sum(nochange_musk)
            relabel_musk = np.repeat(relabel_musk, 3, axis=-1)
            random_musk = np.repeat(random_musk, 3, axis=-1)
            nochange_musk = np.repeat(nochange_musk, 3, axis=-1)
            if np.random.uniform() < self.random_unmoved_rate:
                random_goal = np.random.uniform([-0.4, -0.15, 0.02], [0.4, 0.15, 0.20], size=(sample_size, num_obj, goal_dim))
                new_goal = future_ag*relabel_musk + old_goal*nochange_musk + random_goal*random_musk
            else:
                new_goal = future_ag*relabel_musk + old_goal*np.logical_or(nochange_musk, random_musk) 
        else:
            new_goal = future_ag
        transitions['g'][her_indexes] = new_goal
        # to get the params to re-compute reward
        info = {'gripper_arr': transitions['gripper_next']}
        # transitions['r'] = np.expand_dims([self.reward_func(transitions['ag_next'][i], transitions['g'][i], None) for i in range(len(transitions['g']))], 1)
        # transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], info), 1)
        transitions['r'] = np.expand_dims(-np.mean((np.linalg.norm(transitions['ag_next'] - transitions['g'], axis=-1) > 0.05),axis=-1), 1) # TODO
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        
        return transitions