import time

from garage.sampler import utils
from garage.misc import special
from garage.logger import logger, tabular
from garage.sampler import parallel_sampler
from garage.sampler.stateful_pool import singleton_pool
from garage.tf.misc import tensor_utils
from garage.tf.samplers.batch_sampler import BatchSampler
import numpy as np

from embed2learn.samplers.utils import sliding_window

# TODO: improvements to garage so that you don't need to rwrite a whole sampler
# to change the rollout process

def rollout(env,
            agent,
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            always_return_paths=False):

    observations = []
    tasks = []
    tasks_gt = []
    latents = []
    latent_infos = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []

    # Resets
    o = env.reset()
    agent.reset()

    # Sample embedding network
    # NOTE: it is important to do this _once per rollout_, not once per
    # timestep, since we need correlated noise.
    t = env.active_task_one_hot
    task_gt = env.active_task_one_hot_gt
    z, latent_info = agent.get_latent(t)

    if animated:
        env.render()

    path_length = 0
    while path_length < max_path_length:
        #a, agent_info = agent.get_action(np.concatenate((t, o)))
        a, agent_info = agent.get_action_from_latent(z, o)
        # latent_info = agent_info["latent_info"]
        next_o, r, d, env_info = env.step(a)
        observations.append(agent.observation_space.flatten(o))
        tasks.append(t)
        tasks_gt.append(task_gt)
        # z = latent_info["mean"]
        latents.append(agent.latent_space.flatten(z))
        latent_infos.append(latent_info)
        rewards.append(r)
        actions.append(agent.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        tasks=tensor_utils.stack_tensor_list(tasks),
        tasks_gt=tensor_utils.stack_tensor_list(tasks_gt),
        latents=tensor_utils.stack_tensor_list(latents),
        latent_infos=tensor_utils.stack_tensor_dict_list(latent_infos),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )

# Partial parallel_sampler API to modify the rollout function
def _worker_collect_one_path(g, max_path_length, scope=None):
    g = parallel_sampler._get_scoped_g(g, scope)
    path = rollout(g.env, g.policy, max_path_length)
    return path, len(path["rewards"])

def sample_paths(policy_params,
                 max_samples,
                 max_path_length=np.inf,
                 scope=None):
    """
    :param policy_params: parameters for the policy. This will be updated on
     each worker process
    :param max_samples: desired maximum number of samples to be collected. The
     actual number of collected samples might be greater since all trajectories
     will be rolled out either until termination or until max_path_length is
     reached
    :param max_path_length: horizon / maximum length of a single trajectory
    :return: a list of collected paths
    """
    singleton_pool.run_each(
        parallel_sampler._worker_set_policy_params,
        [(policy_params, scope)] * singleton_pool.n_parallel)
    return singleton_pool.run_collect(
        _worker_collect_one_path,
        threshold=max_samples,
        args=(max_path_length, scope),
        show_prog_bar=True)

#TODO: can this use VectorizedSampler?
class TaskEmbeddingSampler(BatchSampler):

    def obtain_samples(self, itr, batch_size=None, whole_paths=True):
        if not batch_size:
            batch_size = self.algo.max_path_length * self.n_envs

        cur_policy_params = self.algo.policy.get_param_values()
        paths = sample_paths(
            policy_params=cur_policy_params,
            max_samples=batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        if whole_paths:
            return paths
        else:
            paths_truncated = truncate_paths(paths, batch_size)
            return paths_truncated

    #TODO: vectorize
    def process_samples(self, itr, paths):
        baselines = []
        returns = []

        max_path_length = self.algo.max_path_length
        action_space = self.algo.env.action_space
        observation_space = self.algo.env.observation_space

        if hasattr(self.algo.baseline, "predict_n"):
            all_path_baselines = self.algo.baseline.predict_n(paths)
        else:
            all_path_baselines = [
                self.algo.baseline.predict(path) for path in paths
            ]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["deltas"] = deltas

        # calculate trajectory tensors (TODO: probably can do this in TF)
        for idx, path in enumerate(paths):
            # baselines
            path['baselines'] = all_path_baselines[idx]
            baselines.append(path['baselines'])

            # returns
            path["returns"] = special.discount_cumsum(path["rewards"],
                                                      self.algo.discount)
            returns.append(path["returns"])

            # Calculate trajectory samples
            #
            # Pad and flatten action and observation traces
            act = tensor_utils.pad_tensor(path['actions'], max_path_length)
            obs = tensor_utils.pad_tensor(path['observations'],
                                          max_path_length)
            act_flat = action_space.flatten_n(act)
            obs_flat = observation_space.flatten_n(obs)
            # Create a time series of stacked [act, obs] vectors
            #XXX now the inference network only looks at obs vectors
            #act_obs = np.concatenate([act_flat, obs_flat], axis=1)  # TODO reactivate for harder envs?
            act_obs = obs_flat
            # act_obs = act_flat
            # Calculate a forward-looking sliding window of the stacked vectors
            #
            # If act_obs has shape (n, d), then trajs will have shape
            # (n, window, d)
            #
            # The length of the sliding window is determined by the trajectory
            # inference spec. We smear the last few elements to preserve the
            # time dimension.
            window = self.algo.inference.input_space.shape[0]
            trajs = sliding_window(act_obs, window, 1, smear=True)
            trajs_flat = self.algo.inference.input_space.flatten_n(trajs)
            path['trajectories'] = trajs_flat

            # trajectory infos
            _, traj_infos = self.algo.inference.get_latents(trajs)
            path['trajectory_infos'] = traj_infos

        ev = special.explained_variance_1d(
            np.concatenate(baselines), np.concatenate(returns))

        #DEBUG CPU vars ######################
        cpu_adv = tensor_utils.concat_tensor_list(
            [path["advantages"] for path in paths])
        cpu_deltas = tensor_utils.concat_tensor_list(
            [path["deltas"] for path in paths])
        cpu_act = tensor_utils.concat_tensor_list(
            [path["actions"] for path in paths])
        cpu_obs = tensor_utils.concat_tensor_list(
            [path["observations"] for path in paths])
        cpu_agent_infos = tensor_utils.concat_tensor_dict_list(
            [path["agent_infos"] for path in paths])

        if self.algo.center_adv:
            cpu_adv = utils.center_advantages(cpu_adv)

        if self.algo.positive_adv:
            cpu_adv = utils.shift_advantages_to_positive(cpu_adv)
        #####################################

        # Hindsight trajectories ##############################
        hindsight_data = []
        all_tasks_one_hots = self.env.env.all_task_one_hots
        while len(hindsight_data) <= 100:
            success = False
            task1_id = int(np.random.rand() * len(all_tasks_one_hots))
            task2_id = int(np.random.rand() * len(all_tasks_one_hots))
            concated_task = np.concatenate([all_tasks_one_hots[task1_id], all_tasks_one_hots[task2_id]], axis=0)
            for path in paths:
                # Hard code everything for now... No time before deadline
                # TODO cleanup
                if path['tasks'][0].all() == all_tasks_one_hots[task1_id].all():
                    # search for the seconde traj to assemble
                    for _path in paths:
                        if _path['tasks'][0].all() == all_tasks_one_hots[task2_id].all():
                            # concat obs, task inputs and actions sequence
                            distance = np.linalg.norm(path['observations'][-1, :] - _path['observations'][0, :])
                            learning_rate = (-np.tanh(distance) / 2. + 1)
                            if learning_rate <= 0.5:
                                learning_rate = 0.
                                break
                            second_traj = _path['observations'].copy()
                            second_traj[:, 2:] = path['observations'][:, 2:]
                            concat_obs = np.vstack((path['observations'], second_traj))
                            concat_act = np.vstack((path['actions'], _path['actions']))
                            concat_tasks = np.array([np.copy(concated_task) for i in range(len(concat_obs))])
                            concat_lr = np.array([learning_rate for i in range(len(concat_obs))])
                            concat_path = dict(
                                observations=concat_obs,
                                actions=concat_act,
                                tasks=concat_tasks,
                                hs_lr=concat_lr[:, np.newaxis],
                            )
                            hindsight_data.append(concat_path)
                            success = True
                            break

                if success:
                    break

        print(len(hindsight_data))
        tabular.record('Hindsight_data_len', len(hindsight_data))
        # process hindsight data like sampled data
        hindsight_obs = [path["observations"] for path in hindsight_data]
        hindsight_obs = tensor_utils.pad_tensor_n(hindsight_obs, max_path_length*2)

        hindsight_actions = [path["actions"] for path in hindsight_data]
        hindsight_actions = tensor_utils.pad_tensor_n(hindsight_actions, max_path_length*2)

        hindsight_tasks = [path["tasks"] for path in hindsight_data]
        hindsight_tasks = tensor_utils.pad_tensor_n(hindsight_tasks, max_path_length*2)

        hindsight_lr = [path['hs_lr'] for path in hindsight_data]
        hindsight_lr = tensor_utils.pad_tensor_n(hindsight_lr, max_path_length*2)

        hindsight_data = dict(
            observations=hindsight_obs,
            actions=hindsight_actions,
            tasks=hindsight_tasks,
            lr=hindsight_lr)

        # make all paths the same length
        obs = [path["observations"] for path in paths]
        obs = tensor_utils.pad_tensor_n(obs, max_path_length)

        actions = [path["actions"] for path in paths]
        actions = tensor_utils.pad_tensor_n(actions, max_path_length)

        tasks = [path["tasks"] for path in paths]
        tasks = tensor_utils.pad_tensor_n(tasks, max_path_length)

        tasks_gt = [path['tasks_gt'] for path in paths]
        tasks_gt = tensor_utils.pad_tensor_n(tasks_gt, max_path_length)

        latents = [path['latents'] for path in paths]
        latents = tensor_utils.pad_tensor_n(latents, max_path_length)

        rewards = [path["rewards"] for path in paths]
        rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

        returns = [path["returns"] for path in paths]
        returns = tensor_utils.pad_tensor_n(returns, max_path_length)

        baselines = tensor_utils.pad_tensor_n(baselines, max_path_length)

        trajectories = tensor_utils.stack_tensor_list(
            [path["trajectories"] for path in paths])

        agent_infos = [path["agent_infos"] for path in paths]
        agent_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length)
            for p in agent_infos
        ])

        latent_infos = [path["latent_infos"] for path in paths]
        latent_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length)
            for p in latent_infos
        ])

        trajectory_infos = [path["trajectory_infos"] for path in paths]
        trajectory_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length)
            for p in trajectory_infos
        ])

        env_infos = [path["env_infos"] for path in paths]
        env_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos
        ])

        valids = [np.ones_like(path["returns"]) for path in paths]
        valids = tensor_utils.pad_tensor_n(valids, max_path_length)

        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        ent = np.sum(
            self.algo.policy.distribution.entropy(agent_infos) *
            valids) / np.sum(valids)

        samples_data = dict(
            observations=obs,
            actions=actions,
            tasks=tasks,
            latents=latents,
            trajectories=trajectories,
            rewards=rewards,
            baselines=baselines,
            returns=returns,
            valids=valids,
            agent_infos=agent_infos,
            latent_infos=latent_infos,
            trajectory_infos=trajectory_infos,
            env_infos=env_infos,
            paths=paths,
            cpu_adv=cpu_adv,  #DEBUG
            cpu_deltas=cpu_deltas,  #DEBUG
            cpu_obs=cpu_obs,  #DEBUG
            cpu_act=cpu_act,  #DEBUG
            cpu_agent_infos=cpu_agent_infos,  # DEBUG
        )

        tabular.record('Iteration', itr)
        tabular.record('AverageDiscountedReturn',
                              average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('NumTrajs', len(paths))
        tabular.record('Entropy', ent)
        tabular.record('Perplexity', np.exp(ent))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))

        return samples_data, hindsight_data

    #TODO: embedding-specific diagnostics
    def log_diagnostics(self, paths):
        return super().log_diagnostics(paths)
