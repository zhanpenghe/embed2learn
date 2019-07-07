import argparse
from types import SimpleNamespace

from akro.tf import Box
from garage.envs.env_spec import EnvSpec
from garage.experiment import run_experiment
import numpy as np

from embed2learn.algos import PPOTaskEmbedding
from embed2learn.baselines import MultiTaskGaussianMLPBaseline
from embed2learn.baselines import MultiTaskLinearFeatureBaseline
from embed2learn.envs import MultiClassMultiTaskEnv
from embed2learn.envs.multi_task_env import TfEnv
from embed2learn.embeddings import EmbeddingSpec
from embed2learn.embeddings import GaussianMLPEmbedding
from embed2learn.embeddings.utils import concat_spaces
from embed2learn.experiment import TaskEmbeddingRunner
from embed2learn.policies import GaussianMLPMultitaskPolicy

from env_lists import EASY_MODE_DICT, EASY_MODE_ARGS_KWARGS


N_TASKS = len(EASY_MODE_DICT.keys())


def run_task(v):

    v = SimpleNamespace(**v)
    with TaskEmbeddingRunner() as runner:
        # Environment
        env = TfEnv(
                MultiClassMultiTaskEnv(
                    task_env_cls_dict=EASY_MODE_DICT,
                    task_args_kwargs=EASY_MODE_ARGS_KWARGS,
                ))

        # Latent space and embedding specs
        # TODO(gh/10): this should probably be done in Embedding or Algo
        latent_lb = np.zeros(v.latent_length, )
        latent_ub = np.ones(v.latent_length, )
        latent_space = Box(latent_lb, latent_ub)

        # trajectory space is (TRAJ_ENC_WINDOW, act_obs) where act_obs is a stacked
        # vector of flattened actions and observations
        act_lb, act_ub = env.action_space.bounds
        act_lb_flat = env.action_space.flatten(act_lb)
        act_ub_flat = env.action_space.flatten(act_ub)
        obs_lb, obs_ub = env.observation_space.bounds
        obs_lb_flat = env.observation_space.flatten(obs_lb)
        obs_ub_flat = env.observation_space.flatten(obs_ub)
        # act_obs_lb = np.concatenate([act_lb_flat, obs_lb_flat])
        # act_obs_ub = np.concatenate([act_ub_flat, obs_ub_flat])
        act_obs_lb = obs_lb_flat
        act_obs_ub = obs_ub_flat
        # act_obs_lb = act_lb_flat
        # act_obs_ub = act_ub_flat
        traj_lb = np.stack([act_obs_lb] * v.inference_window)
        traj_ub = np.stack([act_obs_ub] * v.inference_window)
        traj_space = Box(traj_lb, traj_ub)

        task_embed_spec = EmbeddingSpec(env.task_space, latent_space)
        traj_embed_spec = EmbeddingSpec(traj_space, latent_space)
        task_obs_space = concat_spaces(env.task_space, env.observation_space)
        env_spec_embed = EnvSpec(task_obs_space, env.action_space)

        # TODO(): rename to inference_network
        traj_embedding = GaussianMLPEmbedding(
            name="inference",
            embedding_spec=traj_embed_spec,
            hidden_sizes=(200, 200),  # was the same size as policy in Karol's paper
            std_share_network=True,
            init_std=2.0,
        )

        # Embeddings
        task_embedding = GaussianMLPEmbedding(
            name="embedding",
            embedding_spec=task_embed_spec,
            hidden_sizes=(200, 200),
            std_share_network=True,
            init_std=v.embedding_init_std,
            max_std=v.embedding_max_std,
        )

        # Multitask policy
        policy = GaussianMLPMultitaskPolicy(
            name="policy",
            env_spec=env.spec,
            task_space=env.task_space,
            embedding=task_embedding,
            hidden_sizes=(200, 200),
            std_share_network=True,
            init_std=v.policy_init_std,
        )

        extra = v.latent_length + N_TASKS
        # baseline = MultiTaskGaussianMLPBaseline(
        #     env_spec=env.spec,
        #     extra_dims=extra,
        #     regressor_args=dict(hidden_sizes=(200, 200)),
        # )
        baseline = MultiTaskLinearFeatureBaseline(env.spec)

        algo = PPOTaskEmbedding(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            inference=traj_embedding,
            max_path_length=v.max_path_length,
            n_itr=2000,
            discount=0.99,
            lr_clip_range=0.2,
            policy_ent_coeff=v.policy_ent_coeff,
            embedding_ent_coeff=v.embedding_ent_coeff,
            inference_ce_coeff=v.inference_ce_coeff,
            use_softplus_entropy=v.use_softplus_entropy,
        )
        runner.setup(algo, env, batch_size=v.batch_size,
            max_path_length=v.max_path_length)
        runner.train(n_epochs=int(1e7), plot=False)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Play a pickled policy.')
#     parser.add_argument('variant_index', metavar='variant_index', type=int,
#                     help='The index of variants to use for experiment')
#     args = parser.parse_args()

#     from variants import TE_EASY_CONFIGS
    
#     config = TE_EASY_CONFIGS[args.variant_index]
#     exp_prefix = 'corl_te_easy_10tasks_usesp{}_latentlen{}'.format(config['use_softplus_entropy'], config['latent_length'])

#     run_experiment(
#         run_task,
#         exp_prefix=exp_prefix,
#         n_parallel=1,
#         seed=1,
#         variant=config,
#         plot=False,
#     )


if __name__ == '__main__':
    import os
    import joblib
    import tensorflow as tf
    from garage.misc import special
    from embed2learn.samplers.task_embedding_sampler import rollout
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pkl',
        type=str,
        default=None,
        help='Path of the pickle file to test experiment from.')
    parser.add_argument(
        '--csv',
        type=str,
        default='test.csv',
        help='Path of the csv file to save test information.')

    args = parser.parse_args()
    pkl = args.pkl
    csv_path = os.path.join(os.getcwd(), args.csv)
    with tf.Session() as sess:
        with open(pkl, 'rb') as pkl_file:
            experiment = joblib.load(pkl_file)
        print(experiment)
        env = experiment['env']
        policy = experiment['policy']

        n_rollouts = 10
        n_tasks = env.env.num_tasks

        # task_rollout_paths: [n_tasks x n_rollouts]
        task_rollout_paths = [[None] * n_rollouts for i in range(n_tasks)]
        task_idx_name = dict()

        for rollout_idx in range(n_rollouts):
            for task in range(n_tasks):
                paths, task_idx, active_env = rollout(
                    env=env,
                    agent=policy,
                    max_path_length=150,
                    always_return_paths=True,
                )
                if 'task_type' in dir(active_env):
                    name = '{}-{}'.format(str(active_env.__class__.__name__), active_env.task_type)
                else:
                    name = str(active_env.__class__.__name__)
                if task_idx not in task_idx_name:
                    task_idx_name[task_idx] = name
                task_rollout_paths[task_idx][rollout_idx] = paths

        n_tasks = len(task_rollout_paths)
        n_rollouts = len(task_rollout_paths[0])
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a+') as csv:
            if not file_exists:
                head = 'Itr,'
                for task in range(n_tasks):
                    head += 'Task{},AverageDiscountedReturn,UndiscountedReturn,SuccessRate,'.format(task)
                head = head[:-1] + '\n'
                print(head)
                csv.write(head)
                csv.flush()

            task_paths = []
            for i in range(n_tasks):
                d = dict(returns=[], success=[], rewards=[])
                task_paths.append(d)

            for meta_task, meta_path in enumerate(task_rollout_paths):
                for rollout_idx, rollout_path in enumerate(meta_path):
                    rollout_path["returns"] = special.discount_cumsum(rollout_path["rewards"], 0.99)
                    task_paths[meta_task]["returns"].append(np.mean(rollout_path["returns"]))
                    task_paths[meta_task]["rewards"].append(np.sum(rollout_path["rewards"]))
                    if "success" in rollout_path["env_infos"]:
                        if np.sum(rollout_path["env_infos"]["success"]) >= 1:
                            task_paths[meta_task]["success"].append(1)
                        else:
                            task_paths[meta_task]["success"].append(0)

            line = '{},'.format(pkl)
            for meta_task, meta_task_rollout in enumerate(task_paths):
                if not meta_task_rollout["success"]:
                    success_rate = None
                else:
                    success_rate = np.mean(meta_task_rollout["success"])
                average_discounted_return = np.mean(meta_task_rollout["returns"])
                undiscounted_returns = np.mean(meta_task_rollout["rewards"])
                line = line + '{},{},{},{},'.format(task_idx_name[meta_task], average_discounted_return, undiscounted_returns, success_rate)
            
            line = line[:-1] + '\n'
            print(line)
            csv.write(line)
            csv.flush()
