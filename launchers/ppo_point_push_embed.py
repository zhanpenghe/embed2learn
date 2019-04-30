from types import SimpleNamespace

from akro.tf import Box
from garage.envs import EnvSpec
from garage.experiment import LocalRunner, run_experiment
import numpy as np
import tensorflow as tf
import itertools
import collections

from pathlib import Path
from embed2learn.algos import PPOTaskEmbedding
from embed2learn.baselines import MultiTaskGaussianMLPBaseline
from embed2learn.envs import PointEnv
from embed2learn.envs import MultiTaskEnv
from embed2learn.envs.multi_task_env import TfEnv
from embed2learn.envs import MultiPointsPushEnv
from embed2learn.embeddings import EmbeddingSpec
from embed2learn.embeddings import GaussianMLPEmbedding, GaussianSentenceEmbedding
from embed2learn.embeddings.utils import concat_spaces
from embed2learn.experiment import TaskEmbeddingRunner
from embed2learn.policies import GaussianMLPMultitaskPolicy
from embed2learn.samplers import TaskEmbeddingSampler
from embed2learn.util import TaskDescriptionVectorizer

import gym

N = 4
# goals = circle(3.0, N)

input_tasks = [
    {
        "description": "move red box towards green box",
        "target_description": "red",
        "destination_description": "green",
    },
    {
        "description": "move green box towards yellow box",
        "target_description": "green",
        "destination_description": "yellow",
    },
    {
        "description": "move yellow box towards blue box",
        "target_description": "yellow",
        "destination_description": "blue",
    },
    {
        "description": "move blue box towards red box",
        "target_description": "yellow",
        "destination_description": "red",
    },
]

goal_descriptions = [t['description'] for t in input_tasks]

objects = [
    "red",    # rgba 1  0  0 1
    "green",  # rgba 0  .8 0 1
    "yellow", # rgba 1  1  0 1
    "blue",   # rgba .3 .3 1 1
]

task_description_vectorizer = TaskDescriptionVectorizer(
        corpus=goal_descriptions,
        max_sentence_length=6
    )

sentence_code_dim = task_description_vectorizer.sentence_code_dim
word_encoding_dim = task_description_vectorizer.sentence_code_dim

max_sentence_length = task_description_vectorizer.max_sentence_length


goal_codes = task_description_vectorizer.transform(goal_descriptions)
dictionary = task_description_vectorizer.dictionary



# task_description="move red box towards blue box", target_description="red", destination_description="blue"

TASKS = [
    {
        "goal_description": task_description_vectorizer.transform_one(t["description"]),
        "target_label": t["target_description"],
        "destination_label": t["destination_description"],
    } for i, t in enumerate(input_tasks)
]


def run_task(v):
    v = SimpleNamespace(**v)

    with TaskEmbeddingRunner() as runner:
        # Environment
        # TODO

        env = TfEnv(
                MultiPointsPushEnv(
                    tasks=TASKS,
                    sentence_code_dim=task_description_vectorizer.sentence_code_dim,
                    max_sentence_length=task_description_vectorizer.max_sentence_length,
                    )
            )

        # Latent space and embedding specs
        # TODO(gh/10): this should probably be done in Embedding or Algo
        latent_lb = np.zeros(v.latent_length, )
        latent_ub = np.ones(v.latent_length, )
        latent_space = Box(latent_lb, latent_ub)

        # trajectory space is (TRAJ_ENC_WINDOW, act_obs) where act_obs is a stacked
        # vector of flattened actions and observations
        act_lb, act_ub = env.action_space.bounds
        # act_lb_flat = env.action_space.flatten(act_lb)
        # act_ub_flat = env.action_space.flatten(act_ub)
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
        # task_obs_space = concat_spaces(env.task_space, env.observation_space)
        # env_spec_embed = EnvSpec(task_obs_space, env.action_space)

        # TODO(): rename to inference_network
        traj_embedding = GaussianMLPEmbedding(
            name="inference",
            embedding_spec=traj_embed_spec,
            hidden_sizes=(20, 10),  # was the same size as policy in Karol's paper
            std_share_network=True,
            init_std=2.0,
            mean_output_nonlinearity=None,
            min_std=v.embedding_min_std,
        )

        # Embeddings
        task_embedding = GaussianSentenceEmbedding(
            name="embedding",
            embedding_spec=task_embed_spec,
            hidden_sizes=(10,),
            std_share_network=True,
            init_std=v.embedding_init_std,
            max_std=v.embedding_max_std,
            mean_output_nonlinearity=None,
            min_std=v.embedding_min_std,
            sentence_code_dim=word_encoding_dim,
            sentence_embedding_dict_dim=10,
        )

        # Multitask policy
        policy = GaussianMLPMultitaskPolicy(
            name="policy",
            env_spec=env.spec,
            task_space=env.task_space,
            embedding=task_embedding,
            hidden_sizes=(32, 16),
            std_share_network=True,
            max_std=v.policy_max_std,
            init_std=v.policy_init_std,
            min_std=v.policy_min_std,
            n_tasks=4,
        )

        extra = len(v.tasks) + v.latent_length
        baseline = MultiTaskGaussianMLPBaseline(env_spec=env.spec, extra_dims=extra)

        algo = PPOTaskEmbedding(
            env=env,
            policy=policy,
            baseline=baseline,
            inference=traj_embedding,
            max_path_length=v.max_path_length,
            discount=0.99,
            lr_clip_range=0.2,
            policy_ent_coeff=v.policy_ent_coeff,
            embedding_ent_coeff=v.embedding_ent_coeff,
            inference_ce_coeff=v.inference_ce_coeff,
            use_softplus_entropy=False,
            stop_ce_gradient=True,
            max_sentence_length=max_sentence_length,
            word_encoding_dim=word_encoding_dim,
        )
        runner.setup(algo, env, batch_size=v.batch_size,
            max_path_length=v.max_path_length)

        runner.train(n_epochs=600, plot=False)

config = dict(
    dictionary=dictionary,
    tasks=TASKS,
    latent_length=6,
    inference_window=10,
    batch_size=2048 * len(TASKS),
    policy_ent_coeff=1e-5,  # 2e-2
    embedding_ent_coeff=1e-5,  # 1e-2
    inference_ce_coeff=1e-3,  # 1e-2
    max_path_length=100,
    embedding_init_std=1.0,
    embedding_max_std=2.,
    embedding_min_std=0.2,
    policy_init_std=1.0,
    policy_max_std=None,
    policy_min_std=None,
)

run_experiment(
    run_task,
    exp_prefix='ppo_point_push_embed_sentence',
    n_parallel=1,
    seed=1,
    variant=config,
    plot=False,
)
