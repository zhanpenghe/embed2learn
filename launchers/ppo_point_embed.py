from types import SimpleNamespace

from akro.tf import Box
from garage.envs import EnvSpec
from garage.experiment import LocalRunner, run_experiment
import numpy as np
import tensorflow as tf
import itertools
import collections

from embed2learn.algos import PPOTaskEmbedding
from embed2learn.baselines import MultiTaskGaussianMLPBaseline
from embed2learn.envs import PointEnv
from embed2learn.envs import MultiTaskEnv
from embed2learn.envs.multi_task_env import TfEnv
from embed2learn.embeddings import EmbeddingSpec
from embed2learn.embeddings import GaussianMLPEmbedding, GaussianSentenceEmbedding
from embed2learn.embeddings.utils import concat_spaces
from embed2learn.experiment import TaskEmbeddingRunner
from embed2learn.policies import GaussianMLPMultitaskPolicy
from embed2learn.samplers import TaskEmbeddingSampler


def circle(r, n):
    for t in np.arange(0, 2 * np.pi, 2 * np.pi / n):
        yield r * np.sin(t), r * np.cos(t)

def build_dataset(words, vocab_size):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    word_to_index_map = {}
    index = 0
    for word, _ in count:
        word_to_index_map[word] = index
        index += 1
    return word_to_index_map

def description_to_code(description, dictionary, max_sentence_length):
    code = []
    for s in description.split(' '):
        code.append(dictionary[s])
    while len(code) < max_sentence_length:
        code.append(0)
    return code

N = 4
goals = circle(3.0, N)
goal_descriptions = ['move right', 'move up', 'move left', 'move down']
words = list(itertools.chain.from_iterable([s.split(' ') for s in goal_descriptions]))
vocab_size = len(words)
word_to_index_map = build_dataset(words, vocab_size)
max_sentence_length = 3
sentence_code_dim = len(word_to_index_map)
goal_codes = [description_to_code(s, word_to_index_map, max_sentence_length) for s in goal_descriptions]
TASKS = {
    str(i + 1): {
        'args': [],
        'kwargs': {
            'goal': g,
            'goal_description': goal_codes[i],
            'sentence_code_dim': sentence_code_dim,
            'max_sentence_length': max_sentence_length,
            'never_done': True,
            'completion_bonus': 0.0,
            'action_scale': 0.1,
            'random_start': False,
        }
    }
    for i, g in enumerate(goals)
}


def run_task(v):
    v = SimpleNamespace(**v)

    with TaskEmbeddingRunner() as runner:
        task_names = sorted(v.tasks.keys())
        task_args = [v.tasks[t]['args'] for t in task_names]
        task_kwargs = [v.tasks[t]['kwargs'] for t in task_names]

        # Environment
        env = TfEnv(
                MultiTaskEnv(
                    task_env_cls=PointEnv,
                    task_args=task_args,
                    task_kwargs=task_kwargs))

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
        # task_obs_space = concat_spaces(env.task_space, env.observation_space)
        # env_spec_embed = EnvSpec(task_obs_space, env.action_space)

        # TODO(): rename to inference_network
        traj_embedding = GaussianMLPEmbedding(
            name="inference",
            embedding_spec=traj_embed_spec,
            hidden_sizes=(20, 10),  # was the same size as policy in Karol's paper
            std_share_network=True,
            init_std=2.0,
            mean_output_nonlinearity=tf.nn.tanh,
            min_std=v.embedding_min_std,
        )

        # Embeddings
        task_embedding = GaussianSentenceEmbedding(
            name="embedding",
            embedding_spec=task_embed_spec,
            hidden_sizes=(20, 20),
            std_share_network=True,
            init_std=v.embedding_init_std,
            max_std=v.embedding_max_std,
            mean_output_nonlinearity=tf.nn.tanh,
            min_std=v.embedding_min_std,
            sentence_code_dim=sentence_code_dim,
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

        extra = v.latent_length + len(v.tasks)
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
            use_softplus_entropy=True,
            stop_ce_gradient=True,
        )

        runner.setup(algo, env, batch_size=v.batch_size,
            max_path_length=v.max_path_length)
        runner.train(n_epochs=600, plot=False)

config = dict(
    tasks=TASKS,
    latent_length=2,
    inference_window=2,
    batch_size=1024 * len(TASKS),
    policy_ent_coeff=192e-2,  # 2e-2
    embedding_ent_coeff=2.2e-3,  # 1e-2
    inference_ce_coeff=5e-2,  # 1e-2
    max_path_length=100,
    embedding_init_std=1.0,
    embedding_max_std=2.0,
    embedding_min_std=0.38,
    policy_init_std=1.0,
    policy_max_std=None,
    policy_min_std=None,
)

run_experiment(
    run_task,
    exp_prefix='ppo_point_embed_random_start_192_polent_300maxpath',
    n_parallel=1,
    seed=1,
    variant=config,
    plot=False,
)
