from types import SimpleNamespace

from akro.tf import Box
from garage.envs.env_spec import EnvSpec
from garage.experiment import run_experiment
import numpy as np

from embed2learn.algos import PPOTaskEmbedding
from embed2learn.baselines import MultiTaskGaussianMLPBaseline
from embed2learn.envs import MultiTaskEnv
from embed2learn.envs.multi_task_env import TfEnv
from embed2learn.embeddings import EmbeddingSpec
from embed2learn.embeddings import GaussianMLPEmbedding
from embed2learn.embeddings.utils import concat_spaces
from embed2learn.experiment import TaskEmbeddingRunner
from embed2learn.policies import GaussianMLPMultitaskPolicy

from multiworld.envs.mujoco.sawyer_xyz.sawyer_box_close_6dof import SawyerBoxClose6DOFEnv


N_TASKS = 2
EXP_PREFIX = 'corl_te_box_closing'


def run_task(v):

    hand_low = np.array((-0.5, 0.40, 0.05))
    hand_high = np.array((0.5, 1, 0.5))

    HAND_INITS = np.random.uniform(low=hand_low, high=hand_high, size=(N_TASKS, len(hand_low))).tolist()
    print(HAND_INITS)
    TASKS = {
        str(i + 1): {
            "args": [],
            "kwargs": {
                'hand_init_pos': tuple(h),
            }
        }
        for i, h in enumerate(HAND_INITS)
    }
    v['tasks'] = TASKS
    v = SimpleNamespace(**v)

    task_names = sorted(v.tasks.keys())
    task_args = [v.tasks[t]['args'] for t in task_names]
    task_kwargs = [v.tasks[t]['kwargs'] for t in task_names]

    with TaskEmbeddingRunner() as runner:
        # Environment
        env = TfEnv(
                MultiTaskEnv(
                    task_env_cls=SawyerBoxClose6DOFEnv,
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
        task_obs_space = concat_spaces(env.task_space, env.observation_space)
        env_spec_embed = EnvSpec(task_obs_space, env.action_space)

        # TODO(): rename to inference_network
        traj_embedding = GaussianMLPEmbedding(
            name="inference",
            embedding_spec=traj_embed_spec,
            hidden_sizes=(200, 100),  # was the same size as policy in Karol's paper
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
            hidden_sizes=(200, 100),
            std_share_network=True,
            init_std=v.policy_init_std,
        )

        extra = v.latent_length + len(v.tasks)
        baseline = MultiTaskGaussianMLPBaseline(
            env_spec=env.spec,
            extra_dims=extra,
            regressor_args=dict(hidden_sizes=(200, 100)),
        )

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
            use_softplus_entropy=True,
        )
        runner.setup(algo, env, batch_size=v.batch_size,
            max_path_length=v.max_path_length)
        runner.train(n_epochs=2000, plot=False)

config = dict(
    latent_length=3,
    inference_window=6,
    batch_size=4096 * N_TASKS,
    policy_ent_coeff=5e-3,  # 1e-2
    embedding_ent_coeff=1e-3,  # 1e-3
    inference_ce_coeff=5e-3,  # 1e-4
    max_path_length=200,
    embedding_init_std=1.0,
    embedding_max_std=2.0,
    policy_init_std=1.0,
)

run_experiment(
    run_task,
    exp_prefix=EXP_PREFIX,
    n_parallel=1,
    seed=1,
    variant=config,
    plot=False,
)
