from garage.experiment import LocalRunner, run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.samplers import BatchSampler

from embed2learn.envs import PointEnv

def run_task(*_):
    with LocalRunner() as runner:
        env = TfEnv(
            PointEnv(
                goal=(1, 1),
                action_scale=0.1,
                never_done=True),)

        policy = GaussianMLPPolicy(
            name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=50,
            discount=0.99,
            max_kl_step=0.01,
        )

        batch_size = 4000
        max_path_length = 50
        n_envs = batch_size // max_path_length

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=batch_size, plot=False)

run_experiment(
    run_task,
    snapshot_mode="last",
    seed=1,
)
