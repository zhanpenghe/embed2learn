from types import SimpleNamespace

from garage.experiment import LocalRunner, run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from embed2learn.util import TaskDescriptionVectorizer
from embed2learn.envs import MultiPointsPushEnv



from embed2learn.policies import GaussianMLPPolicy


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


TASKS = [
    {
        "goal_description": task_description_vectorizer.transform_one(t["description"]),
        "target_label": t["target_description"],
        "destination_label": t["destination_description"],
    } for i, t in enumerate(input_tasks)
]



def run_task(v):
    with LocalRunner() as runner:
        v = SimpleNamespace(**v)

        # Environment
        # TODO

        env = MultiPointsPushEnv(
            tasks=TASKS,
            sentence_code_dim=task_description_vectorizer.sentence_code_dim,
            max_sentence_length=task_description_vectorizer.max_sentence_length,
            )

        env = TfEnv(env)

        # Policy
        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=(256,128),
            init_std=v.policy_init_std,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(hidden_sizes=(256,128)),
        )

        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            batch_size=v.batch_size,  # 4096
            max_path_length=v.max_path_length,
            discount=0.99,
            lr_clip_range=0.2,
            optimizer_args=dict(batch_size=32, max_epochs=10),
        )

        runner.setup(algo, env)
        runner.train(n_epochs=2000, batch_size=v.batch_size)


config = dict(
    batch_size=4096,
    max_path_length=500,  # 50
    policy_init_std=1.0,  # 1.0
)

run_experiment(
    run_task,
    exp_prefix='point_pusher_ppo',
    n_parallel=4,
    seed=1,
    variant=config,
    plot=True,
)
