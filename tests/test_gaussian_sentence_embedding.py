import tensorflow as tf
import numpy as np
from akro.tf import Box

from embed2learn.envs.multi_task_env import TfEnv
from embed2learn.envs import MultiTaskEnv
from embed2learn.envs import PointEnv
from embed2learn.embeddings import EmbeddingSpec
from embed2learn.embeddings.gaussian_sentence_embedding import GaussianSentenceEmbedding

def circle(r, n):
    for t in np.arange(0, 2 * np.pi, 2 * np.pi / n):
        yield r * np.sin(t), r * np.cos(t)
goals = circle(3.0, 2)

TASKS = {
    str(i + 1): {
        'args': [],
        'kwargs': {
            'goal': g,
            'never_done': True,
            'completion_bonus': 0.0,
            'action_scale': 0.1,
            'random_start': False,
            'sentence_code_dim':2,
            'max_sentence_length': 4,
        }
    }
    for i, g in enumerate(goals)
}


def test_gse():

    task_names = sorted(TASKS.keys())
    task_args = [TASKS[t]['args'] for t in task_names]
    task_kwargs = [TASKS[t]['kwargs'] for t in task_names]
    latent_length = 4
    latent_lb = np.zeros(latent_length, )
    latent_ub = np.ones(latent_length, )
    latent_space = Box(latent_lb, latent_ub)

    env = TfEnv(
        MultiTaskEnv(
            task_env_cls=PointEnv,
            task_args=task_args,
            task_kwargs=task_kwargs))
    print(env.task_space)
    task_embed_spec = EmbeddingSpec(env.task_space, latent_space)
    embeddings = GaussianSentenceEmbedding(
        embedding_spec=task_embed_spec, 
    )


test_gse()
