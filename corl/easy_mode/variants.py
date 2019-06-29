# use_softplus_entropy = [True, False]
# latent_length = 8, 10


use_softplus_entropy = [True, False]
latent_length = [8, 10]


TE_EASY_CONFIGS = []
N_TASK = 10


for sp in use_softplus_entropy:
    for ll in latent_length:
        TE_EASY_CONFIGS.append(
            dict(
                latent_length=ll,
                inference_window=20,
                batch_size=4096 * N_TASK,
                policy_ent_coeff=5e-3,  # 1e-2
                embedding_ent_coeff=1e-3,  # 1e-3
                inference_ce_coeff=5e-3,  # 1e-4
                max_path_length=200,
                embedding_init_std=1.0,
                embedding_max_std=2.0,
                policy_init_std=1.0,
                use_softplus_entropy=sp,
            )
        )

