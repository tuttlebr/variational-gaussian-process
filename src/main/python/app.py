from logging import basicConfig, info

basicConfig(
    format="%(asctime)s %(message)s",
    level="INFO",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import variational_gaussian_process as bf
from config import Config


for embed_id in Config.embedding_split:
    try:
        info("_" * 80)
        model = bf.VGPModel()
        info("Currently working on embed_id: {}".format(embed_id))
        model.prepare_data(embed_id)
        model.begin_training()
        bf.plot_distributions(model)
    except Exception as e:
        info("{} failed.\nReason: {}".format(embed_id, e))
