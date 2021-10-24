from logging import basicConfig, info

basicConfig(
    format="%(asctime)s %(message)s",
    level="INFO",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import os
from pathlib import Path
import tensorflow as tf
from attrdict import AttrDict
from tensorflow.python.client import device_lib


def check_tensor_core_gpu_present():

    """example use, if you want to review gpu compute capability.
    info("Tensor Core GPU Present:", check_tensor_core_gpu_present())"""

    local_device_protos = device_lib.list_local_devices()
    for line in local_device_protos:
        if "compute capability" in str(line):
            compute_capability = float(
                line.physical_device_desconfig_dict.split(
                    "compute capability: "
                )[-1]
            )
            if compute_capability >= 7.0:
                return True


def initialize_globals():
    config_dict = AttrDict()

    config_dict.model_directory = "model_directory"
    config_dict.log_directory = "log_directory"
    config_dict.train = True
    config_dict.load_weights = None

    config_dict.data = "/app/src/unittest/sample_data.csv"

    config_dict.embedding_split = {"qa_0": "2021-10-13"}

    config_dict.manual_peers = None
    config_dict.embed_id_col_name = "embedding"
    config_dict.kpi_col_name = "y"
    config_dict.date_col_name = "ds"
    config_dict.top_n_peer_count = 10

    config_dict.batch_size_per_replica = 16
    config_dict.n_epoch = 1000
    config_dict.dense_layer_geometry = 16
    config_dict.dense_layer_count = 0
    config_dict.learning_rate = 0.005
    config_dict.upper_bound = 0.975
    config_dict.lower_bound = 0.025
    config_dict.kernel_initializer = tf.keras.initializers.Ones()
    config_dict.kernel_constraint = None  # tf.keras.constraints.MaxNorm(4)
    config_dict.activation = tf.keras.layers.ReLU()
    config_dict.normalization = None  # tf.keras.layers.LayerNormalization()

    # vgp kernel provider variables
    config_dict.constant_initializer = 0.54
    config_dict.kernel_size = 730
    config_dict.num_inducing_points = int(config_dict.kernel_size * 0.85)

    info(
        "[+] Available devices: {}".format(
            tf.config.list_physical_devices("GPU")
        )
    )

    if config_dict.train:
        info("[+] Training")
    else:
        info("[-] Not Training")
    if config_dict.load_weights:
        try:
            assert os.path.isdir(
                os.path.dirname(config_dict.load_weights)
            ) or os.path.isfile(config_dict.load_weights)
            info(
                "[+] Model Weights Directory Exists: {}".format(
                    config_dict.load_weights
                )
            )
        except Exception as runtime_exception:
            info(
                "[-] Model Weights Directory Doesn't Exist: {}".format(
                    config_dict.load_weights,
                )
            )
    try:
        assert os.path.isdir(config_dict.model_directory)
        info("[+] Model Directory Exists:".format(config_dict.model_directory))
    except Exception as runtime_exception:
        info(
            "[-] Model Directory Not Found - Creating Automatically:".format(
                config_dict.model_directory,
            )
        )
        Path(config_dict.model_directory).mkdir(parents=True, exist_ok=True)
    try:
        assert os.path.isdir(config_dict.log_directory)
        info("[+] Log Directory Exists:".format(config_dict.log_directory))
    except Exception as runtime_exception:
        info(
            "[-] Log Directory Not Found - Creating Automatically:".format(
                config_dict.log_directory
            )
        )
        Path(config_dict.log_directory).mkdir(parents=True, exist_ok=True)
    for i in config_dict:
        info(" * {}: {}".format(i, config_dict[i]))
    return config_dict


Config = initialize_globals()
