from logging import basicConfig, info

basicConfig(
    format="%(asctime)s %(message)s",
    level="CRITICAL",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import tensorflow as tf
import os
import uuid
from sklearn.preprocessing import StandardScaler as DataScaler
import numpy as np
import pandas as pd

from config import Config
from evaluate import performance_report
from feeder import (
    dataframe_cointegration,
    dataframe_linear_interpolation,
    dataframe_pivot,
    dataframe_split_by_date,
    read_csv,
    retrieve_inputs,
)
from vgpmodel import get_distributions, plot_uncertainty, vgp_model


class VGPModel:
    def __init__(self):
        tf.random.set_seed(42)
        self.early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=3,
            min_delta=0.01,
            restore_best_weights=True,
        )

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(Config.log_directory)
        )

        self.auto_decay = tf.keras.callbacks.ReduceLROnPlateau(
            patience=2, factor=0.8
        )

        self.callbacks = [
            self.early_stopping_callback,
            self.tensorboard_callback,
            self.auto_decay,
        ]

        self.model = vgp_model()

    def prepare_data(self, embed_id):
        self.embed_id = embed_id
        self.split_on_date = Config.embedding_split[self.embed_id]
        df_total = read_csv(Config.data)
        df_total = dataframe_pivot(df_total)
        df_total = dataframe_linear_interpolation(df_total)
        self.train_df, self.eval_df = dataframe_split_by_date(
            df_total, self.split_on_date
        )
        self.train_df, self.eval_df, self.peer_list = dataframe_cointegration(
            self.train_df, self.eval_df, self.embed_id
        )

        self.x_train, self.y_train = retrieve_inputs(
            self.train_df, self.peer_list
        )
        self.x_eval, self.y_eval = retrieve_inputs(
            self.eval_df, self.peer_list
        )

        x_tf_scaler = DataScaler()
        x_tf_scaler.fit(self.x_train)

        y_tf_scaler = DataScaler()
        y_tf_scaler.fit(self.y_train.reshape(-1, 1))

        self.x_tf_scaler = x_tf_scaler
        self.y_tf_scaler = y_tf_scaler

    def begin_training(self):
        def var_loss(y, rv_y):
            return rv_y.variational_loss(
                y,
                kl_weight=np.array(Config.batch_size_per_replica, np.float64)
                / Config.kernel_size,
            )

        # Ideal steps per train/eval

        STEPS_PER_EPOCH = (
            self.y_train.shape[0] // Config.batch_size_per_replica
        )
        VALIDATION_STEPS = (
            self.y_eval.shape[0] // Config.batch_size_per_replica
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=Config.learning_rate
        )

        self.model.compile(
            loss=var_loss,
            optimizer=optimizer,
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(),
                tf.keras.metrics.MeanAbsoluteError(),
            ],
        )

        self.model.fit(
            x=self.x_tf_scaler.transform(self.x_train),
            y=self.y_tf_scaler.transform(self.y_train.reshape(-1, 1)).ravel(),
            batch_size=Config.batch_size_per_replica,
            epochs=Config.n_epoch,
            shuffle=True,
            callbacks=self.callbacks,
            validation_data=(
                self.x_tf_scaler.transform(self.x_eval),
                self.y_tf_scaler.transform(self.y_eval.reshape(-1, 1)).ravel(),
            ),
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS,
            verbose=False,
        )


def plot_distributions(model_object):
    all_data = np.concatenate((model_object.x_train, model_object.x_eval))
    df_predictions = get_distributions(all_data, model_object)
    df_train_eval = pd.concat(
        [model_object.train_df, model_object.eval_df], ignore_index=True
    )
    df_all = df_train_eval.merge(
        df_predictions, left_index=True, right_index=True
    )
    key = "{}-{}".format(model_object.embed_id, uuid.uuid4())
    df_all.to_csv("dist_{}.csv".format(key), index=False)
    titles = "Variational Gaussian Process Backcast Store Number {}".format(
        model_object.embed_id
    )
    info("{}".format(performance_report(df_all, model_object.split_on_date)))
    return plot_uncertainty(
        df_all, threshold_date=model_object.split_on_date, title=titles
    )
