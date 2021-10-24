from logging import basicConfig, info

basicConfig(
    format="%(asctime)s %(message)s", level="CRITICAL", datefmt="%Y-%m-%d %H:%M:%S",
)
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from config import Config
from matplotlib import pylab as plt

# set color scheme
sns.set(rc={"figure.figsize": (24, 12)})
colors = [
    (0.89370079, 0.09448819, 0.21653543),
    (0.0, 0.39370079, 0.57086614),
    (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
    (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
]
c1, c2, c3, c4 = colors[0], colors[1], colors[2], colors[3]


# Variational Gussian Process kernel
class RBFKernelFn(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RBFKernelFn, self).__init__(**kwargs)

        self._amplitude = self.add_weight(
            initializer=tf.constant_initializer(0), dtype=np.float64, name="amplitude",
        )

        self._length_scale = self.add_weight(
            initializer=tf.constant_initializer(0),
            dtype=np.float64,
            name="length_scale",
        )

    def call(self, x):
        # some tf.keras nonsense for calling something...
        return x

    @property
    def kernel(self):
        amplitude = tf.nn.softplus(0.1 * self._amplitude, name="amplitude")

        length_scale = tf.nn.softplus(5.0 * self._length_scale, name="length_scale")

        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=amplitude, length_scale=length_scale
        )


def vgp_model(learning_rate=Config.learning_rate):

    tf.keras.backend.clear_session()

    model = tf.keras.Sequential(name="vgpModel")

    model.add(
        tf.keras.layers.InputLayer(input_shape=[Config.top_n_peer_count], name="inputs")
    )

    for _ in range(Config.dense_layer_count):
        model.add(
            tf.keras.layers.Dense(
                Config.dense_layer_geometry,
                activation=None,
                kernel_initializer=Config.kernel_initializer,
                kernel_constraint=Config.kernel_constraint,
                name="d_{}".format(l),
            )
        )
        if Config.normalization:
            model.add(Config.normalization)
        model.add(Config.activation)
    model.add(
        tf.keras.layers.Dense(
            1, activation=None, use_bias=False, name="point_prediction",
        )
    )

    x_range = [0, Config.kernel_size]

    model.add(
        tfp.layers.VariationalGaussianProcess(
            num_inducing_points=Config.num_inducing_points,
            kernel_provider=RBFKernelFn(),
            event_shape=[1],
            inducing_index_points_initializer=tf.constant_initializer(
                np.linspace(*x_range, num=Config.num_inducing_points, dtype=np.float64)[
                    ..., np.newaxis
                ]
            ),
            unconstrained_observation_noise_variance_initializer=tf.constant_initializer(
                np.array(Config.constant_initializer).astype(np.float64)
            ),
            name="vgp_output",
        )
    )

    return model


def get_distributions(data, model, distributions=300):

    yhat = model.model(model.x_tf_scaler.transform(data))

    df = pd.DataFrame({"id": range(len(data))})

    q_lower = Config.lower_bound
    q_higher = Config.upper_bound

    for i in range(distributions):
        p = 1 + i
        print("Sampling progress: {0:.0%}".format(p / distributions), end="\r")
        y = yhat.sample().numpy().ravel()
        y = model.y_tf_scaler.inverse_transform(y.reshape(-1, 1)).ravel()
        samp = pd.DataFrame(data={"dist_{}".format(i): y})
        df = pd.concat([df, samp], axis=1)
    df["y_hat_lower"] = df[df.columns[df.columns.str.contains("dist")]].quantile(
        axis="columns", q=q_lower
    )
    df["y_hat_upper"] = df[df.columns[df.columns.str.contains("dist")]].quantile(
        axis="columns", q=q_higher
    )
    df["y_hat"] = df[df.columns[df.columns.str.contains("dist")]].quantile(
        axis="columns", q=0.5
    )

    return df


def plot_uncertainty(
    df_prediction, threshold_date, title="Variational Gaussian Process Backcast",
):
    date_split = pd.to_datetime(threshold_date)
    df_prediction[Config.date_col_name] = pd.to_datetime(
        df_prediction[Config.date_col_name]
    )

    y_hat_upper = df_prediction.y_hat_upper.to_numpy().astype(np.float64)
    y_hat = df_prediction.y_hat.to_numpy().astype(np.float64)
    y_hat_lower = df_prediction.y_hat_lower.to_numpy().astype(np.float64)
    y = df_prediction[Config.kpi_col_name].to_numpy().astype(np.float64)
    x = df_prediction[Config.date_col_name].to_numpy()

    uncertainty = df_prediction[
        df_prediction.columns[df_prediction.columns.str.contains("dist")]
    ]
    # df_train = df_prediction[df_prediction[Config.date_col_name] < date_split]
    # df_test = df_prediction[df_prediction[Config.date_col_name] >= date_split]

    fig, ax = plt.subplots()

    ax.plot(
        df_prediction.ds.values, uncertainty.values, lw=1, color=c4, alpha=0.005,
    )
    ax.plot(
        x,
        y_hat_upper,
        lw=0.5,
        color=c4,
        label="Upper bounds",
        alpha=0.5,
        linestyle="--",
    )
    ax.plot(x, y_hat, lw=1, color=c3, label="Prediction", linestyle="--")
    ax.plot(x, y, lw=1, color=c2, label="Actual")
    ax.plot(
        x,
        y_hat_lower,
        lw=0.5,
        color=c4,
        label="Lower bounds",
        alpha=0.5,
        linestyle="--",
    )
    ax.axvline(date_split, linestyle="-", label="intervention date", color=c1)
    ax.legend(loc="lower left")
    ax.legend(loc="upper left")
    ax.set_ylabel("KPI")
    ax.set_xlabel("Date")
    fig.autofmt_xdate()
    ax.set(title=title)
    plt.savefig(fname="{}.png".format(title), dpi=300)

    return fig, ax
