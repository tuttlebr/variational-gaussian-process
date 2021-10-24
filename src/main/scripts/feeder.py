from logging import basicConfig, info

basicConfig(
    format="%(asctime)s %(message)s", level="INFO", datefmt="%Y-%m-%d %H:%M:%S",
)
import multiprocessing as mp

import numpy as np
import pandas as pd
from cointegration import cointegration
from config import Config


def standardize_dates(s):
    dates = {date: pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)


# 1
def read_csv(
    data,
    values=Config.kpi_col_name,
    date_index=Config.date_col_name,
    embed=Config.embed_id_col_name,
    min_samples=Config.kernel_size,
):

    dataframe = pd.read_csv(data, usecols=[date_index, embed, values])

    dataframe[date_index] = standardize_dates(dataframe[date_index])

    try:
        dataframe[embed] = dataframe[embed].astype(int)
    except:
        dataframe[embed] = dataframe[embed].astype(str)

    dataframe[values] = dataframe[values].astype(
        np.float64
    )  # keep this for numerical stability during

    dataframe.columns = dataframe.columns.astype(str)

    if (
        len(dataframe.groupby(embed).filter(lambda x: x[embed].count() > min_samples))
        == 0
    ):
        info(
            "kernel_size set to large, try setting it lower than : {}".format(
                min_samples
            )
        )

    return dataframe.groupby(embed).filter(lambda x: x[embed].count() > min_samples)


# 2
def dataframe_pivot(dataframe):
    dataframe = (
        pd.pivot_table(
            dataframe,
            values=Config.kpi_col_name,
            index=Config.date_col_name,
            columns=[Config.embed_id_col_name],
        )
        .reset_index(drop=False)
        .sort_values(by=Config.date_col_name, ascending=True)
    )

    dataframe.columns = dataframe.columns.astype("object")

    return dataframe


# 3
def dataframe_linear_interpolation(dataframe):
    dataframe = [
        dataframe[col].interpolate(method="linear") for col in dataframe.columns
    ]
    dataframe = pd.concat(dataframe, axis=1, keys=[s.name for s in dataframe])

    dataframe.dropna(inplace=True, axis=1)

    dataframe.columns = dataframe.columns.astype("object")

    return dataframe


# 4
def dataframe_split_by_date(dataframe, split_on_date, index=Config.date_col_name):

    train_df = dataframe[dataframe[index] < split_on_date]

    eval_df = dataframe[dataframe[index] >= split_on_date]

    return train_df, eval_df


# 5
def dataframe_cointegration(train_dataframe, eval_dataframe, target_embed_id_id):

    train_dataframe_ = train_dataframe.rename(
        columns={str(target_embed_id_id): Config.kpi_col_name}
    )
    train_dataframe_coint = train_dataframe_.tail(365)

    eval_dataframe_ = eval_dataframe.rename(
        columns={str(target_embed_id_id): Config.kpi_col_name}
    )

    if not Config.manual_peers:
        info("Running automatic peer selection...")
        results = []
        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap_async(
            cointegration,
            [
                (
                    row,
                    train_dataframe_coint[Config.kpi_col_name].values,
                    train_dataframe_coint[row].values,
                )
                for i, row in enumerate(
                    train_dataframe_coint.drop(
                        labels=[Config.date_col_name, Config.kpi_col_name], axis=1,
                    )
                )
            ],
        ).get()
        pool.close()

        results.sort(reverse=False, key=lambda x: x[1])

        # do not select peers also in test.
        keys1 = [i[0] for i in results]
        keys2 = Config.embedding_split.keys()
        results = list(keys1 - keys2)

        coint_vars = [i for i in results[: Config.top_n_peer_count]]

        info("Automatically Selected Peers: {}".format(coint_vars))

        peer_list = coint_vars.copy()

        coint_vars.append(Config.kpi_col_name)

        coint_vars.append(Config.date_col_name)

    else:
        coint_vars = list(Config.manual_peers)

        info("User Selected Peers: {}".format(coint_vars))

        peer_list = coint_vars.copy()

        coint_vars.append(Config.kpi_col_name)

        coint_vars.append(Config.date_col_name)

    return train_dataframe_[coint_vars], eval_dataframe_[coint_vars], peer_list


# 6
def retrieve_inputs(dataframe, peer_list):
    x = dataframe[peer_list].values

    y = dataframe[Config.kpi_col_name].values.astype(np.float64)

    return x, y
