import pandas as pd
from config import Config
from feeder import dataframe_split_by_date
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, r2_score)


def performance_report(dataframe, split_on_date):
    train, test = dataframe_split_by_date(dataframe, split_on_date)
    return pd.DataFrame(
        data={
            "Test": [
                explained_variance_score(test[Config.kpi_col_name], test.y_hat),
                mean_absolute_error(test[Config.kpi_col_name], test.y_hat),
                mean_squared_error(test[Config.kpi_col_name], test.y_hat),
                r2_score(test[Config.kpi_col_name], test.y_hat),
            ],
            "Train": [
                explained_variance_score(train[Config.kpi_col_name], train.y_hat),
                mean_absolute_error(train[Config.kpi_col_name], train.y_hat),
                mean_squared_error(train[Config.kpi_col_name], train.y_hat),
                r2_score(train[Config.kpi_col_name], train.y_hat),
            ],
            "Total": [
                explained_variance_score(
                    dataframe[Config.kpi_col_name], dataframe.y_hat
                ),
                mean_absolute_error(dataframe[Config.kpi_col_name], dataframe.y_hat),
                mean_squared_error(dataframe[Config.kpi_col_name], dataframe.y_hat),
                r2_score(dataframe[Config.kpi_col_name], dataframe.y_hat),
            ],
        },
        index=[
            "explained_variance_score",
            "mean_absolute_error",
            "mean_squared_error",
            "r2_score",
        ],
    )
