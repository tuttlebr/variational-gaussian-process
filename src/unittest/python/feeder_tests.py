import unittest

from config import Config
from feeder import (dataframe_cointegration, dataframe_linear_interpolation,
                    dataframe_pivot, dataframe_split_by_date, read_csv)
from pandas.core.frame import DataFrame


class Testing(unittest.TestCase):
    def test_read_csv(self):
        df = read_csv(sample_data)
        self.assertIsInstance(df, DataFrame)

    def test_dataframe_pivot(self):
        df = read_csv(sample_data)
        dfp = dataframe_pivot(df)
        self.assertIsInstance(dfp, DataFrame)

    def test_dataframe_linear_interpolation(self):
        df = read_csv(sample_data)
        dfp = dataframe_pivot(df)
        dfi = dataframe_linear_interpolation(dfp)
        self.assertIsInstance(dfi, DataFrame)

    def test_dataframe_split_by_date(self):
        df = read_csv(sample_data)
        dfp = dataframe_pivot(df)
        dfi = dataframe_linear_interpolation(dfp)
        train_df, eval_df = dataframe_split_by_date(dfi, split_on_date)
        self.assertIsInstance(train_df, DataFrame)
        self.assertIsInstance(eval_df, DataFrame)

    def test_dataframe_cointegration(self):
        df = read_csv(sample_data)
        dfp = dataframe_pivot(df)
        dfi = dataframe_linear_interpolation(dfp)
        train_df, eval_df = dataframe_split_by_date(dfi, split_on_date)
        train_df, eval_df, peer_list = dataframe_cointegration(
            train_df, eval_df, embed_id
        )

        self.assertIsInstance(train_df, DataFrame)
        self.assertIsInstance(eval_df, DataFrame)
        self.assertTrue(embed_id not in peer_list)


global sample_data
global split_on_date
global embed_id
sample_data = "/app/src/unittest/sample_data.csv"
for embed_id, split_on_date in Config.embedding_split.items():
    pass

if __name__ == "__main__":
    unittest.main()
