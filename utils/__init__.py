from collections import deque, OrderedDict
from typing import List, Literal, Optional, Union, Dict, Hashable, Any, Collection
import toml
import logging
import pandas as pd
import numpy as np
import datetime as dt
import torch
import copy

    
class Defaults:
    """singleton class storing the default settings"""
    fundamental_data_cols = (
        # "figi",
        # "year",
        "operating_roic", 
        "normalized_roe", 
        "return_on_asset", 
        "return_com_eqy", 
        "ebit_margin", 
        "fcf_margin_after_oper_lea_pymt", 
        "gross_margin", 
        "eff_tax_rate", 
        "ebitda_margin", 
        "net_debt_to_shrhldr_eqty", 
        "fixed_charge_coverage_ratio", 
        "net_debt_to_ebitda", 
        "acct_rcv_days", 
        "cash_conversion_cycle", 
        "invent_days", 
        "net_income_growth", 
        "sales_rev_turn_growth",
    )
    price_multiples_data_cols = (
        "best_cur_ev_to_ebitda",
        "fcf_yield_with_cur_entp_val",
        "px_last",
        "px_to_book_ratio",
        "px_to_sales_ratio"
    )
    stationary_price_multiples_data_cols = (
        "best_cur_ev_to_ebitda",
        "fcf_yield_with_cur_entp_val",
        "px_to_book_ratio",
        "px_to_sales_ratio"
    )
    nonstationary_price_multiples_data_cols  = {
        "px_last"
    }
    dtype = torch.float32
    padding_val = torch.tensor(-1e10, dtype=dtype)
    padding_dims = (1, 0, 0, 0) # default is to pad on the left
    window_size_weeks = 260
    window_size_years = 10
    minmax_name_mapper = {"min": "BottomValue", "max": "TopValue"}
    FREQ = Literal["D", "W", "M", "Q", "H", "Y"]

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Defaults, cls).__new__(cls)
            return cls.instance


def get_future_returns(ref_date: dt.date, price_df: pd.DataFrame):
    df = price_df.copy(deep=True)
    df.index = df.index.astype("datetime64[ns]")
    base_date_ind = df.index.to_series().lt(np.datetime64(ref_date)).sum()
    _3m_ind = df.index.to_series().lt(np.datetime64(ref_date + dt.timedelta(days=90))).sum()
    _6m_ind = df.index.to_series().lt(np.datetime64(ref_date + dt.timedelta(days=180))).sum()
    _1yr_ind = df.index.to_series().lt(np.datetime64(ref_date + dt.timedelta(days=365))).sum()
    _3yr_ind = df.index.to_series().lt(np.datetime64(ref_date + dt.timedelta(days=365 * 3))).sum()
    base_price = df.iloc[base_date_ind]
    _3m_returns= (df.iloc[ _3m_ind - 5 : _3m_ind + 5] / base_price).mean()
    # _3m_returns_std = (df.iloc[ _3m_ind - 5 : _3m_ind + 5] / base_price).std()
    _6m_returns = (df.iloc[ _6m_ind - 10 : _6m_ind + 10] / base_price).mean()
    _1yr_returns = (df.iloc[ _1yr_ind - 20 : _1yr_ind + 20] / base_price).mean()
    _3yr_returns = (df.iloc[ _3yr_ind - 60 : _3yr_ind + 60] / base_price).mean()
    res = pd.concat([_3m_returns, _6m_returns, _1yr_returns, _3yr_returns], axis=1)
    res.columns = ["3m", "6m", "1yr", "3yr"]
    return res

def fillna(input: torch.tensor, fillval: torch.tensor, inplace: bool=False) -> Optional[torch.tensor]:
    mask = torch.isnan(input)
    if inplace:
        input[mask] = fillval
    else:
        new_input = copy.deepcopy(input)
        new_input[mask] = fillval
        return new_input

def differencer(
        df: pd.DataFrame,
        skip_columns: Collection[Any],
        orient: Literal[1, 0, "columns", "index"]=0,
        ) -> pd.DataFrame:
    """
    :param skip_column: skip columns
    :param orient: 
        if orient = 0/'index' - timeseries along the index axis
        if orient = 1/'columns' - timeseries along the columns
    """
    if orient in (0, "index"):
        df_ = copy.deepcopy(df)
    elif orient in (1, "columns"):
        df_ = df.T
    else: raise ValueError(f"orient {orient} not recognized!")
    df__ = df_.pct_change(periods=-1, fill_method=None)\
        .replace(float("inf"), np.nan)\
            .replace(float("-inf"), np.nan)
    df__ = np.log(df__)
    if skip_columns is not None:
        for col in skip_columns:
            df__[col] = df_[col]
    if orient in (0, "index"):
        return df__
    else:
        return df__.T

class Scaler:

    def __init__(self, 
        df: pd.DataFrame, 
        orient: Literal[1, 0, "columns", "index"]=0,
        ):
        """
        :param df: input dataframe
        :param orient: orientation of the data. 
            0 or "index" - fields stored in the columns, entries along the index
            1 or "columns" - field stored in the index, entries along the columns
        """
        if orient in (0, "index"):
            self.df = copy.deepcopy(df)
        elif orient in (1, "columns"):
            self.df = df.T
        else: raise ValueError(f"orient {orient} not recognized!")

    @staticmethod
    def winsorize(series: pd.Series, min_val: Any, max_val: Any):
        series_ = copy.deepcopy(series)
        series_ = series_.clip(lower=min_val, upper=max_val)
        return series_

    def minmax_scaler(self,
            minmax_df: pd.DataFrame, 
            skip_columns: Optional[Collection[Any]]=None,
            minmax_name_mapper: Dict[Hashable, str]=Defaults.minmax_name_mapper
            ) -> pd.DataFrame:
        """winsorize and scale the input data with minmax scaler
        
        
        :param minmax_df: dataframe with the min/max information, with the index as the 
            dataset's columns (fields), and the columns as the min/max
        
        :param minmax_name_mapper: maps the columns names in minmax_df to min/max
        """

        min_fld, max_fld = minmax_name_mapper.get("min"), minmax_name_mapper.get("max")
        df_ = copy.deepcopy(self.df)
        for field, sres in df_.items():
            if skip_columns is not None:
                if field in skip_columns:
                    pass
                else:
                    min_val = minmax_df.loc[field, min_fld]
                    max_val = minmax_df.loc[field, max_fld]
                    df_[field] = self.winsorize(sres, min_val, max_val)
                    df_[field] = df_[field].apply(lambda x: (x - min_val) / (max_val - min_val)).values
            else:
                min_val = minmax_df.loc[field, min_fld]
                max_val = minmax_df.loc[field, max_fld]
                df_[field] = self.winsorize(sres, min_val, max_val)
                df_[field] = df_[field].apply(lambda x: (x - min_val) / (max_val - min_val)).values
        return df_

    @classmethod
    def scale_minmax(cls,
                df: pd.DataFrame,
                minmax_df: pd.DataFrame,
                skip_columns: Optional[Collection[Any]]=None,
                orient: Literal[1, 0, "columns", "index"]=0,
                minmax_name_mapper: Dict[Hashable, str]=Defaults.minmax_name_mapper
                ) -> pd.DataFrame:
        """classmethod version of `minmax_scaler`"""
        return cls(df, orient).minmax_scaler(minmax_df, skip_columns, minmax_name_mapper)

    @staticmethod
    def minmax_unscaler(
                df,
                minmax_df: pd.DataFrame, 
                skip_columns: Optional[Collection[Any]]=None,
                minmax_name_mapper: Dict[Hashable, str]=Defaults.minmax_name_mapper
                ) -> pd.DataFrame:
        """restore the scaled data into its orignal form"""
        df_ = copy.deepcopy(df)
        min_fld, max_fld = minmax_name_mapper.get("min"), minmax_name_mapper.get("max")
        for field, sres in df_.items():
            if skip_columns is not None:
                if field not in skip_columns:
                    min_val = minmax_df.loc[field, min_fld]
                    max_val = minmax_df.loc[field, max_fld]
                    df_[field] = sres.apply(lambda x: x * (max_val - min_val) + min_val)
                else:
                    pass
            else:
                min_val = minmax_df.loc[field, min_fld]
                max_val = minmax_df.loc[field, max_fld]
                df_[field] = sres.apply(lambda x: x * (max_val - min_val) + min_val)

        return df_
