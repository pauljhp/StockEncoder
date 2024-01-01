from collections import deque, OrderedDict
from typing import List, Literal, Optional, Union, Dict, Hashable, Any
import toml
import logging
import pandas as pd
import numpy as np
import datetime as dt
import torch
import copy

    

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
    dtype = torch.float32
    padding_val = torch.tensor(-1e10, dtype=dtype)
    padding_dims = (1, 0, 0, 0) # default is to pad on the left
    window_size_weeks = 260
    window_size_years = 10

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Defaults, cls).__new__(cls)
            return cls.instance