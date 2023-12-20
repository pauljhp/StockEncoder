import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from utils.database import SQLDatabase
from typing import Tuple, Dict, Any, Generator


class Defaults:
    """singleton class storing the default settings"""
    fundamental_data_cols = (
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
    dtype = torch.float32
    padding_val = torch.tensor(-1e10, dtype=dtype)
    padding_dims = (1, 0, 0, 0) # default is to pad on the left

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Defaults, cls).__new__(cls)
            return cls.instance

DEFAULTS = Defaults()

class FundamentalDataset(Dataset):
    data = SQLDatabase.to_pandas(f"SELECT {', '.join(DEFAULTS.fundamental_data_cols)} FROM fundamental_data_stock_encoder ORDER BY figi ASC, year DESC")
    data = data.set_index(["figi", "year"]).sort_index()

    def __init__(
            self, 
            window_size: int=10, 
            dtype: torch.dtype=DEFAULTS.dtype,
            padding_val: torch.tensor=DEFAULTS.padding_val):
        super().__init__()
        self.figi_list = self.data.index.get_level_values("figi").unique()
        self.year_list = self.data.index.get_level_values("year").unique()
        self.min_year = self.year_list.min()
        self.max_year = self.year_list.max()
        self.window_size = window_size
        self.dtype = dtype
        self.padding_val = padding_val
    
    def __getitem__(self, idx: Tuple[int, int]) -> torch.tensor:
        """dunder method for indexing items. Note idx should a tuple.
        :param idx: takes tuples of (idx_of_company, idx_of_year)
        """
        figi_idx, year_idx = idx
        figi = self.figi_list[figi_idx]
        year = self.year_list[year_idx]
        if year >= self.min_year + self.window_size:
            slice = self.data.loc[figi].loc[year - self.window_size: year]
        else:
            slice = self.data.loc[figi].loc[: year]
            padding_size = self.min_year - (year - self.window_size)
            slice = F.pad(slice, pad=(padding_size, 0, 0, 0), mode="constant", value=self.padding_val)

        slice = torch.tensor(slice, dtype=self.dtype)
        return slice

    def __len__(self) -> Tuple[int, int]:
        figi_len = len(self.figi_list)
        year_len = len(self.year_list) # period earlier will be padded
        return (figi_len, year_len)
    
    def __iter__(self) -> Generator[torch.tensor]:
        figi_len, year_yen = self.__len__()
        for i in range(figi_len)



class PriceDataset(Dataset):
    """Timeseries containing the historical prices and multiples for securities.
    Iterating this dataset returns the differenced price series and winsorized
    multiples series of specified window sizes, as well as the target 
    (look-ahead) return of that particular security, as specified time horizon.
    """
    pass

class MacroDataset(Dataset):
    pass