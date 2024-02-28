import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from utils.database import SQLDatabase
import utils
from typing import (
    Tuple, Dict, Any, 
    Generator, Collection, 
    Optional, Union
    )
import datetime as dt
import itertools


DEFAULTS = utils.Defaults
FIGI_LIST = SQLDatabase.to_pandas(
    """SELECT DISTINCT figi FROM price_multiples_stock_encoder
    INTERSECT
    SELECT DISTINCT figi FROM fundamental_data_stock_encoder;
    """
    )
PERIOD_LIST = SQLDatabase.to_pandas(
    """SELECT DISTINCT period, DATEPART(year, period) as year
    FROM price_multiples_stock_encoder
    WHERE DATEPART(year, period) > ((
		SELECT  min(year) FROM fundamental_data_stock_encoder
		) + 2) AND
        DATEPART(year, period) <= (
            SELECT max(year) FROM fundamental_data_stock_encoder
        )
    ORDER BY period ASC;
    """
    )

class FundamentalDataset(Dataset):
    """annual dataset containing key financial data of ~10k companies globally"""
    table_name = "fundamental_data_stock_encoder"
    # data = SQLDatabase.to_pandas(f"SELECT {', '.join(DEFAULTS.fundamental_data_cols)} FROM fundamental_data_stock_encoder ORDER BY figi ASC, year DESC")
    # print(data.head())
    # data = data.set_index(["figi", "year"]).sort_index()
    # data = data.ffill(axis="columns") # forward fill missing data from the previous year
    figi_list = FIGI_LIST
    period_list = PERIOD_LIST
    year_list = SQLDatabase.to_pandas(
        "SELECT DISTINCT year FROM fundamental_data_stock_encoder ORDER BY year ASC"
        )
    stats_dict = {'BottomValue': {'operating_roic': 0.2335, 
                'normalized_roe': -40.5183, 
                'return_on_asset': -2.4256, 
                'return_com_eqy': -7.7047, 
                'ebit_margin': 0.2171, 
                'fcf_margin_after_oper_lea_pymt': -119.5501, 
                'gross_margin': 3.6507, 
                'eff_tax_rate': 0.0, 
                'ebitda_margin': 0.9084, 
                'net_debt_to_shrhldr_eqty': -117.629, 
                'fixed_charge_coverage_ratio': 0.586, 
                'net_debt_to_ebitda': -11.3862, 
                'acct_rcv_days': 1.0873, 
                'cash_conversion_cycle': -984.7053, 
                'invent_days': 0.4766, 
                'net_income_growth': -1.9929175325229869, 
                'sales_rev_turn_growth': -0.3562543997133397
                }, 
              'TopValue': {'operating_roic': 54.7816, 
                'normalized_roe': 66.2741, 
                'return_on_asset': 32.5935, 
                'return_com_eqy': 64.9997, 
                'ebit_margin': 60.685, 
                'fcf_margin_after_oper_lea_pymt': 67.3786, 
                'gross_margin': 95.5145, 
                'eff_tax_rate': 111.8484, 
                'ebitda_margin': 78.828, 
                'net_debt_to_shrhldr_eqty': 671.3544, 
                'fixed_charge_coverage_ratio': 856.8845, 
                'net_debt_to_ebitda': 16.7727, 
                'acct_rcv_days': 251.1219, 
                'cash_conversion_cycle': 723.4013, 
                'invent_days': 798.0315, 
                'net_income_growth': 2.70071738865442, 
                'sales_rev_turn_growth': 1.3398632010585674
                }
            }
    stats = pd.DataFrame.from_dict(stats_dict, orient="columns")

    def get_winsorization_stats(self):
        stats_query = """WITH Sample AS (
                SELECT * FROM fundamental_data_stock_encoder TABLESAMPLE (2 PERCENT) REPEATABLE (123)
            ),
            Percentiles AS (
                SELECT 
                    operating_roic, 
                    normalized_roe, 
                    return_on_asset, 
                    return_com_eqy, 
                    ebit_margin, 
                    fcf_margin_after_oper_lea_pymt, 
                    gross_margin, 
                    eff_tax_rate, 
                    ebitda_margin, 
                    net_debt_to_shrhldr_eqty, 
                    fixed_charge_coverage_ratio, 
                    net_debt_to_ebitda, 
                    acct_rcv_days, 
                    cash_conversion_cycle, 
                    invent_days, 
                    net_income_growth, 
                    sales_rev_turn_growth,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY operating_roic) OVER () AS BottomPercentile1,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY operating_roic) OVER () AS TopPercentile1,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY normalized_roe) OVER () AS BottomPercentile2,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY normalized_roe) OVER () AS TopPercentile2,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY return_on_asset) OVER () AS BottomPercentile3,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY return_on_asset) OVER () AS TopPercentile3,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY return_com_eqy) OVER () AS BottomPercentile4,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY return_com_eqy) OVER () AS TopPercentile4,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY ebit_margin) OVER () AS BottomPercentile5,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY ebit_margin) OVER () AS TopPercentile5,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY fcf_margin_after_oper_lea_pymt) OVER () AS BottomPercentile6,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY fcf_margin_after_oper_lea_pymt) OVER () AS TopPercentile6,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY gross_margin) OVER () AS BottomPercentile7,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY gross_margin) OVER () AS TopPercentile7,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY eff_tax_rate) OVER () AS BottomPercentile8,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY eff_tax_rate) OVER () AS TopPercentile8,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY ebitda_margin) OVER () AS BottomPercentile9,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY ebitda_margin) OVER () AS TopPercentile9,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY net_debt_to_shrhldr_eqty) OVER () AS BottomPercentile10,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY net_debt_to_shrhldr_eqty) OVER () AS TopPercentile10,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY fixed_charge_coverage_ratio) OVER () AS BottomPercentile11,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY fixed_charge_coverage_ratio) OVER () AS TopPercentile11,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY net_debt_to_ebitda) OVER () AS BottomPercentile12,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY net_debt_to_ebitda) OVER () AS TopPercentile12,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY acct_rcv_days) OVER () AS BottomPercentile13,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY acct_rcv_days) OVER () AS TopPercentile13,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY cash_conversion_cycle) OVER () AS BottomPercentile14,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY cash_conversion_cycle) OVER () AS TopPercentile14,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY invent_days) OVER () AS BottomPercentile15,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY invent_days) OVER () AS TopPercentile15,
                    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY net_income_growth) OVER () AS BottomPercentile16,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY net_income_growth) OVER () AS TopPercentile16,
                    PERCENTILE_CONT(0.02) WITHIN GROUP (ORDER BY sales_rev_turn_growth) OVER () AS BottomPercentile17,
                    PERCENTILE_CONT(0.98) WITHIN GROUP (ORDER BY sales_rev_turn_growth) OVER () AS TopPercentile17
                FROM Sample
            )
            SELECT 
                'BottomValue' as ValueType,
                MIN(operating_roic) as operating_roic,
                MIN(normalized_roe) as normalized_roe,
                MIN(return_on_asset) as return_on_asset,
                MIN(return_com_eqy) as return_com_eqy,
                MIN(ebit_margin) as ebit_margin,
                MIN(fcf_margin_after_oper_lea_pymt) as fcf_margin_after_oper_lea_pymt,
                MIN(gross_margin) as gross_margin,
                MIN(eff_tax_rate) as eff_tax_rate,
                MIN(ebitda_margin) as ebitda_margin,
                MIN(net_debt_to_shrhldr_eqty) as net_debt_to_shrhldr_eqty,
                MIN(fixed_charge_coverage_ratio) as fixed_charge_coverage_ratio,
                MIN(net_debt_to_ebitda) as net_debt_to_ebitda,
                MIN(acct_rcv_days) as acct_rcv_days,
                MIN(cash_conversion_cycle) as cash_conversion_cycle,
                MIN(invent_days) as invent_days,
                MIN(net_income_growth) as net_income_growth,
                MIN(sales_rev_turn_growth) as sales_rev_turn_growth
            FROM Percentiles
            WHERE operating_roic >= BottomPercentile1 AND
                normalized_roe >= BottomPercentile2 AND
                return_on_asset >= BottomPercentile3 AND
                return_com_eqy >= BottomPercentile4 AND
                ebit_margin >= BottomPercentile5 AND
                fcf_margin_after_oper_lea_pymt >= BottomPercentile6 AND
                gross_margin >= BottomPercentile7 AND
                eff_tax_rate >= BottomPercentile8 AND
                ebitda_margin >= BottomPercentile9 AND
                net_debt_to_shrhldr_eqty >= BottomPercentile10 AND
                fixed_charge_coverage_ratio >= BottomPercentile11 AND
                net_debt_to_ebitda >= BottomPercentile12 AND
                acct_rcv_days >= BottomPercentile13 AND
                cash_conversion_cycle >= BottomPercentile14 AND
                invent_days >= BottomPercentile15 AND
                net_income_growth >= BottomPercentile16 AND
                sales_rev_turn_growth >= BottomPercentile17
            UNION ALL
            SELECT 
                'TopValue' as ValueType,
                MAX(operating_roic) as operating_roic,
                MAX(normalized_roe) as normalized_roe,
                MAX(return_on_asset) as return_on_asset,
                MAX(return_com_eqy) as return_com_eqy,
                MAX(ebit_margin) as ebit_margin,
                MAX(fcf_margin_after_oper_lea_pymt) as fcf_margin_after_oper_lea_pymt,
                MAX(gross_margin) as gross_margin,
                MAX(eff_tax_rate) as eff_tax_rate,
                MAX(ebitda_margin) as ebitda_margin,
                MAX(net_debt_to_shrhldr_eqty) as net_debt_to_shrhldr_eqty,
                MAX(fixed_charge_coverage_ratio) as fixed_charge_coverage_ratio,
                MAX(net_debt_to_ebitda) as net_debt_to_ebitda,
                MAX(acct_rcv_days) as acct_rcv_days,
                MAX(cash_conversion_cycle) as cash_conversion_cycle,
                MAX(invent_days) as invent_days,
                MAX(net_income_growth) as net_income_growth,
                MAX(sales_rev_turn_growth) as sales_rev_turn_growth
            FROM Percentiles
            WHERE operating_roic <= TopPercentile1 AND
                normalized_roe <= TopPercentile2 AND
                return_on_asset <= TopPercentile3 AND
                return_com_eqy <= TopPercentile4 AND
                ebit_margin <= TopPercentile5 AND
                fcf_margin_after_oper_lea_pymt <= TopPercentile6 AND
                gross_margin <= TopPercentile7 AND
                eff_tax_rate <= TopPercentile8 AND
                ebitda_margin <= TopPercentile9 AND
                net_debt_to_shrhldr_eqty <= TopPercentile10 AND
                fixed_charge_coverage_ratio <= TopPercentile11 AND
                net_debt_to_ebitda <= TopPercentile12 AND
                acct_rcv_days <= TopPercentile13 AND
                cash_conversion_cycle <= TopPercentile14 AND
                invent_days <= TopPercentile15 AND
                net_income_growth <= TopPercentile16 AND
                sales_rev_turn_growth <= TopPercentile17;
            ---
            --Stats
            WITH sampled AS (
                SELECT * FROM price_multiples_stock_encoder TABLESAMPLE (1 PERCENT) REPEATABLE (123)
            ), temp_median AS (
                SELECT 
                    'median' as statistic,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY best_cur_ev_to_ebitda) OVER() as best_cur_ev_to_ebitda,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fcf_yield_with_cur_entp_val) OVER() as fcf_yield_with_cur_entp_val,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY px_to_book_ratio) OVER() as px_to_book_ratio,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY px_to_sales_ratio) OVER() as px_to_sales_ratio
                FROM sampled
            ), Percentiles AS (
                SELECT 
                    best_cur_ev_to_ebitda, 
                    fcf_yield_with_cur_entp_val,
                    px_to_book_ratio,
                    px_to_sales_ratio,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY best_cur_ev_to_ebitda) OVER () AS BottomPercentile1,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY best_cur_ev_to_ebitda) OVER () AS TopPercentile1,
                    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY fcf_yield_with_cur_entp_val) OVER () AS BottomPercentile2,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY fcf_yield_with_cur_entp_val) OVER () AS TopPercentile2,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY px_to_book_ratio) OVER () AS BottomPercentile3,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY px_to_book_ratio) OVER () AS TopPercentile3,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY px_to_sales_ratio) OVER () AS BottomPercentile4,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY px_to_sales_ratio) OVER () AS TopPercentile4
                FROM sampled
            )
            SELECT 
                'mean' as statistic,
                AVG(best_cur_ev_to_ebitda) as best_cur_ev_to_ebitda,
                AVG(fcf_yield_with_cur_entp_val) as fcf_yield_with_cur_entp_val,
                AVG(px_to_book_ratio) as px_to_book_ratio,
                AVG(px_to_sales_ratio) as px_to_sales_ratio
            FROM sampled
            UNION ALL
            SELECT 
                'stddev' as statistic,
                STDEV(best_cur_ev_to_ebitda) as best_cur_ev_to_ebitda,
                STDEV(fcf_yield_with_cur_entp_val) as fcf_yield_with_cur_entp_val,
                STDEV(px_to_book_ratio) as px_to_book_ratio,
                STDEV(px_to_sales_ratio) as px_to_sales_ratio
            FROM sampled
            UNION ALL
            SELECT 
                statistic,
                MAX(best_cur_ev_to_ebitda) as best_cur_ev_to_ebitda,
                MAX(fcf_yield_with_cur_entp_val) as fcf_yield_with_cur_entp_val,
                MAX(px_to_book_ratio) as px_to_book_ratio,
                MAX(px_to_sales_ratio) as px_to_sales_ratio
            FROM temp_median
            GROUP BY statistic
            UNION ALL
            SELECT 
                'BottomValue' as statistic,
                MIN(best_cur_ev_to_ebitda) as best_cur_ev_to_ebitda,
                MIN(fcf_yield_with_cur_entp_val) as fcf_yield_with_cur_entp_val,
                MIN(px_to_book_ratio) as px_to_book_ratio,
                MIN(px_to_sales_ratio) as px_to_sales_ratio
                FROM Percentiles
                WHERE best_cur_ev_to_ebitda >= BottomPercentile1 AND
                    best_cur_ev_to_ebitda >= BottomPercentile2 AND
                    best_cur_ev_to_ebitda >= BottomPercentile3 AND
                    best_cur_ev_to_ebitda >= BottomPercentile4
            UNION ALL
            SELECT 
                'TopValue' as statistic,
                MAX(best_cur_ev_to_ebitda) as best_cur_ev_to_ebitda,
                MAX(fcf_yield_with_cur_entp_val) as fcf_yield_with_cur_entp_val,
                MAX(px_to_book_ratio) as px_to_book_ratio,
                MAX(px_to_sales_ratio) as px_to_sales_ratio
                FROM Percentiles
                WHERE best_cur_ev_to_ebitda <= TopPercentile1 AND 
                    fcf_yield_with_cur_entp_val <= TopPercentile2 AND
                    px_to_book_ratio <= TopPercentile3 AND
                    px_to_sales_ratio <= TopPercentile4"""
        stats = SQLDatabase.to_pandas(stats_query)
        return stats

    def __init__(
            self, 
            window_size: int=DEFAULTS.window_size_years, 
            dtype: torch.dtype=DEFAULTS.dtype,
            padding_val: torch.tensor=DEFAULTS.padding_val,
            freq: DEFAULTS.FREQ="Y"):
        super().__init__()
        # self.figi_list = self.data.index.get_level_values("figi").unique()
        # self.year_list = self.data.index.get_level_values("year").unique()
        self.min_year = self.year_list.year.min()
        self.max_year = self.year_list.year.max()
        self.window_size = window_size
        self.dtype = dtype
        self.padding_val = padding_val
        # year_range = range(self.min_year + 2, self.max_year)
        periods = pd.Index(self.period_list.period)\
                .astype("datetime64[ns]")\
                    .to_period(freq=freq)\
                        .unique()\
                            .sort_values()\
                                .astype("datetime64[ns]")\
                                    .values
        self.index = list(
            itertools.product(
                tuple(range(len(self.figi_list))), 
                tuple(periods)
                )
            )
        
    def get_index(self, idx: int) -> Tuple[int, Union[dt.date, np.datetime64]]:
        return self.index[idx]
    
    def __getitem__(
            self,
            idx: int
            ) -> Tuple[torch.tensor]:
        """dunder method for indexing items. Note idx should a tuple.
        :param idx: takes tuples of (idx_of_company, period (dt.datetime))
            Note: periods earlier than March 31st will only access data 2 years ago
        """
        figi_idx, period = self.get_index(idx)
        
        if isinstance(period, dt.date):
            pass
        elif isinstance(period, np.datetime64):
            period = pd.to_datetime(period).date()
        else:
            raise TypeError(f"the second item in idx must be either datetime or np.datetime64, got {type(period)}!")
        
        figi = self.figi_list.loc[figi_idx, "figi"]
        if period.year < self.min_year + 2 or period.year > self.max_year:
            raise ValueError(f"{period} outside of valid range! Must be between year {self.min_year + 2} and {self.max_year}")
        if period <= dt.date(period.year, 3, 31): # hard-code annual report date to Mar 31st, until availability of more granular data
            year = period.year - 2
        else:
            year = period.year - 1

        if year >= self.min_year + self.window_size:
            begin_year = year - self.window_size + 1
            query = f"SELECT {', '.join(DEFAULTS.fundamental_data_cols)} FROM {self.table_name} WHERE (year between {begin_year} AND {year}) AND (figi = '{figi}')"
            slice = SQLDatabase.to_pandas(query).astype(float).ffill()
            slice = utils.Scaler.scale_minmax(slice, self.stats)
            slice = torch.tensor(slice.values, dtype=self.dtype)
            padding_mask = torch.tensor(
                [False for _ in range(self.window_size)],
                dtype=torch.bool
            )
        else:
            query = f"SELECT {', '.join(DEFAULTS.fundamental_data_cols)} FROM {self.table_name} WHERE (year <= {year}) AND (figi = '{figi}')"
            slice = SQLDatabase.to_pandas(query).astype(float).ffill()
            slice = utils.Scaler.scale_minmax(slice, self.stats)
            slice = torch.tensor(slice.values, dtype=self.dtype)
            padding_size = self.min_year - (year - self.window_size + 1)
            slice = F.pad(slice, pad=(0, 0, padding_size, 0), mode="constant", value=self.padding_val)
            padding_mask = torch.tensor(
                [True for _ in range(padding_size)] \
                + [False for _ in range(self.window_size - padding_size)],
                dtype=torch.bool
            )

        return utils.fillna(slice, self.padding_val), padding_mask # fill na with the padding as all these will be ignored during training

    def __len__(self) -> int:
        figi_len, year_len = self.get_len()
        return (figi_len * year_len)
    

    def get_len(self) -> Tuple[int, int]:
        figi_len = len(self.figi_list)
        year_len = len(self.year_list) # period earlier will be padded
        return (figi_len, year_len)
    
    def __iter__(self) -> Generator[torch.tensor, None, None]:
        figi_len, year_len = self.get_len()
        for i in range(figi_len):
            for j in self.year_list.year.values[2:]:
                yield self.__getitem__((i, dt.date(j, 6, 30)))



class PriceDataset(Dataset):
    """Timeseries (weekly) containing the historical prices and multiples for 
    securities.
    Iterating this dataset returns the differenced price series and winsorized
    multiples series of specified window sizes, as well as the target 
    (look-ahead) return of that particular security, as specified time horizon.
    """
    table_name = "price_multiples_stock_encoder"
    # data = SQLDatabase.to_pandas(f"SELECT {', '.join(DEFAULTS.fundamental_data_cols)} FROM fundamental_data_stock_encoder ORDER BY figi ASC, year DESC")
    # print(data.head())
    # data = data.set_index(["figi", "year"]).sort_index()
    # data = data.ffill(axis="columns") # forward fill missing data from the previous year
    figi_list = FIGI_LIST
    period_list = PERIOD_LIST

    stats_dict = {
        "TopValue": {
            "best_cur_ev_to_ebitda": 1.915, 
            "fcf_yield_with_cur_entp_val": -34.78195,
            "px_to_book_ratio": 0.1693,
            "px_to_sales_ratio": 0.0779},
        "BottomValue": {
            "best_cur_ev_to_ebitda": 181.86475,	
            "fcf_yield_with_cur_entp_val": 16.01975,
            "px_to_book_ratio": 26.5518,
            "px_to_sales_ratio": 30.341475
            }
        }
    stats = pd.DataFrame.from_dict(stats_dict, orient="columns")

    def get_winsorization_stats(self) -> pd.DataFrame:
        query = """WITH Sample AS (
                SELECT * FROM price_multiples_stock_encoder TABLESAMPLE (1 PERCENT) REPEATABLE (123)
            ),
            Percentiles AS (
                SELECT 
                    best_cur_ev_to_ebitda, 
                    fcf_yield_with_cur_entp_val,
                    px_to_book_ratio,
                    px_to_sales_ratio,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY best_cur_ev_to_ebitda) OVER () AS BottomPercentile1,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY best_cur_ev_to_ebitda) OVER () AS TopPercentile1,
                    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY fcf_yield_with_cur_entp_val) OVER () AS BottomPercentile2,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY fcf_yield_with_cur_entp_val) OVER () AS TopPercentile2,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY px_to_book_ratio) OVER () AS BottomPercentile3,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY px_to_book_ratio) OVER () AS TopPercentile3,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY px_to_sales_ratio) OVER () AS BottomPercentile4,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY px_to_sales_ratio) OVER () AS TopPercentile4
                FROM Sample
            )
            SELECT 
                'BottomValue' as ValueType,
                MIN(best_cur_ev_to_ebitda) as best_cur_ev_to_ebitda,
                MIN(fcf_yield_with_cur_entp_val) as fcf_yield_with_cur_entp_val,
                MIN(px_to_book_ratio) as px_to_book_ratio,
                MIN(px_to_sales_ratio) as px_to_sales_ratio
            FROM Percentiles
            WHERE best_cur_ev_to_ebitda >= BottomPercentile1 AND
                best_cur_ev_to_ebitda >= BottomPercentile2 AND
                best_cur_ev_to_ebitda >= BottomPercentile3 AND
                best_cur_ev_to_ebitda >= BottomPercentile4
            UNION ALL
            SELECT 
                'TopValue' as ValueType,
                MAX(best_cur_ev_to_ebitda) as best_cur_ev_to_ebitda,
                MAX(fcf_yield_with_cur_entp_val) as fcf_yield_with_cur_entp_val,
                MAX(px_to_book_ratio) as px_to_book_ratio,
                MAX(px_to_sales_ratio) as px_to_sales_ratio
            FROM Percentiles
            WHERE best_cur_ev_to_ebitda <= TopPercentile1 AND 
                fcf_yield_with_cur_entp_val <= TopPercentile2 AND
                px_to_book_ratio <= TopPercentile3 AND
                px_to_sales_ratio <= TopPercentile4;
            """
            
        winsor_stats = SQLDatabase.to_pandas(query).set_index("ValueType")
        return winsor_stats

    def __init__(
            self, 
            window_size: int=DEFAULTS.window_size_weeks, # ~5 years 
            dtype: torch.dtype=DEFAULTS.dtype,
            padding_val: torch.tensor=DEFAULTS.padding_val,
            stationary_cols: Optional[Collection[Any]]=DEFAULTS.stationary_price_multiples_data_cols,
            nonstationary_cols: Optional[Collection[Any]]=DEFAULTS.nonstationary_price_multiples_data_cols,
            # winsorize: bool=False
            ):
        super().__init__()
        # self.figi_list = self.data.index.get_level_values("figi").unique()
        # self.year_list = self.data.index.get_level_values("year").unique()
        self.min_period = self.period_list.period.min()
        self.max_period = self.period_list.period.max()
        self.window_size = window_size
        self.dtype = dtype
        self.padding_val = padding_val
        self.stationary_cols = stationary_cols
        self.nonstationary_cols = nonstationary_cols
        self.index = list(
            itertools.product(
                tuple(range(len(self.figi_list))), 
                tuple(self.period_list.period)
                )
            )
        
    def get_index(self, idx: int) -> Tuple[int, Union[dt.date, np.datetime64]]:
        return self.index[idx]
    

    def process_timeseries(self, df: pd.DataFrame):
        """difference non-static series and scale static series"""
        differenced = utils.differencer(df, skip_columns=self.stationary_cols, orient=0)
        scaled = utils.Scaler.scale_minmax(differenced, minmax_df=self.stats, skip_columns=self.nonstationary_cols)
        return scaled

    def __getitem__(
            self, 
            idx: int
            ) -> Tuple[torch.tensor, torch.tensor]:
        """dunder method for indexing items. Note idx should a tuple.
        :param idx: takes tuples of (idx_of_company, idx_of_year)
        """
        figi_idx, period = self.get_index(idx)
        figi = self.figi_list.loc[figi_idx, "figi"]
        if isinstance(period, np.datetime64):
            pass
        elif isinstance(period, dt.date):
            period = np.datetime64(period)
        else:
            raise TypeError(f"the second item in idx must be either datetime or np.datetime64, got {type(period)}!")
        
        if period in self.period_list:
            end_period = period.strftime("%Y-%m-%d")
        else:
            period_idx = self.period_list["period"].lt(period).sum() - 1 # get periods up to the closest period in `self.period_list`
            end_period = self.period_list["period"].iloc[period_idx].strftime("%Y-%m-%d")
        
        if period_idx >= self.window_size:
            begin_period = self.period_list.loc[period_idx - self.window_size, "period"].strftime("%Y-%m-%d")
            query = f"SELECT {', '.join(DEFAULTS.price_multiples_data_cols)} FROM {self.table_name} WHERE (period between '{begin_period}' AND '{end_period}') AND (figi = '{figi}')"
            slice = SQLDatabase.to_pandas(query).astype(float).ffill()
            slice = self.process_timeseries(slice)\
                .replace(float("inf"), np.nan)\
                .replace(float("-inf"), np.nan)
            slice = torch.tensor(slice.values, dtype=self.dtype)
            padding_mask = torch.tensor([False for _ in range(self.window_size)],
                                dtype=torch.bool)
        else:
            query = f"SELECT {', '.join(DEFAULTS.price_multiples_data_cols)} FROM {self.table_name} WHERE (period <= '{end_period}') AND (figi = '{figi}')"
            slice = SQLDatabase.to_pandas(query).astype(float).ffill()
            slice = self.process_timeseries(slice)\
                .replace(float("inf"), np.nan)\
                .replace(float("-inf"), np.nan) 
            slice = torch.tensor(slice.values, dtype=self.dtype)
            padding_size = (self.window_size - period_idx)
            padding_mask = torch.tensor([True for _ in range(padding_size)] \
                                + [False for _ in range(self.window_size - padding_size)],
                                dtype=torch.bool)
            slice = F.pad(slice, pad=(0, 0, padding_size, 0), mode="constant", value=self.padding_val)

        return (utils.fillna(slice, self.padding_val), padding_mask)

    def __len__(self) -> int:
        figi_len, period_len = self.get_len()
        return (figi_len * period_len)
    
    def get_len(self) -> Tuple[int, int]:
        figi_len = len(self.figi_list)
        period_len = len(self.period_list) # period earlier will be padded
        return (figi_len, period_len)
    
    def __iter__(self) -> Generator[torch.tensor, None, None]:
        figi_len, year_len = self.get_len()
        for i in range(figi_len):
            for j in self.period_list.period.values:
                yield self.__getitem__((i, j))

class MacroDataset(Dataset):
    pass

class EstimateDataset(Dataset):
    pass