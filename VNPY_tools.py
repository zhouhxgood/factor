import pandas as pd
import numpy as np
from vnpy.trader.object import BarData, HistoryRequest
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.datafeed import get_datafeed
from vnpy.trader.utility import ArrayManager, BarGenerator
from functools import lru_cache
from datetime import datetime, timedelta
import concurrent.futures
from typing import List, Tuple


def bardata_to_dataframe(bars: BarData) -> pd.DataFrame:
    """
    将BarData对象列表转换为pandas DataFrame

    Args:
        bars (Bardata): BarData对象列表

    Returns:
        pd.DataFrame: bar数据的DataFrame
    """
    data = []
    for bar in bars:
        data.append([
            bar.datetime,
            bar.open_price,
            bar.high_price,
            bar.low_price,
            bar.close_price,
            bar.volume,
            bar.open_interest,
            bar.turnover
        ])

    df = pd.DataFrame(
        data,
        columns=[
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
            "turnover"
        ]
    )

    df = df.set_index("datetime")

    return df


def dataframe_to_bardata(df: pd.DataFrame, symbol: str, exchange: str, interval: Interval) -> list:
    """
    将DataFrame转换为BarData对象列表

    Args:
        df (pd.DataFrame): 包含OHLCV数据的DataFrame
        symbol (str): 合约代码
        exchange (str): 交易所代码
        interval (Interval): 数据周期

    Returns:
        list: BarData对象列表
    """
    # 检查DataFrame是否包含"datetime"列名
    if "datetime" not in df.columns:
        df = df.reset_index().rename(columns={"index": "datetime"})

    bars = []
    for _, row in df.iterrows():
        bar = BarData(
            gateway_name="BACKTESTING",
            symbol=symbol,
            exchange=Exchange(exchange),
            datetime=row["datetime"],
            open_price=row["open"],
            high_price=row["high"],
            low_price=row["low"],
            close_price=row["close"],
            volume=row["volume"],
            open_interest=row["open_interest"],
            turnover=row["turnover"],
            interval=interval
        )
        bars.append(bar)
    return bars


def check_and_clean_data(df: pd.DataFrame) -> tuple:
    """
    检查DataFrame中的数据完整性,并删除异常数据
    """
    # 确保索引是datetime类型，并统一时区
    df["oi_change"] = df["open_interest"].diff()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 将索引转换为UTC时区
    df.index = df.index.tz_localize(None)

    # 检查是否存在NaN值
    nan_data = df[df.isna().any(axis=1)]

    # 检查开盘价、最高价、最低价、收盘价是否为0
    zero_data = df[(df['open'] == 0) |
                   (df['high'] == 0) |
                   (df['low'] == 0) |
                   (df['close'] == 0)]

    # 检查时间间隔
    if len(df) > 1:
        time_diff = df.index[1:] - df.index[:-1]
        median_diff = pd.Timedelta(time_diff.median())
        is_daily = median_diff >= pd.Timedelta(days=1)
    else:
        is_daily = True  # 默认为日线数据

    # 检查并标记时间戳为早上8:59和晚上20:59的数据
    if not is_daily:
        target_times = [pd.Timestamp('08:59').time(), pd.Timestamp('20:59').time()]
        time_filter = df.index.map(lambda x: x.time() in target_times)
        filtered_time_data = df[time_filter]
    else:
        filtered_time_data = pd.DataFrame()

    # 检查价格、持仓不变且成交量为0的数据
    if not is_daily:
        #创建一个list,用于储存从10:16到10：29以及21：00-01：00的时间戳列表，用于后续的数据处理
        #异常数据的特征，
        time_list = []
        for i in range(0, 14):
            time_list.append((pd.Timestamp("10:16") + pd.Timedelta(minutes=i)).time())
        for i in range(0, 241):
            time_list.append((pd.Timestamp("21:00") + pd.Timedelta(minutes=i)).time())

        zero_volume_data = df[
            (df['open'] == df['high']) & (df['high'] == df['low']) & (df['volume'] == 0) & (df['oi_change'] == 0)]
        zero_volumne_data = zero_volume_data[zero_volume_data.index.map(lambda x: x.time() in time_list)]
    else:
        zero_volumne_data = pd.DataFrame()

    # 持仓量口径调整，2020年1月1日后，成交量、持仓量统计口径由双边调整为单边
    cutoff_date = pd.Timestamp('2020-01-01').tz_localize(None)
    mask = df.index < cutoff_date
    df.loc[mask, 'open_interest'] = df.loc[mask, 'open_interest'] / 2
    df.loc[mask, 'volume'] = df.loc[mask, 'volume'] / 2
    df.loc[mask, 'turnover'] = df.loc[mask, 'turnover'] / 2

    # 将所有异常数据合并
    anomaly_data = pd.concat([nan_data, zero_data, zero_volumne_data, filtered_time_data])
    anomaly_data = anomaly_data[~anomaly_data.index.duplicated(keep='first')]

    # 删除异常数据
    cleaned_data = df.drop(anomaly_data.index)

    # 按时间排序
    cleaned_data = cleaned_data.sort_index()

    return anomaly_data, cleaned_data


def get_data(symbol, exchange, start, end, interval="D", src="db", type="df"):
    if interval.upper() == "D" or interval.upper() == "DAY" or interval.upper() == "1D":
        interval = Interval.DAILY
    elif interval.upper() == "H" or interval.upper() == "HOUR":
        interval = Interval.HOUR
    elif interval.upper() == "M" or interval.upper() == "MIN" or interval.upper() == "MINUTE" or interval.upper() == "1M" or interval.upper() == "1MIN":
        interval = Interval.MINUTE
    elif interval.upper() == "T" or interval.upper() == "TICK":
        interval = Interval.TICK
    else:
        interval = interval

    if exchange == "SHFE" or exchange == "SHF":
        exc = Exchange.SHFE
    elif exchange == "WI":
        exc = Exchange.WI
    else:
        exc = exchange
    if src.upper() == "DB":
        bardata = get_database().load_bar_data(symbol=symbol,
                                               exchange=exc,
                                               start=start, end=end, interval=interval)
    elif src.upper() == "WIND":
        datafeed = get_datafeed()
        req = HistoryRequest(symbol=symbol, exchange=exc, start=start, end=end, interval=interval)
        bardata = datafeed.query_bar_history(req)

    if type.upper() == "DF":
        df = bardata_to_dataframe(bardata)
        anomaly_data, cleaned_data = check_and_clean_data(df)
        return cleaned_data[["open", "high", "low", "close", "volume", "open_interest", "turnover"]]
    else:
        return bardata


class NewArrayManager(ArrayManager):
    def __init__(self, size: int = 100) -> None:
        super().__init__(size)

    def init_from_df(self, df):
        self.open_array = df.open.values[-self.size:]
        self.high_array = df.high.values[-self.size:]
        self.low_array = df.low.values[-self.size:]
        self.close_array = df.close.values[-self.size:]
        self.volume_array = df.volume.values[-self.size:]
        self.turnover_array = df.turnover.values[-self.size:]
        self.open_interest_array = df.open_interest.values[-self.size:]
        self.count += df.shape[0]
        if not self.inited and self.count >= self.size:
            self.inited = True


if __name__ == "__main__":
    df = get_data(symbol="CUFI", exchange="WI", start="2020-01-01", end="2024-06-19", interval="D")
    print(df)
