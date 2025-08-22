import matplotlib.pyplot as plt
import numpy as np
import os
import talib
from typing import Union
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
import json
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.datafeed import get_datafeed
from vnpy.trader.object import BarData, HistoryRequest


def get_data(symbol, exchange, start, end, interval="D", src="db", type="df"):
    # interval 映射
    interval_map = {
        "D": Interval.DAILY, "DAY": Interval.DAILY, "1D": Interval.DAILY,
        "H": Interval.HOUR, "HOUR": Interval.HOUR,
        "M": Interval.MINUTE, "MIN": Interval.MINUTE, "MINUTE": Interval.MINUTE, "1M": Interval.MINUTE,
        "1MIN": Interval.MINUTE,
        "T": Interval.TICK, "TICK": Interval.TICK
    }
    interval = interval_map.get(interval.upper(), interval)

    # exchange 映射
    exchange_map = {"SHFE": Exchange.SHFE, "SHF": Exchange.SHFE, "WI": Exchange.WI}
    if isinstance(exchange, Exchange):
        exchange_str = exchange.value  # 取枚举的字符串值
    else:
        exchange_str = str(exchange)
    exc = exchange_map.get(exchange_str.upper(), exchange)

    # 数据源
    if src.upper() == "DB":
        bardata = get_database().load_bar_data(symbol=symbol, exchange=exc,
                                               start=start, end=end, interval=interval)
    elif src.upper() == "WIND":
        req = HistoryRequest(symbol=symbol, exchange=exc, start=start, end=end, interval=interval)
        bardata = get_datafeed().query_bar_history(req)
    else:
        raise ValueError(f"Unknown data source: {src}")

    # 输出格式
    if type.upper() == "DF":
        df = pd.DataFrame.from_records(
            (
                (bar.datetime, bar.open_price, bar.high_price, bar.low_price,
                 bar.close_price, bar.volume, bar.open_interest, bar.turnover)
                for bar in bardata
            ),
            columns=[
                "datetime", "open", "high", "low", "close",
                "volume", "open_interest", "turnover"
            ]
        )
        df.set_index("datetime", inplace=True)
        cleaned_data = check_and_clean_data(df)
        return cleaned_data[["open", "high", "low", "close", "volume", "open_interest", "turnover"]]

    return bardata

def check_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    检查 DataFrame 的数据完整性，并删除异常数据
    返回: (异常数据, 清洗后的数据)
    """
    # 确保索引为 DatetimeIndex，并去掉时区信息
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)

    # 计算持仓变化
    df["oi_change"] = df["open_interest"].diff()

    # ----------- 异常数据检测 -----------
    nan_mask = df.isna().any(axis=1)

    zero_price_mask = (df["open"] == 0) | (df["high"] == 0) | (df["low"] == 0) | (df["close"] == 0)

    # 判断是否为日线
    if len(df) > 1:
        median_diff = (df.index[1:] - df.index[:-1]).median()
        is_daily = median_diff >= pd.Timedelta(days=1)
    else:
        is_daily = True

    # 特定时间点过滤
    time_values = df.index.time
    if not is_daily:
        special_times = {pd.Timestamp("08:59").time(), pd.Timestamp("20:59").time()}
        filtered_time_mask = np.isin(time_values, list(special_times))
    else:
        filtered_time_mask = np.zeros(len(df), dtype=bool)

    # 价格不变 & 成交量=0 & OI不变，且时间在特定区间
    if not is_daily:
        time_range1 = pd.date_range("10:16", "10:29", freq="1min").time
        time_range2 = pd.date_range("21:00", "01:00", freq="1min").time  # 跨午夜
        abnormal_times = set(time_range1) | set(time_range2)

        flat_mask = (
            (df["open"] == df["high"]) &
            (df["high"] == df["low"]) &
            (df["volume"] == 0) &
            (df["oi_change"] == 0) &
            np.isin(time_values, list(abnormal_times))
        )
    else:
        flat_mask = np.zeros(len(df), dtype=bool)

    # ----------- 汇总异常数据 -----------
    anomaly_mask = nan_mask | zero_price_mask | filtered_time_mask | flat_mask
    anomaly_data = df[anomaly_mask].copy()

    # ----------- 持仓/成交量口径调整 -----------
    cutoff_date = pd.Timestamp("2020-01-01")
    before_cutoff = df.index < cutoff_date
    df.loc[before_cutoff, ["open_interest", "volume", "turnover"]] /= 2

    # 清洗数据
    cleaned_data = df[~anomaly_mask].sort_index()

    return cleaned_data

def round_to(value: float, target: float) -> float:
    value: Decimal = Decimal(str(value))
    target: Decimal = Decimal(str(target))
    rounded: float = float(int(round(value / target)) * target)
    return rounded


def indicator(df, sma=None, rsi=None, ema=None, boll=None, macd=None, drop: Union[str, bool] = False):
    # result = indicator(df, sma=20, ema=20, boll=2, macd=[12, 26, 9], drop=True)
    ind_list = ['close']
    col_list = ['volume', 'open_interest', 'turnover']
    if rsi is not None:
        df['rsi'] = talib.RSI(df.close, timeperiod=rsi)
        ind_list.append('rsi')
    if sma is not None:
        df['sma'] = talib.SMA(df.close, timeperiod=sma)
        ind_list.append('sma')
    if ema is not None:
        # 计算指数移动平均线（EMA）
        df['ema'] = talib.EMA(df.close, timeperiod=ema)
        ind_list.append('ema')
    if boll is not None:
        # 布林带, 30天简单移动平均
        df['upperband'], _, df['lowerband'] = talib.BBANDS(df.close, timeperiod=30, nbdevup=boll, nbdevdn=boll,
                                                           matype=0)
        ind_list.append('upperband')
        ind_list.append('lowerband')
    if macd is not None:
        if len(macd) == 3:
            fastperiod = macd[0]
            slowperiod = macd[1]
            signalperiod = macd[2]
        else:
            fastperiod = 12  # 短期EMA的时间周期
            slowperiod = 26  # 长期EMA的时间周期
            signalperiod = 9  # 信号线的时间周期naisai
        #  macd
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(
            df.close,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
        ind_list += ['macd', 'macdsignal', 'macdhist']
    df.dropna(inplace=True, how='any')
    if drop == 'price':
        col_list += ind_list
        return df[col_list]
    elif drop:
        return df[ind_list]
    else:
        return df


def plotLearning(scores, filename, x=None, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig(filename)


def check_data(df: pd.DataFrame, target_shape: tuple, clip=False) -> pd.DataFrame:
    current_shape = df.shape
    if current_shape[0] < target_shape[0]:
        rows_to_add = target_shape[0] - current_shape[0]
        zeros_to_add = pd.DataFrame(0, index=np.arange(rows_to_add), columns=df.columns)
        df = pd.concat([df, zeros_to_add], ignore_index=True)
    elif current_shape[0] > target_shape[0] and clip:
        df = df.iloc[:target_shape[0], :]
    if current_shape[1] < target_shape[1]:
        cols_to_add = target_shape[1] - current_shape[1]
        for i in range(cols_to_add):
            df[f'new_col_{i}'] = 0
    elif current_shape[1] > target_shape[1] and clip:
        df = df.iloc[:, :target_shape[1]]
    return df


def write_log(file_path, model_params):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as _:
            pass
    with open(file_path, 'a+') as file:
        file.seek(0, 0)
        file_empty = (file.read(1) == '')
        if not file_empty:
            file.write('\n')
        log_message = ', '.join(f"{key}={value}" for key, value in model_params.items())
        file.write(log_message + '\n')


def time_forward(t, delta, symbol='CUFI', exchange='WI', interval='d', mode='date'):
    #  根据数据采样频率选择回溯时间
    data, para = pd.DataFrame(), 1
    while len(data) < delta:
        if interval == 'm':
            s = datetime.strptime(t, '%Y-%m-%d') - 10 * para * timedelta(minutes=delta)
        elif interval == 'd':
            s = datetime.strptime(t, '%Y-%m-%d') - 5 * para * timedelta(days=delta)
        elif interval == '1h' or interval == 'h':
            s = datetime.strptime(t, '%Y-%m-%d') - 600 * para * timedelta(minutes=delta)
        else:
            s = datetime.strptime(t, '%Y-%m-%d') - 10 * para * timedelta(days=delta)
        s = s.strftime('%Y-%m-%d %H:%M:%S')
        data = get_data(symbol=symbol, exchange=exchange, interval=interval, start=s, end=t)
        para += 1
    res = data[len(data) - delta - 1:]
    if mode == 'date':
        return res.index[0]
    return res


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def close_changes(df):
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")
    df_mid = df.copy()
    df_mid.loc[:, 'daily_change'] = 100 * df['close'].pct_change()
    df = df_mid
    df_positive = df[df['daily_change'] > 0]  # 涨的
    df_negative = df[df['daily_change'] < 0]  # 跌的
    up_counts = len(df_positive) / len(df)
    down_counts = len(df_negative) / len(df)
    up_counts = round(100 * up_counts, 2)
    down_counts = round(100 * down_counts, 2)
    mean_change_positive = round(df_positive['daily_change'].mean(), 4)
    mean_change_negative = round(df_negative['daily_change'].mean(), 4)
    variance = round(df['daily_change'].var(), 4)
    return np.array([up_counts, mean_change_positive]), np.array([down_counts, mean_change_negative]), variance


def standardize(df, pos=True):
    res = (df - df.mean()) / df.std(ddof=1).replace(0, 1)
    if pos:
        return positional_encoding(res)
    return res


def positional_encoding(df):
    window_size, d_model = len(df), len(df.columns)
    position = np.arange(window_size)
    div_term = 10000.0 ** (np.arange(0, (d_model + 1) // 2, 1) / d_model)
    pos_encoding = np.zeros((window_size, d_model))
    omega = position[:, np.newaxis] / div_term
    pos_encoding[:, 0::2] = np.sin(omega)
    if d_model % 2 == 0:
        pos_encoding[:, 1::2] = np.cos(omega)
    else:
        pos_encoding[:, 1::2] = np.cos(omega[:, :d_model // 2])
    return df + pos_encoding


def features_extract(df, window_size=256, intervals_n=8):
    features_n = len(df.columns)
    res = pd.DataFrame(np.nan, index=df.index, columns=df.columns)

    def cal_state(x, y):
        if x == 0:
            return 0
        elif x > 0:
            return x // y + 1
        else:
            return x // y

    # 滑动窗口计算每个点的位置
    for i in range(window_size - 1, len(df)):
        # 获取当前窗口的数据
        window_data = df[i - window_size + 1:i + 1]

        # 计算窗口的均值和标准差
        window_mean = window_data.mean(axis=0)
        window_std = window_data.std(axis=0)
        # 定义区间边界：均值上下三个标准差
        b = window_data.iloc[-1]
        center = window_data.iloc[-1] - window_mean
        states = pd.Series(index=window_std.index)
        for j in range(features_n):
            states.iloc[j] = cal_state(center.iloc[j], window_std.iloc[j])

        res.iloc[i] = states.clip(lower=-intervals_n / 2, upper=intervals_n / 2)
    res = res.dropna(how='any')
    return res


def cal_commission(price, comm):
    comm = float(comm)
    if comm > 1:
        return comm
    else:
        return price * comm


if __name__ == '__main__':
    config_path = r"D:\PythonProject\drl\config\config.json"
    with open(config_path, "r", encoding="utf=8") as json_file:
        config = json.load(json_file)
    fu_str = "ZN"
    close = 17000
    #  ZN, AL, PB
    price_tick = float(config["futures"]["non_ferrous"][fu_str + "_main"]["price_tick"])
    size = float(config["futures"]["non_ferrous"][fu_str + "_main"]["size"])
    commission = float(config["futures"]["non_ferrous"][fu_str + "_main"]["commission"])
    one_price_tick_com = cal_commission(size * close, commission)
    print(commission)
    print(price_tick * size)
    print(one_price_tick_com)
