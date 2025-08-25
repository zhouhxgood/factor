import numpy as np
import pandas as pd
from scipy import stats
from sympy import series
from constant import Variable


def shannon_entropy(data, bins=5):
    # 把连续数值切成 bins 段
    categories = pd.cut(data, bins=bins, labels=False)
    probs = pd.Series(categories).value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs))


def construct_templates(timeseries_data, m):  # 时间序列分割
    num_windows = len(timeseries_data) - m + 1
    return np.array([timeseries_data[x : x + m] for x in range(0, num_windows)])


def fuzzy_membership(dist, r, n=2):
    return np.exp(- (dist ** n) / r)


def get_matches(templates, r):
    return len(list(filter(lambda x: is_match(x[0], x[1], r), combinations(templates))))


def combinations(x):  # 生成上三角矩阵索引
    idx = np.stack(np.triu_indices(len(x), k=1), axis=-1)
    return x[idx]


def is_match(template_1, template_2, r):
    return np.all([abs(x - y) < r for (x, y) in zip(template_1, template_2)])  # 切比雪夫距离


# def is_match(template_1, template_2, r):
#     euclidean_distance = np.linalg.norm(template_1 - template_2)
#     return euclidean_distance < r  # 欧几里得距离


def sample_entropy(timeseries_data, window_size, r):  # r是0.1到0.25倍的数据标准差
    r *= float(np.std(timeseries_data))
    B = get_matches(construct_templates(timeseries_data, window_size), r)
    A = get_matches(construct_templates(timeseries_data, window_size + 1), r)

    if B == 0:
        return np.inf
    if A == 0:
        return -np.inf
    return -np.log(A / B)


def factor_metrics(df, future_returns):
    if isinstance(future_returns, (pd.Series, pd.DataFrame)):
        future_returns = future_returns.squeeze()  # 转换为Series
    else:
        future_returns = pd.Series(future_returns, index=df.index)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame索引应为日期时间格式")

    # 检查长度是否匹配
    if len(df) != len(future_returns):
        raise ValueError("df和future_returns的长度必须相同")
    results = {}

    for col in df.columns:
        factor_data = df[col]
        data_df = pd.DataFrame({
            'factor': factor_data,
            'returns': future_returns
        }).dropna()

        if len(data_df) < 2:  # 需要至少2个点计算相关性和其他指标
            results[col] = {
                'IC': np.nan,
                'IC_std': np.nan,
                'ICIR': np.nan,
                'Rank_IC': np.nan,
                'Rank_IC_std': np.nan,
                'Rank_ICIR': np.nan,
            }
            continue


        ic = data_df['factor'].corr(data_df['returns'])
        rank_ic = data_df['factor'].corr(data_df['returns'], method='spearman')
        ic_std = data_df['factor'].std()
        icir = ic / ic_std if ic_std != 0 else np.nan
        results[col] = {
            'IC': ic,
            'IC_std': ic_std,
            'ICIR': icir,
            'Rank_IC': rank_ic,
        }

    return pd.DataFrame(results).T


class Factor:
    def __init__(self, df, data, to_return=True):
        """
        因子计算
        函数开头表示因子所属两大类的类别
        VP: volume-price factor 量价相关因子
        F: fundamental factor 基本面因子
        二级类别
        Vol: volatility 波动率
        Diff: difference 差分
        R: ratio 比率
        LR: linear regression 线性回归
        Ent: entropy 熵

        因子计算函数名的格式为: 大类_二级类别_名称
        例如动量因子
        VP_R_mom
        动量因子的基本定义是K日累计涨幅，是量价大类下的一种比率因子
        参数:
        df: DataFrame, 包含原始数据的DataFrame
        data: Series或array-like, 目标默认为下一时间单位收益率序列
        to_return: 是否
        """
        self.df = df.copy()
        self._validate_columns()
        target_series = self.check_data(data)
        if to_return:
            self.df = self.df.iloc[:-2]  # 如果是计算得到的收益率，最后一条记录的下一个收益率是未知的，直接删除最后一行
            target_series = target_series.pct_change().shift(-1)
        # 索引对齐
        self.df['target'] = target_series
        self.df.columns = [c.upper() for c in self.df.columns]
        #  2020年之前成交量是双边计算的
        if "VOLUME" in self.df.columns and isinstance(self.df.index, pd.DatetimeIndex):
            self.df.loc[self.df.index < pd.Timestamp("2020-01-01"), "VOLUME"] /= 2


    def check_columns(self, columns):
        """检查指定的列是否存在于DataFrame中"""
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"以下列不存在于DataFrame中: {missing_cols}")

    def fill_na(self, data=None, method='ffill', **kwargs):
        """
        填充缺失值
        """
        interpolate_methods = {'linear', 'time', 'index', 'polynomial', 'spline'}
        if data is None:
            if method == 'ffill':
                self.df.ffill(inplace=True)
            elif method in interpolate_methods:
                self.df.interpolate(method=method, inplace=True, **kwargs)
            else:
                raise ValueError(f"不支持的填充方法: {method}")
            self.df.dropna(how='any', inplace=True)
            return None
        else:
            data = data.upper()
            series = self.check_data(data)
            if method == 'ffill':
                series = series.ffill()
            elif method in interpolate_methods:
                series = series.interpolate(method=method, **kwargs)
            else:
                raise ValueError(f"不支持的填充方法: {method}")
            series.dropna(how='any', inplace=True)
            return series


    def check_data(self, data, num=1):
        if isinstance(data, str):
            self.check_columns([data])
            s = self.df[data]
        elif isinstance(data, pd.Series):
            if num > 1:
                raise ValueError(f"计算多变量因子所需变量数目不够，num={num}")
            s = data
        elif isinstance(data, list):
            s = self.df[data]
        elif isinstance(data, pd.DataFrame):
            s = data.iloc[:, num]
        else:
            raise TypeError("data 必须是列名 (str)、列表、 Series 或 DataFrame")
        return s

    """
    因子计算分为单变量因子和多变量因子，如果使用self.df则会先检查变量是否匹配
    如果是单变量因子
    data: str 或 pd.Series 或 pd.DataFrame
        - 如果是字符串，则从self.df中取该列
        - 如果是Series，则直接使用该序列
        - 如果是DataFrame，则默认取第一列
    如果是多变量因子
    data: str 或 pd.Series 或 pd.DataFrame
        - 如果是字符串列表，则从self.df中取这些变量列
        - 如果是DataFrame，则默认取输入df中计算因子所需变量数目个列
    其余参数参见注释
    """

    def VP_R_mom(self, data=None, window=5):
        """
        基本动量因子，即多日累计涨跌幅
        window: int 计算窗口
        """
        if data is None:
            data = self.check_data("CLOSE", num=1)
            return data.pct_change(window).rename("VP_R_mom")
        elif isinstance(data, str):
            data = data.upper()
        return self.check_data(data, num=1).pct_change(window).rename("VP_R_mom")


    def VP_R_tr(self, data=None):
        """
        换手率因子，成交量 / 持仓量
        """
        if data is None:
            data = self.check_data(["VOLUME", "OPEN_INTEREST"], num=2)
            return (data["VOLUME"] / data["OPEN_INTEREST"]).rename("VP_R_tr")
        else:
            data = self.check_data(data, num=2)
            return (data.iloc[:, 0] / data.iloc[:, 1]).rename("VP_R_tr")


    def _validate_columns(self):
        """
        检查输入的df列名是否都在Variable枚举中
        不区分大小写
        """
        df_cols_upper = {c.upper() for c in self.df.columns}
        allowed_cols = Variable.allowed_columns()

        illegal_cols = df_cols_upper - allowed_cols
        if illegal_cols:
            raise ValueError(
                f"DataFrame 包含非法列: {illegal_cols}. "
                f"允许的列名为: {allowed_cols}"
            )









    def volatility_factor(self, data, window=20):
        """
        计算波动率因子

        参数:
        data: str, 价格列名
        window: int, 计算窗口

        返回:
        Series, 波动率因子值
        """
        self.check_columns([data])
        return self.df[data].pct_change().rolling(window=window).std()

    def ma_deviation_factor(self, data, window=20):
        """
        计算移动平均偏离度因子

        参数:
        data: str, 价格列名
        window: int, 计算窗口

        返回:
        Series, 移动平均偏离度因子值
        """
        self.check_columns([data])
        ma = self.df[data].rolling(window=window).mean()
        return (self.df[data] - ma) / ma

    def rsi_factor(self, data, window=14):
        """
        计算相对强弱指标(RSI)因子

        参数:
        data: str, 价格列名
        window: int, 计算窗口

        返回:
        Series, RSI因子值
        """
        self.check_columns([data])
        delta = self.df[data].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def macd_factor(self, data, fast=12, slow=26, signal=9):
        """
        计算MACD因子

        参数:
        data: str, 价格列名
        fast: int, 快速EMA周期
        slow: int, 慢速EMA周期
        signal: int, 信号线周期

        返回:
        DataFrame, 包含MACD、信号线和柱状图
        """
        self.check_columns([data])
        ema_fast = self.df[data].ewm(span=fast).mean()
        ema_slow = self.df[data].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal

        return pd.DataFrame({
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }, index=self.df.index)

    def bollinger_bands_factor(self, data, window=20, num_std=2):
        """
        计算布林带因子

        参数:
        data: str, 价格列名
        window: int, 计算窗口
        num_std: int, 标准差倍数

        返回:
        DataFrame, 包含布林带上轨、中轨、下轨和价格位置
        """
        self.check_columns([data])
        middle_band = self.df[data].rolling(window=window).mean()
        std = self.df[data].rolling(window=window).std()

        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)

        # 价格在布林带中的位置 (0-1)
        band_position = (self.df[data] - lower_band) / (upper_band - lower_band)

        return pd.DataFrame({
            'bollinger_upper': upper_band,
            'bollinger_middle': middle_band,
            'bollinger_lower': lower_band,
            'bollinger_position': band_position
        }, index=self.df.index)

    def volume_ma_factor(self, volume_col, window=20):
        """
        计算成交量移动平均因子

        参数:
        volume_col: str, 成交量列名
        window: int, 计算窗口

        返回:
        Series, 成交量移动平均因子值
        """
        self.check_columns([volume_col])
        return self.df[volume_col].rolling(window=window).mean()

    def volume_ratio_factor(self, volume_col, window=20):
        """
        计算量比因子

        参数:
        volume_col: str, 成交量列名
        window: int, 计算窗口

        返回:
        Series, 量比因子值
        """
        self.check_columns([volume_col])
        return self.df[volume_col] / self.df[volume_col].rolling(window=window).mean()

    def obv_factor(self, data, volume_col):
        """
        计算能量潮(OBV)因子

        参数:
        data: str, 价格列名
        volume_col: str, 成交量列名

        返回:
        Series, OBV因子值
        """
        self.check_columns([data, volume_col])
        price_change = self.df[data].diff()
        obv = pd.Series(0, index=self.df.index)
        obv[price_change > 0] = self.df[volume_col]
        obv[price_change < 0] = -self.df[volume_col]

        return obv.cumsum()

    def price_volume_correlation_factor(self, data, volume_col, window=20):
        """
        计算量价相关性因子

        参数:
        data: str, 价格列名
        volume_col: str, 成交量列名
        window: int, 计算窗口

        返回:
        Series, 量价相关性因子值
        """
        self.check_columns([data, volume_col])
        return self.df[data].rolling(window=window).corr(self.df[volume_col])

    def intraday_volatility_factor(self, high_col, low_col, open_col):
        """
        计算日内波动率因子

        参数:
        high_col: str, 最高价列名
        low_col: str, 最低价列名
        open_col: str, 开盘价列名

        返回:
        Series, 日内波动率因子值
        """
        self.check_columns([high_col, low_col, open_col])
        return (self.df[high_col] - self.df[low_col]) / self.df[open_col]

    def close_open_ratio_factor(self, close_col, open_col):
        """
        计算收盘开盘比因子

        参数:
        close_col: str, 收盘价列名
        open_col: str, 开盘价列名

        返回:
        Series, 收盘开盘比因子值
        """
        self.check_columns([close_col, open_col])
        return self.df[close_col] / self.df[open_col] - 1

    def basis_factor(self, basis_col, data):
        """
        计算基差因子

        参数:
        basis_col: str, 基差列名
        data: str, 价格列名

        返回:
        Series, 基差因子值
        """
        self.check_columns([basis_col, data])
        return self.df[basis_col] / self.df[data]

    def calculate_ic(self, factor_series):
        """
        计算因子与目标收益率的IC值

        参数:
        factor_series: Series, 因子值序列

        返回:
        float, IC值
        """
        # 对齐数据并移除NaN值
        aligned_data = pd.DataFrame({
            'factor': factor_series,
            'returns': self.target_series
        }).dropna()

        if len(aligned_data) < 2:
            return np.nan

        return aligned_data['factor'].corr(aligned_data['returns'])

    def evaluate_factor(self, factor_series, rolling_window=12):
        """
        评估因子性能

        参数:
        factor_series: Series, 因子值序列
        rolling_window: int, 滚动窗口大小

        返回:
        dict, 包含各种评估指标
        """
        # 对齐数据并移除NaN值
        aligned_data = pd.DataFrame({
            'factor': factor_series,
            'returns': self.target_series
        }).dropna()

        if len(aligned_data) < 2:
            return {
                'IC': np.nan,
                'IC_std': np.nan,
                'ICIR': np.nan,
                'IC_t_stat': np.nan,
                'IC_p_value': np.nan
            }

        # 计算滚动IC
        rolling_ic = []
        for i in range(rolling_window, len(aligned_data) + 1):
            window_data = aligned_data.iloc[i - rolling_window:i]
            ic = window_data['factor'].corr(window_data['returns'])
            rolling_ic.append(ic)

        rolling_ic = pd.Series(rolling_ic)

        # 计算IC指标
        ic = aligned_data['factor'].corr(aligned_data['returns'])
        ic_std = rolling_ic.std()
        icir = rolling_ic.mean() / ic_std if ic_std != 0 else np.nan

        # 计算IC的t统计量和p值
        if len(rolling_ic) > 1:
            ic_t_stat, ic_p_value = stats.ttest_1samp(rolling_ic, 0)
        else:
            ic_t_stat, ic_p_value = np.nan, np.nan

        return {
            'IC': ic,
            'IC_std': ic_std,
            'ICIR': icir,
            'IC_t_stat': ic_t_stat,
            'IC_p_value': ic_p_value
        }


# 示例用法
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json
    from utils import *


    config_path = r"D:\PythonProject\drl\config\config.json"
    with open(config_path, "r", encoding="utf=8") as json_file:
        config = json.load(json_file)
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    start = '2016-02-01'
    train_end = "2021-01-01"
    val_end = "2025-8-25"
    pred_end = "2025-07-01"
    fu_list = ['CU', 'AL', 'ZN', 'NI', 'PB', 'SN']
    fu = 'CU'
    freq = 'd'
    sector = "non_ferrous"
    price_tick = float(config["futures"][sector][fu + "_main"]["price_tick"])
    size = float(config["futures"][sector][fu + "_main"]["size"])
    comm = config["futures"][sector][fu + "_main"]["commission"]
    exc = config["futures"][sector][fu + "_main"]["exchange"]
    exc = getattr(Exchange, exc)
    dfzl = get_data(symbol=f"{fu}ZL", exchange=exc, start=start, end=val_end, interval=freq)
    df = pd.read_excel('./data/cu.xlsx')
    df.columns = ['date', 'inventory', 'profit', 'basis', 'warrant']
    df.set_index('date', inplace=True)
    df = pd.concat([df, dfzl], axis=1)
    df = df[df.index >= '2016-01-01']
    df.dropna(subset='close', inplace=True)
    fa = Factor(df=df, data='close')
    res = fa.fill_na('inventory', method='ffill')
    # fa.fill_na(method='ffill')
    f = fa.VP_R_mom(data='close', window=10)
    f2 = fa.VP_R_tr().to_frame()
    ta = fa.df['TARGET']
    factor_m = factor_metrics(f2, ta)
    print(f)

    # 创建示例数据
    # np.random.seed(42)
    # n_periods = 100
    #
    # # 创建日期索引
    # dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    #
    # # 创建价格和量数据
    # df = pd.DataFrame({
    #     'close': np.cumprod(1 + np.random.randn(n_periods) * 0.01) * 100,
    #     'open': np.cumprod(1 + np.random.randn(n_periods) * 0.01) * 100 * (1 + np.random.randn(n_periods) * 0.005),
    #     'high': np.cumprod(1 + np.random.randn(n_periods) * 0.01) * 100 * (
    #                 1 + np.abs(np.random.randn(n_periods) * 0.01)),
    #     'low': np.cumprod(1 + np.random.randn(n_periods) * 0.01) * 100 * (
    #                 1 - np.abs(np.random.randn(n_periods) * 0.01)),
    #     'basis': np.random.randn(n_periods) * 0.1,
    #     'volume': np.random.lognormal(10, 1, n_periods),
    #     'open_interest': np.random.lognormal(12, 0.8, n_periods),
    #     'turnover': np.random.lognormal(15, 1.2, n_periods)
    # }, index=dates)
    #
    # # 创建目标收益率序列
    # target_returns = df['close'].pct_change().shift(-1)  # 下一期收益率
    #
    # # 初始化因子计算类
    # factor_calculator = Factor(df, target_returns)
    #
    # # 计算各种因子
    # momentum = factor_calculator.momentum_factor('close', window=20)
    # volatility = factor_calculator.volatility_factor('close', window=20)
    # rsi = factor_calculator.rsi_factor('close', window=14)
    # volume_ma = factor_calculator.volume_ma_factor('volume', window=20)
    # volume_ratio = factor_calculator.volume_ratio_factor('volume', window=20)
    #
    # # 计算IC值
    # momentum_ic = factor_calculator.calculate_ic(momentum)
    # volatility_ic = factor_calculator.calculate_ic(volatility)
    #
    # print(f"动量因子IC值: {momentum_ic:.4f}")
    # print(f"波动率因子IC值: {volatility_ic:.4f}")
    #
    # # 评估因子性能
    # momentum_eval = factor_calculator.evaluate_factor(momentum)
    # print("\n动量因子评估:")
    # for key, value in momentum_eval.items():
    #     print(f"{key}: {value:.4f}")
    #
    # # 计算多列因子
    # intraday_vol = factor_calculator.intraday_volatility_factor('high', 'low', 'open')
    # close_open_ratio = factor_calculator.close_open_ratio_factor('close', 'open')
    # basis_factor = factor_calculator.basis_factor('basis', 'close')
    #
    # print(f"\n日内波动率因子IC值: {factor_calculator.calculate_ic(intraday_vol):.4f}")
    # print(f"收盘开盘比因子IC值: {factor_calculator.calculate_ic(close_open_ratio):.4f}")
    # print(f"基差因子IC值: {factor_calculator.calculate_ic(basis_factor):.4f}")
