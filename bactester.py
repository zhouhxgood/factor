from utils import *
from VNPY_tools import *
import torch
import matplotlib.dates as mdates



def calculate_profit_and_equity(df, price_tick, slippage, commission=0.0, size=1):
    indexes = df.index
    df = df.copy()
    # 初始化持仓、成本价、盈利和净值列
    df['entry_price'] = np.nan
    df['trading_pnl'] = 0.0
    df['floating_pnl'] = 0.0
    df['total_pnl'] = 0.0
    df['order_status'] = 0  # 记录订单是否成交
    df['start_pos'] = 0  # 起始仓位
    df['end_pos'] = 0  # 结束仓位
    df['commission'] = 0.0
    # df['prev_entry'] = np.nan

    total_orders = 0
    filled_orders = 0
    prev_entry = np.nan
    current_entry = np.nan

    for i in range(1, len(df)):
        prev_index = df.index[i - 1]
        current_index = df.index[i]

        prev_action = df.loc[prev_index, 'action']
        prev_position = df.loc[prev_index, 'end_pos']
        target_position = prev_action

        # 获取价格数据
        prev_close = df.loc[prev_index, 'close']
        current_open = df.loc[current_index, 'open']
        current_high = df.loc[current_index, 'high']
        current_low = df.loc[current_index, 'low']
        current_close = df.loc[current_index, 'close']

        delta = target_position - prev_position
        current_position = prev_position

        trading_pnl = 0.0  # 平仓实现的盈亏
        floating_pnl = 0.0  # 持仓浮动盈亏

        # --- 处理平仓逻辑 ---
        if delta != 0 and prev_position != 0:
            total_orders += 1  # delta不等于零说明有下单操作，总订单数加一
            if prev_position > 0:
                # 平多仓，成交价取（前收盘-滑点）和当前开盘的较高者
                exit_price_candidate = round_to(prev_close - slippage, price_tick)
            else:
                # 平空仓，成交价取（前收盘+滑点）和当前开盘的较低者
                exit_price_candidate = round_to(prev_close + slippage, price_tick)
            # 检查平仓价是否在价格范围内
            if exit_price_candidate <= current_high and prev_position > 0:  # 平多仓
                exit_price_candidate, trade = max(exit_price_candidate, current_open), True
            elif current_low <= exit_price_candidate and prev_position < 0:  # 平空仓
                exit_price_candidate, trade = min(exit_price_candidate, current_open), True
            else:
                trade = False
            if trade:
                filled_orders += 1
                current_position = 0
                prev_entry = current_entry
                trading_pnl += (exit_price_candidate - prev_entry) * prev_position * size
                current_entry = exit_price_candidate
                df.loc[current_index, 'entry_price'] = current_entry
                df.loc[current_index, 'order_status'] += 1

        # --- 处理开仓逻辑 ---
        if delta != 0 and current_position == 0 and target_position != 0:
            total_orders += 1
            if target_position > 0:
                # 开多仓，成交价取（前收盘+滑点）和当前开盘的较低者
                entry_price_candidate = min(round_to(prev_close + slippage, price_tick), current_open)
            else:
                # 开空仓，成交价取（前收盘-滑点）和当前开盘的较高者
                entry_price_candidate = max(round_to(prev_close - slippage, price_tick), current_open)
            # 检查开仓价是否在价格范围内
            if current_low <= entry_price_candidate and target_position > 0:
                entry_price_candidate, trade = min(entry_price_candidate, current_open), True
            elif entry_price_candidate <= current_high and target_position < 0:
                entry_price_candidate, trade = max(entry_price_candidate, current_open), True
            else:
                trade = False
            if trade:
                filled_orders += 1
                current_position = target_position
                prev_entry = current_entry
                current_entry = entry_price_candidate
                df.loc[current_index, 'entry_price'] = current_entry
                df.loc[current_index, 'order_status'] += 1

        # --- 计算浮动盈亏 ---
        if current_position != 0 and not np.isnan(current_entry):
            floating_pnl = (current_close - current_entry) * current_position * size

        # 更新当前行的数据

        df.loc[current_index, 'trading_pnl'] = trading_pnl
        df.loc[current_index, 'floating_pnl'] = floating_pnl
        df.loc[current_index, 'start_pos'] = prev_position
        df.loc[current_index, 'end_pos'] = current_position
        df.loc[current_index, 'commission'] = np.abs(current_position - prev_position) * cal_commission(size * current_entry, commission)
        # df.loc[current_index, 'prev_entry'] = prev_entry

    # 浮动盈亏差分(持仓收益) + 交易盈亏 = 每个时间间隔的收益
    df['total_pnl'] = df['floating_pnl'].diff() + df['trading_pnl']
    df['total_pnl'] = df['total_pnl'].fillna(0)
    df['commission'] = df['commission'].fillna(0)
    df['size'] = size
    df['total_pnl'] = df['total_pnl'] - df['commission']

    # df['NAV'] = (df['total_pnl'] / df['close'] + 1).cumprod()
    df['NAV'] = df['total_pnl'].cumsum()

    fill_rate = filled_orders / total_orders if total_orders > 0 else 0.0
    df.index = indexes
    return df, [filled_orders, total_orders, fill_rate]


def prediction2pic(test_close, returns, nav_test, pos_test, title=None, pic_path=None, pred_start=None):
    # up_info, down_info, variance = close_changes(test_close)
    proportion = (returns > 0).sum() / returns.shape[0]  # 计算胜率l
    flat = (returns == 0).sum() / returns.shape[0]
    wrong = (returns < 0).sum() / returns.shape[0]

    size = 7
    g = (np.sqrt(5) + 1) / 2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(g * size, size), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    ax1.axhline(y=1, color='gray', linestyle='--')
    ax11 = ax1.twinx()
    ax11.plot(test_close.index, test_close, label='price', color='black', alpha=0.5)
    if pred_start:
        pred_start = pd.to_datetime(pred_start)
        before = test_close.index <= pred_start
        after = test_close.index >= pred_start
        ax1.plot(test_close.index[before], nav_test[before], label='NAV', color='blue', alpha=0.5)
        ax1.plot(test_close.index[after], nav_test[after], label='NAV pred', color='red', alpha=0.5)
    else:
        ax1.plot(test_close.index, nav_test, label='NAV', color='blue', alpha=0.5)
    x_min, x_max = ax1.get_xlim()
    y_min, y_max = ax1.get_ylim()

    ax1.text(x_min + 0.02 * (x_max - x_min), y_max - 0.1 * (y_max - y_min), f'win rate:{proportion.iloc[0]:.2%}',
             color='r')
    ax1.text(x_min + 0.02 * (x_max - x_min), y_max - 0.15 * (y_max - y_min), f'flat:{flat.iloc[0]:.2%}', color='r')
    ax1.text(x_min + 0.02 * (x_max - x_min), y_max - 0.2 * (y_max - y_min), f'wrong:{wrong.iloc[0]:.2%}', color='r')

    if title is None:
        title = 'NAV in Testing Env'
    ax1.set_title(title)
    handles, labels = [], []
    for line in ax1.lines + ax11.lines:
        handles.append(line)
        labels.append(line.get_label())
    # 过滤掉标签以单个下划线开头的线条
    filtered_handles = [h for h in handles if not h.get_label().startswith('_')]
    filtered_labels = [l for l in labels if not l.startswith('_')]
    # 使用过滤后的线条和标签调用 legend
    ax1.legend(filtered_handles, filtered_labels, loc='lower left')
    ax1.set_ylabel('Price / NAV')

    ax2.scatter(test_close.index, pos_test, color='red', s=10)
    ax2.set_ylabel('Actions')
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax2.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if not pic_path:
        pic_path = './pic/' + datetime.today().strftime('%Y%m%d') + '.png'
    fig.savefig(pic_path, dpi=1080 / size)  # 分辨率1080p
    plt.close()


if __name__ == '__main__':
    from factor import Factor
    config_path = r"D:\PythonProject\drl\config\config.json"
    with open(config_path, "r", encoding="utf=8") as json_file:
        config = json.load(json_file)
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    start = '2016-02-01'
    train_end = "2021-01-01"
    val_end = "2025-8-25"
    pred_end = "2025-07-01"
    fu = 'CU'
    freq = 'd'
    sector = "non_ferrous"
    price_tick = float(config["futures"][sector][fu + "_main"]["price_tick"])
    size = float(config["futures"][sector][fu + "_main"]["size"])
    comm = config["futures"][sector][fu + "_main"]["commission"]
    exc = config["futures"][sector][fu + "_main"]["exchange"]
    exc = getattr(Exchange, exc)
    env_kwargs = {
        'window_size': 256,
        'mode': "return",
        'over_price': price_tick,
        'price_tick': price_tick,
        'size': size,
        'commission': comm,
    }

    dfzl = get_data(symbol=f"{fu}ZL", exchange=exc, start=start, end=val_end, interval=freq)
    df = pd.read_excel('./data/cu.xlsx')
    df.columns = ['date', 'inventory', 'profit', 'basis', 'warrant']
    df.set_index('date', inplace=True)
    df = pd.concat([df, dfzl], axis=1)
    df = df[df.index >= '2016-01-01']
    fa = Factor(df=df, data='close', to_return=True)
    fa.fill_na(data='profit', method='ffill', fix=True)
    f1 = fa.B_Diff(data='profit', periods=1, add=True)
    name = f1.columns[0]
    df = pd.concat([df, f1], axis=1)
    df["action"] = np.where(
        df[name] > 0, -1,  # 大于0 → -1
        np.where(df[name] < 0, 1, 0)  # 小于0 → 1，等于0 → 0
    )
    df.dropna(how='any', inplace=True)
    res, orders = calculate_profit_and_equity(df, slippage=env_kwargs['over_price'], price_tick=env_kwargs['price_tick'],
                                commission=env_kwargs['commission'], size=env_kwargs['size'])
    print(f"{fu}价差boll成交量{orders[0]},订单量{orders[1]},成交率为{orders[2] * 100:.2f}%")
    plt.plot(res.index, res['NAV'], label='NAV')
    plt.show()