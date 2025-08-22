import matplotlib.pyplot as plt
import json
from vnpy.trader.constant import Exchange
from stats_test import *
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import *
from sklearn.metrics import accuracy_score


config_path = r"D:\PythonProject\drl\config\config.json"
with open(config_path, "r", encoding="utf=8") as json_file:
    config = json.load(json_file)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签


start = '2020-07-01'
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

plt.plot(df)
plt.show()

df = pd.read_excel('./data/cu.xlsx')
df.columns = ['date', 'warrant', 'close', 'basis']
df.set_index('date', inplace=True)
df = pd.concat([df, dfzl], axis=1)
df = df.dropna()
# # col = 'basis'
# col = 'close'
# # col = 'warrant'
# adf_test(df[col], save_path=f'./stats_test/{col}.txt')
df = df.loc[:, ~df.T.duplicated()]
df = df.copy()

# 1. 计算收益率
df['return'] = df['close'].pct_change()  # 或 np.log(df['Close'] / df['Close'].shift(1))
df['target'] = (df['return'].shift(-1) > 0).astype(int)  # 1 = 涨, 0 = 跌
df.dropna(inplace=True)  # 去掉第一个NaN

# 2. 选择特征和目标
# features = ['warrant', 'basis', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'turnover']
features = ['warrant', 'basis']
X = df[features]
y = df['target']

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 4. 初始化 XGBoost 回归模型
# model = XGBRegressor(
#     n_estimators=500,
#     learning_rate=0.1,
#     max_depth=3,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
# )
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'     # 交叉熵损失
)

# 5. 模型训练
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train)],  # 可选验证集
    verbose=True
)

# 6. 预测
y_pred = model.predict(X_test)
y_t = model.predict(X_train)
# 7. 评估模型
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Mean Squared Error: {mse:.6f}")
# print(f"R^2 Score: {r2:.4f}")

print("Accuracy:", accuracy_score(y_test, y_pred))

# 8. 可选：查看特征重要性

plt.barh(features, model.feature_importances_)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.show()


