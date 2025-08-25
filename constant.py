from enum import Enum


class Variable(Enum):
    """
   大宗商品期货研究中用于计算因子的基本变量
   分为量价和基本面两大类
    """
    #  量价类
    OPEN = "OPEN"  # 开盘价
    HIGH = "HIGH"  # 最高价
    LOW = "LOW"  # 最低价
    CLOSE = "CLOSE"  # 收盘价
    VOLUME = "VOLUME"  # 成交量
    OPEN_INTEREST = "OPEN_INTEREST"  # 持仓量
    TURNOVER = "TURNOVER" # 成交额
    SETTLEMENT = "SETTLEMENT"  # 结算价
    SPREAD = "SPREAD"  # 价差

    #  基本面类
    INVENTORY = "INVENTORY"  # 库存
    WARRANT = "WARRANT"  # 仓单
    PRODUCTION = "PRODUCTION"  # 产量
    PROFIT = "PROFIT"  # 利润
    CONSUMPTION = "CONSUMPTION"  # 消费量/表观需求
    IMPORT = "IMPORT"  # 进口量
    EXPORT = "EXPORT"  # 出口量
    CAPACITY = "CAPACITY"  # 产能
    CAPACITY_UTILIZATION = "CAPACITY_UTILIZATION"  # 开工率/产能利用率
    BASIS = "BASIS"  # 基差

    @classmethod
    def allowed_columns(cls):
        return {v.value.upper() for v in cls}


if __name__ == "__main__":
    v = Variable
    lst = v.allowed_columns()
    print(lst)