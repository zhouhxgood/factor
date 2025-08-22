from statsmodels.tsa.stattools import adfuller


def adf_test(series, save_path="/stats_test/adf_result.txt"):
    """
    自回归过程如果存在单位根则说明是一个随机游走，即差分是一个白噪声
    """
    result = adfuller(series, autolag="AIC")
    output_lines = [f"ADF Statistic: {result[0]}", f"p-value: {result[1]}", f"Used lags: {result[2]}",
                    f"Number of observations: {result[3]}", "Critical Values:"]
    for key, value in result[4].items():
        output_lines.append(f"   {key}: {value}")
    output_lines.append(f"IC Best: {result[5]}")

    passed_levels = []
    for level, value in result[4].items():
        if result[0] < value:
            passed_levels.append(level)

    if passed_levels:
        conclusion = f"Conclusion: The series is stationary at the following significance levels: {', '.join(passed_levels)}."
    else:
        conclusion = "Conclusion: The series is non-stationary at the 1%, 5%, and 10% significance levels."

    output_lines.append(conclusion)
    # 保存到文件
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))