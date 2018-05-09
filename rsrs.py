import numpy as np
import pandas as pd


def get_RSRS(high_array, low_array, n=20, m=400):
    # high_array = values.high.values[-(n+m-1):]
    # low_array = values.low.values[-(n+m-1):]
    scores = np.zeros(m)  # 各期斜率
    for i in range(m):
        high = high_array[i:i+30]
        low = low_array[i:i+30]

        # 计算单期斜率
        x = low  # low作为自变量
        X = sm.add_constant(x)  # 添加常数变量
        y = high  # high作为因变量
        model = sm.OLS(y, X)  # 最小二乘法
        results = model.fit()
        score = results.params[1]
        scores[i] = score

        # 记录最后一期的Rsquared(可决系数)
        if i == m-1:
            R_squared = results.rsquared

    # 最近期的标准分
    z_score = (scores[-1]-scores.mean())/scores.std()

    # RSRS得分
    RSRS_socre = z_score*R_squared

    return RSRS_socre
