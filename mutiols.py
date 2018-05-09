import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# 传入：serises和表达式
# 返回：serises，intercept：常数项，其他：系数


def multiols(datasourse, formula):
    est = smf.ols(formula=formula, data=datasourse).fit()
    return est.params


# 示例
# ds = pd.read_csv(
#     'http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

# print(multiols(ds, 'sales ~ TV+radio+newspaper'))

# ds= pd.read_csv('./multivariate_periods.csv', names=['a', 'b', 'c', 'd', 'e'])
# print(multiols(ds, 'e ~ a+b+c+d'))
