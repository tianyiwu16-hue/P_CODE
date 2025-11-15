import numpy as np
y1=np.random.normal(10,5,100)
y2=np.random.normal(10,20,100)

print(np.quantile(y1,0.05))
print(np.quantile(y2,0.08))

np.mean(y1)
np.mean(y2)



import numpy as np
from scipy.stats import chi2_contingency

# 假设这是你的2x3列联表数据
observed_values = np.array([
    [19,15,24 ],  # 第一行观测值
    [16,18,16 ]   # 第二行观测值
])

# 计算卡方值、p值、自由度和期望频数
chi2, p, dof, expected = chi2_contingency(observed_values)

print("Chi-Square Test Statistic: ", chi2)
print("P-value: ", p)
print("Degrees of Freedom: ", dof)
print("Expected Frequencies: \n", expected)



#方法一
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
wine=pd.read_csv("D:/学习工作文件夹/原神表格文件/wine.csv")
y1=wine[wine['Y']==7]['X1']
y2=wine[wine['Y']==8]['X1']
y3=wine[wine['Y']==9]['X1']
f_oneway(y1,y2,y3)
#方法二
import numpy as np
import pandas as pd
from scipy.stats import f_oneway

# 读取CSV文件
wine = pd.read_csv("D:/学习工作文件夹/原神表格文件/wine.csv")
# 提取不同组的数据
y1 = wine[wine['Y'] == 7]['X1']
y2 = wine[wine['Y'] == 8]['X1']
y3 = wine[wine['Y'] == 9]['X1']

# 执行ANOVA测试
f_statistic, p_value = f_oneway(y1, y2, y3)

# 输出结果
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

# 检查每个组的数据是否为空
if y1.empty or y2.empty or y3.empty:
    print("Warning: One or more groups are empty.")
else:
    print("All groups have data.")






import numpy as np
from scipy.stats import chi2_contingency
table=np.array([40,55,60,73]).reshape(2,2)
chi2_contingency(table)

