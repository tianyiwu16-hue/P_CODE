# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:16:51 2024

@author: 20997
"""

#直方图
plt.hist(data,bins=[800,900,1000,1100,1200,1300,1400,
                    1500,1600,1700,1800,1900])
#累计直方图
plt.hist(data,bins=[800,900,1000,1100,1200,1300,1400,
                    1500,1600,1700,1800,1900],cumulative=True)
#频率密度直方图
plt.hist(data,bins=[800,900,1000,1100,1200,1300,1400,
                    1500,1600,1700,1800,1900],density=True)
#绘制分布曲线图
import seaborn as sns
sns.kdeplot(data)
