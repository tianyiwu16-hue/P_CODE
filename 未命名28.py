import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, probplot

print("✅ 所有库加载成功！")

# 定义特征类型
discrete_feat = ['Gender', 'University_Year', 'Major', 'Weekday_Sleep_Start_Period']
continuous_feat = ['Age', 'Study_Hours', 'Exercise_Frequency', 'Screen_Time']

# 颜色映射
from matplotlib.cm import Reds
cmap = Reds





data = pd.read_csv("D:\大学生生活方式数据\student_sleep_patterns.csv")
data.info()




for feat in discrete_feat:
    groups = data.groupby(feat)['Sleep_Quality'].apply(list)
    if len(groups)>1:
        _, p_value = kruskal(*groups)
    else:
        p_value = 0.0
    
    plt.figure(figsize=(8,3))
    plt.subplot(121)
    crosstab = data.groupby(['Sleep_Quality',feat]).size().unstack()
    crosstab.plot(kind='bar',stacked=True,cmap='rocket',ax=plt.gca())
    plt.box(False)

    plt.subplot(122)
    sns.lineplot(data=data.groupby(feat)['Sleep_Quality'].mean(),color='tomato',linewidth=3,marker='o',markersize=10)
    plt.box(False)
    
    plt.suptitle(f'Sleep Quality by {feat} [P Value :{p_value:0.2f}]')
    plt.tight_layout()
    plt.show()




for feat in continuous_feat:
    counts, bins = np.histogram(data[feat].dropna(), bins=50)
    norm = plt.Normalize(counts.mean()-counts.std(), counts.max())
    
    plt.figure(figsize=(8,3))
    
    plt.subplot(1,2,1)
    hist = sns.histplot(x=data[feat].dropna(),bins=50,kde=True)
    for patch in hist.patches:
        height = patch.get_height()  
        patch.set_facecolor(cmap(norm(height)*0.75))
    plt.box(False)
    plt.ylabel('')
    
    plt.subplot(1,2,2)
    probplot(data[feat],plot=plt)
    plt.box(False)
    plt.ylabel('')
    
    plt.suptitle(feat+' Distribution')
    plt.tight_layout()
    plt.show()



for feat in continuous_feat:
    corr = data[['Sleep_Quality']].corrwith(data[feat],method='spearman')
    plt.figure(figsize=(8,3))
    
    plt.subplot(121)
    sns.kdeplot(data=data, x='Sleep_Quality', y=feat, fill=True,levels=20, cmap="rocket_r")
    plt.box(False)

    plt.subplot(122)
    sns.boxplot(data=data,x='Sleep_Quality',y=feat,palette='rocket_r')
    plt.box(False)
    
    plt.suptitle(f'Sleep_Quality X {feat} [Correlation:{corr.values[0]:0.2f}]')
    plt.tight_layout()
    plt.show()




data['Gender'] = data['Gender'].replace({'Male':1,'Female':2,'Other':3}).astype(int)
data['University_Year'] = data['University_Year'].replace({'1st Year':1,'2nd Year':2,'3rd Year':3,'4th Year':4}).astype(int)
data['Weekday_Sleep_Start_Period'] = data['Weekday_Sleep_Start_Period'].replace({ 'Early Morning': 0, 'Morning': 1, 'Afternoon': 2, 'Night': 3, 'Late Night': 4 }).astype(int)
data['Weekend_Sleep_Start_Period'] = data['Weekend_Sleep_Start_Period'].replace({ 'Early Morning': 0, 'Morning': 1, 'Afternoon': 2, 'Night': 3, 'Late Night': 4 }).astype(int)

data['Weekday_Sleep_End_Period'] = data['Weekday_Sleep_End_Period'].replace({ 'Early Morning': 0, 'Morning': 1, 'Afternoon': 2, 'Night': 3, 'Late Night': 4 }).astype(int)
data['Weekend_Sleep_End_Period'] = data['Weekend_Sleep_End_Period'].replace({ 'Early Morning': 0, 'Morning': 1, 'Afternoon': 2, 'Night': 3, 'Late Night': 4 }).astype(int)





plt.figure(figsize=(8,8))
sns.heatmap(data.corr(method='spearman'),annot=True, cmap='rocket',cbar=False,fmt='0.2f')
plt.show()

















