import numpy as np

def generate_random_data(mean=50, size=10, scale=10):
    """
    随机生成一组数据。
    
    参数：
    - mean: 数据的期望均值。
    - size: 数据组大小。
    - scale: 用于控制数据范围的尺度参数。
    
    返回：
    - 随机生成的数据数组。
    """
    data = np.random.normal(loc=mean, scale=scale, size=size)
    return data

# 设定随机种子，便于结果复现
np.random.seed(20250324)

# 生成两组随机数据
data1 = generate_random_data(mean=50, size=10, scale=10)
data2 = generate_random_data(mean=50, size=10, scale=8) # 调整scale可以调整标准差，使其与第一组接近

# 计算并输出各组数据的统计信息
print("数据组 1：", data1)
print("数据组 1 均值：", np.mean(data1))
print("数据组 1 标准差：", np.std(data1))

print("\n数据组 2：", data2)
print("数据组 2 均值：", np.mean(data2))
print("数据组 2 标准差：", np.std(data2))




import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# 假设 creditcard 数据集已加载为 DataFrame
# 示例数据集加载代码（替换为您的实际数据集）
hs300_basic = pd.read_csv("D:/原神文件/hs300_basic.csv")

# （1）计算连续变量的峰度和偏度
def calculate_skew_kurtosis(df):
    """
    计算 DataFrame 中所有连续变量的偏度和峰度。
    """
    continuous_vars = df.select_dtypes(include=[np.number])  # 筛选数值型列
    skewness = continuous_vars.apply(skew)  # 计算偏度
    kurtosis_values = continuous_vars.apply(kurtosis)  # 计算峰度
    return pd.DataFrame({
        'Skewness': skewness,
        'Kurtosis': kurtosis_values
    })

# （2）计算连续变量的各个百分位数
def calculate_percentiles(df, percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]):
    """
    计算 DataFrame 中所有连续变量的指定百分位数。
    """
    continuous_vars = df.select_dtypes(include=[np.number])  # 筛选数值型列
    percentiles_df = continuous_vars.quantile(q=percentiles)
    return percentiles_df

# （3）根据峰度和偏度判断是否服从正态分布
def check_normality(skewness, kurtosis):
    """
    根据偏度和峰度判断变量是否接近正态分布。
    - 偏度接近 0 表示对称性良好。
    - 峰度接近 0 表示尾部厚度与正态分布相似。
    """
    is_normal = (abs(skewness) < 0.5) & (abs(kurtosis) < 0.5)
    return is_normal

# 主程序
if __name__ == "__main__":
    # 假设 creditcard 数据集已经加载
    # creditcard = pd.read_csv('creditcard.csv')
    
    # （1）计算峰度和偏度
    skew_kurtosis = calculate_skew_kurtosis(hs300_basic)
    print("峰度和偏度：\n", skew_kurtosis)
    
    # （2）计算百分位数
    percentiles = calculate_percentiles(hs300_basic)
    print("\n百分位数：\n", percentiles)
    
    # （3）判断是否服从正态分布
    normality_check = skew_kurtosis.apply(lambda row: check_normality(row['Skewness'], row['Kurtosis']), axis=1)
    print("\n是否服从正态分布：\n", normality_check)









import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# （1）计算连续变量的峰度和偏度
def calculate_skew_kurtosis(df):
    """
    计算 DataFrame 中所有连续变量的偏度和峰度。
    """
    continuous_vars = df.select_dtypes(include=[np.number])  # 筛选数值型列
    skewness = continuous_vars.apply(skew)  # 计算偏度
    kurtosis_values = continuous_vars.apply(kurtosis)  # 计算峰度
    return pd.DataFrame({
        'Skewness': skewness,
        'Kurtosis': kurtosis_values
    })

# （2）计算连续变量的各个百分位数
def calculate_percentiles(df, percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]):
    """
    计算 DataFrame 中所有连续变量的指定百分位数。
    """
    continuous_vars = df.select_dtypes(include=[np.number])  # 筛选数值型列
    percentiles_df = continuous_vars.quantile(q=percentiles)
    return percentiles_df

# （3）根据峰度和偏度判断是否服从正态分布
def check_normality(skewness, kurtosis):
    """
    根据偏度和峰度判断变量是否接近正态分布。
    - 偏度接近 0 表示对称性良好。
    - 峰度接近 0 表示尾部厚度与正态分布相似。
    """
    is_normal = (abs(skewness) < 0.5) & (abs(kurtosis) < 0.5)
    return is_normal

# 主程序
if __name__ == "__main__":
    # 加载数据
    try:
        # 如果有实际数据文件
        hs300_basic = pd.read_csv(r"D:\原神文件\hs300_basic.csv")
    except FileNotFoundError:
        print("未找到文件，使用随机数据模拟...")
        # 如果没有实际数据文件，使用随机数据模拟
        np.random.seed(20250324)
        hs300_basic = pd.DataFrame({
            'Var1': np.random.normal(loc=0, scale=1, size=1000),
            'Var2': np.random.exponential(scale=1, size=1000),
            'Var3': np.random.uniform(low=-1, high=1, size=1000)
        })

    # （1）计算峰度和偏度
    skew_kurtosis = calculate_skew_kurtosis(hs300_basic)
    print("峰度和偏度：\n", skew_kurtosis)

    # （2）计算百分位数
    percentiles = calculate_percentiles(hs300_basic)
    print("\n百分位数：\n", percentiles)

    # （3）判断是否服从正态分布
    normality_check = skew_kurtosis.apply(lambda row: check_normality(row['Skewness'], row['Kurtosis']), axis=1)
    print("\n是否服从正态分布：\n", normality_check)





import pandas as pd

# 假设 creditcard 数据集已加载为 DataFrame
# 示例数据集加载代码（替换为您的实际数据集）
creditcard = pd.read_csv("D:\原神文件\creditcard.csv")

# 示例：用随机数据模拟 creditcard 数据集
#np.random.seed(20250324)
#creditcard = pd.DataFrame({
#   'LIMIT_BAL': np.random.randint(10000, 50000, size=1000),
#    'EDUCATION': np.random.choice(['高中', '本科', '硕士', '博士'], size=1000),
 #   'SEX': np.random.choice(['男', '女'], size=1000),
  #  'MARRIAGE': np.random.choice(['未婚', '已婚', '其他'], size=1000)
#})

# （1）按照教育水平和性别分组，计算 LIMIT_BAL 的均值和标准差
grouped_stats = creditcard.groupby(['EDUCATION', 'SEX'])['LIMIT_BAL'].agg(['mean', 'std']).reset_index()
print("按教育水平和性别分组的LIMIT_BAL均值和标准差：\n", grouped_stats)

# （2）建立数据透视表，计算不同性别、婚姻状况以及教育水平下 LIMIT_BAL 的均值
pivot_table = pd.pivot_table(creditcard, values='LIMIT_BAL', index=['SEX', 'MARRIAGE'], columns=['EDUCATION'], aggfunc='mean')
print("\n数据透视表（不同性别、婚姻状况以及教育水平下的LIMIT_BAL均值）：\n", pivot_table)




import pandas as pd
import numpy as np

# （1）将 hs300 数据集导入 Python，并建立时间索引
# 假设 hs300 数据集已加载为 DataFrame
# 示例数据集加载代码（替换为您的实际数据集）
hs300_basic= pd.read_csv("D:\原神文件\hs300_basic.csv")

# 示例：用随机数据模拟 hs300 数据集
#np.random.seed(20250324)
#dates = pd.date_range(start='2023-01-01', periods=100, freq='D')  # 生成连续日期
#hs300 = pd.DataFrame({
 #   'Date': dates,
  #  'pct_chg': np.random.normal(loc=0.0, scale=1.0, size=100)  # 模拟每日涨跌幅
#})

# 设置日期为索引
hs300_basic.set_index('amt', inplace=True)
print("带有时间索引的 hs300 数据：\n", hs300_basic.head())

# （2）计算 pct_chg 的均值、中位数、标准差
mean_pct_chg = hs300_basic['pct_chg'].mean()
median_pct_chg = hs300_basic['pct_chg'].median()
std_pct_chg = hs300_basic['pct_chg'].std()

print("\npct_chg 的描述性统计：")
print(f"均值: {mean_pct_chg:.4f}")
print(f"中位数: {median_pct_chg:.4f}")
print(f"标准差: {std_pct_chg:.4f}")

# （3）使用滑窗技术计算 pct_chg 的均值、中位数、标准差
window_size = 10  # 滑窗大小（例如 10 天）

# 滑窗均值
rolling_mean = hs300_basic['pct_chg'].rolling(window=window_size).mean()
# 滑窗中位数
rolling_median = hs300_basic['pct_chg'].rolling(window=window_size).median()
# 滑窗标准差
rolling_std = hs300_basic['pct_chg'].rolling(window=window_size).std()

# 将结果添加到 DataFrame 中
hs300_basic['Rolling_Mean'] = rolling_mean
hs300_basic['Rolling_Median'] = rolling_median
hs300_basic['Rolling_Std'] = rolling_std

print("\n带有滑窗统计量的 hs300 数据：\n", hs300_basic.tail())


# 检查数据集的列名
print(hs300_basic.columns)























