import pandas as pd

# 设置文件路径
csv_file_path = r"D:\原神文件\PM25.csv"  # 使用原始字符串避免转义问题

# 假设 pm25 是一个包含 PM2.5 数据的 DataFrame
# 示例数据结构：
# pm25 = pd.DataFrame({
#     'pm2.5': [12, 25, 30, 40, 50, 60, 70, 80, 90, 100]
# })

# 等频分箱：将 pm2.5 列分为 5 个箱子，标签为 0~4
PM25['binned'] = pd.qcut(PM25['pm2.5'], q=5, labels=range(5))

# 输出结果
print(PM25)



import pandas as pd

# 设置文件路径
csv_file_path = r"D:\原神文件\PM25.csv"  # 使用原始字符串避免转义问题

# 读取 CSV 文件
PM25 = pd.read_csv(csv_file_path)

# 检查数据集是否成功加载
print("原始数据：")
print(PM25.head())

# 等频分箱：将 pm2.5 列分为 5 个箱子，标签为 0~4
# 注意：确保列名 'pm2.5' 与 CSV 文件中的列名一致
PM25['binned'] = pd.qcut(PM25['pm2.5'], q=5, labels=range(5))

# 输出结果
print("\n分箱后的数据：")
print(PM25)




import pandas as pd

# 设置文件路径
csv_file_path = r"D:\原神文件\hs300_basic.csv"  # 使用原始字符串避免转义问题

# 读取 CSV 文件
PM25 = pd.read_csv(csv_file_path)

# 检查数据集是否成功加载
print("原始数据：")
print(hs300_basic.head())
# 假设 hs300 是一个包含历史数据的 DataFrame
# 示例数据结构：
# hs300 = pd.DataFrame({
#     'date': ['2022-01-01', '2022-01-02', '2022-02-01', '2022-02-02'],
#     'pct_chg': [1.2, -0.5, 2.0, -1.0]
# })

# 确保日期列为 datetime 类型
hs300_basic['date'] = pd.to_datetime(hs300_basic['date'])

# 筛选出 2022 年的数据
hs300_basic_2022 = hs300_basic[(hs300_basic['date'].dt.year == 2022)]

# 按月份分组并计算每月 pct_chg 的平均值
monthly_avg = hs300_basic_2022.groupby(hs300_basic_2022['date'].dt.month)['pct_chg'].mean()

# 找到平均值最大的月份
max_month = monthly_avg.idxmax()
max_avg_value = monthly_avg.max()

# 输出结果
print(f"2022 年中，{max_month} 月的 pct_chg 平均值最大，为 {max_avg_value:.2f}%")






import pandas as pd

# 设置文件路径
csv_file_path = r"D:\原神文件\hs300_basic.csv"  # 使用原始字符串避免转义问题

# 读取 CSV 文件
hs300_basic = pd.read_csv(csv_file_path)

# 检查数据集是否成功加载
print("原始数据：")
print(hs300_basic.head())

# 确保日期列为 datetime 类型
hs300_basic['date'] = pd.to_datetime(hs300_basic['date'])

# 筛选出 2010 年的数据
hs300_basic_2010 = hs300_basic[(hs300_basic['date'].dt.year == 2010)]

# 计算 2010 年 pct_chg 的平均值
average_pct_chg_2010 = hs300_basic_2010['pct_chg'].mean()

# 输出结果
if pd.notna(average_pct_chg_2010):  # 检查是否有有效值
    print(f"2010 年全年的 pct_chg 平均值为: {average_pct_chg_2010:.4f}%")
else:
    print("2010 年没有有效的 pct_chg 数据")


import pandas as pd
import numpy as np

# 设置文件路径
csv_file_path = r"D:\原神文件\creditcard.csv"  # 使用原始字符串避免转义问题

# 读取 CSV 文件
creditcard = pd.read_csv(csv_file_path)

# 检查数据集是否成功加载
print("原始数据：")
print(creditcard.head())

# 确保 AGE 列存在
if 'AGE' not in creditcard.columns:
    raise ValueError("数据集中没有找到 AGE 列，请检查列名是否正确")

# 随机选择 100 个样本点的索引
np.random.seed(42)  # 固定随机种子以确保结果可重复
random_indices = np.random.choice(creditcard.index, size=100, replace=False)

# 将选中的样本点的 AGE 置换为缺失值
creditcard.loc[random_indices, 'AGE'] = np.nan

# 计算置换后 AGE 的平均值
average_age_after_missing = creditcard['AGE'].mean()

# 输出结果
print(f"置换后 AGE 的平均数是: {average_age_after_missing:.3f}")



import pandas as pd

# 设置文件路径
csv_file_path = r"D:\原神文件\PM25.csv"  # 使用原始字符串避免转义问题

# 读取 CSV 文件
PM25 = pd.read_csv(csv_file_path)

# 检查数据集是否成功加载
print("原始数据：")
print(PM25.head())

# 验证说法 1：风向为 NW 的比例为 0.323
wind_direction_counts = PM25['wind_direction'].value_counts(normalize=True)
nw_ratio = wind_direction_counts.get('NW', 0)  # 获取 NW 的比例，默认为 0
print(f"风向为 NW 的比例为: {nw_ratio:.3f}")

# 验证说法 2：平均来看，2013 年（year=2013）的 pm2.5 水平最高
PM25['year'] = pd.to_datetime(PM25['date']).dt.year  # 提取年份
yearly_PM25_avg = PM25.groupby('year')['pm2.5'].mean()
max_year = yearly_PM25_avg.idxmax()  # 找到平均值最大的年份
print(f"平均 pm2.5 水平最高的年份是: {max_year}，其平均值为: {yearly_PM25_avg[max_year]:.2f}")

# 验证说法 3：平均来看，1 点（hour=1）的 pm2.5 水平最高
PM25['hour'] = pd.to_datetime(PM25['time']).dt.hour  # 提取小时
hourly_PM25_avg = PM25.groupby('hour')['pm2.5'].mean()
max_hour = hourly_PM25_avg.idxmax()  # 找到平均值最大的小时
print(f"平均 pm2.5 水平最高的小时是: {max_hour}，其平均值为: {hourly_PM25_avg[max_hour]:.2f}")

# 验证说法 4：风向为 SE 的比例为 0.349
se_ratio = wind_direction_counts.get('SE', 0)  # 获取 SE 的比例，默认为 0
print(f"风向为 SE 的比例为: {se_ratio:.3f}")

















# 假设 hs300 是一个包含历史数据的 DataFrame
# 示例数据结构：
# hs300 = pd.DataFrame({
#     'date': ['2025-03-01', '2025-03-02', '2025-03-03'],
#     'pct_chg': [1.2, -0.5, 8.0]
# })

# 计算 pct_chg 的均值和标准差
mean_pct_chg = hs300_basic['pct_chg'].mean()
std_pct_chg = hs300_basic['pct_chg'].std()

# 定义上下界
upper_bound = mean_pct_chg + 3 * std_pct_chg
lower_bound = mean_pct_chg - 3 * std_pct_chg

# 筛选出超过均值 ± 3 倍标准差的交易日
outliers = hs300_basic[(hs300_basic['pct_chg'] > upper_bound) | (hs300_basic['pct_chg'] < lower_bound)]

# 统计符合条件的交易日数量
num_outliers = len(outliers)

# 输出结果
print(f"pct_chg 超过均值 ± 3 倍标准差的交易日有 {num_outliers} 个")
























# 导入 CSV 文件
try:
    df = pd.read_csv(csv_file_path)
    print("数据集成功导入！")
    print("前 5 行数据：")
    print(df.head())  # 打印前 5 行数据以确认导入成功
except Exception as e:
    print(f"导入数据时出错：{e}")
    
    
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# 设置文件路径
csv_file_path = r"D:\原神文件\PM25.csv" 

# （1）导入数据集
df = pd.read_csv(csv_file_path)
print("数据集成功导入！")
print("前 5 行数据：")
print(df.head())

# （2）随机生成 100 个空值
random_indices = np.random.choice(df.index, size=100, replace=False)
df['LIMIT_BAL'].iloc[random_indices] = np.nan
print("\n随机生成的空值：")
print(df['LIMIT_BAL'].isnull().sum())

# （3）删除空值所在行
df_dropped = df.dropna(subset=['LIMIT_BAL'])
print("\n删除空值后的数据：")
print(df_dropped.head())
print(f"删除后剩余记录数：{len(df_dropped)}")

# （4）使用平均数或中位数填充缺失值
df = pd.read_csv(csv_file_path)  # 重新加载数据
random_indices = np.random.choice(df.index, size=100, replace=False)
df['LIMIT_BAL'].iloc[random_indices] = np.nan

# 使用平均数填充
mean_value = df['LIMIT_BAL'].mean()
df_mean_filled = df.copy()
df_mean_filled['LIMIT_BAL'].fillna(mean_value, inplace=True)
print("\n使用平均数填充缺失值后的数据：")
print(df_mean_filled.head())

# 使用中位数填充
median_value = df['LIMIT_BAL'].median()
df_median_filled = df.copy()
df_median_filled['LIMIT_BAL'].fillna(median_value, inplace=True)
print("\n使用中位数填充缺失值后的数据：")
print(df_median_filled.head())

# （5）基于最近邻法填充缺失值
df = pd.read_csv(csv_file_path)  # 重新加载数据
random_indices = np.random.choice(df.index, size=100, replace=False)
df['LIMIT_BAL'].iloc[random_indices] = np.nan

# 确保所有相关列均为数值型
numeric_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'LIMIT_BAL']
df_for_knn = df[numeric_columns].copy()

# 标准化数据
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_for_knn), columns=df_for_knn.columns)

# 初始化 KNNImputer
imputer = KNNImputer(n_neighbors=5)

# 使用 KNNImputer 填充缺失值
df_filled_scaled = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df_scaled.columns)

# 反标准化回原始尺度
df_filled = pd.DataFrame(scaler.inverse_transform(df_filled_scaled), columns=df_scaled.columns)

# 更新原数据框中的 LIMIT_BAL 列
df['LIMIT_BAL'] = df_filled['LIMIT_BAL']
print("\n基于最近邻法填充缺失值后的数据：")
print(df.head())






import pandas as pd

# 设置文件路径
csv_file_path = r"D:\原神文件\hs300_basic.csv"

# （1）加载数据集
try:
    df = pd.read_csv(csv_file_path)
    print("数据集成功导入！")
except Exception as e:
    print(f"导入数据时出错：{e}")
    exit()

# （2）检查列名是否包含 'date' 和 'pct_chg'
if 'date' not in df.columns or 'pct_chg' not in df.columns:
    print("错误：数据集中缺少 'date' 或 'pct_chg' 列，请检查列名是否正确。")
    exit()

# （3）将 'date' 列转换为日期格式
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')

# （4）筛选出 2010 年的数据
df_2010 = df[(df['date'] >= '2010-01-01') & (df['date'] <= '2010-12-31')]

# （5）计算 2010 年 'pct_chg' 的平均值
if not df_2010.empty:
    avg_pct_chg = df_2010['pct_chg'].mean()
    print(f"\n2010 年全年的 pct_chg 平均值为：{avg_pct_chg:.2f}%")
else:
    print("\n未找到 2010 年的数据，请检查数据集内容。")



import pandas as pd

# 设置文件路径
csv_file_path = r"D:\原神文件\hs300_basic.csv"

# （1）加载数据集
try:
    df = pd.read_csv(csv_file_path)
    print("数据集成功导入！")
except Exception as e:
    print(f"导入数据时出错：{e}")
    exit()

# （2）检查列名是否包含 'pct_chg'
if 'pct_chg' not in df.columns:
    print("错误：数据集中缺少 'pct_chg' 列，请检查列名是否正确。")
    exit()

# （3）计算四分位数和四分位距
Q1 = df['pct_chg'].quantile(0.25)  # 第一四分位数
Q3 = df['pct_chg'].quantile(0.75)  # 第三四分位数
IQR = Q3 - Q1  # 四分位距

# 设定阈值
threshold = Q3 + 2 * IQR

print(f"\n第一四分位数 (Q1): {Q1:.2f}")
print(f"第三四分位数 (Q3): {Q3:.2f}")
print(f"四分位距 (IQR): {IQR:.2f}")
print(f"阈值 (Q3 + 2 * IQR): {threshold:.2f}")

# （4）筛选超过阈值的交易日
outliers = df[df['pct_chg'] > threshold]

# （5）输出结果
print(f"\npct_chg 超过阈值的交易日数量：{len(outliers)}")
print("\n符合条件的交易日数据：")
print(outliers)


import pandas as pd

# 设置文件路径
csv_file_path = r"D:\原神文件\PM25.csv"

# （1）加载数据集
try:
    df = pd.read_csv(csv_file_path)
    print("数据集成功导入！")
except Exception as e:
    print(f"导入数据时出错：{e}")
    exit()

# （2）检查列名是否包含 'cbwd'
if 'cbwd' not in df.columns:
    print("错误：数据集中缺少 'cbwd' 列，请检查列名是否正确。")
    exit()

# （3）查看 cbwd 列的唯一值
unique_values = df['cbwd'].unique()
print(f"\ncbwd 列的唯一值：{unique_values}")
print(f"唯一值数量：{len(unique_values)}")

# （4）对 cbwd 列进行虚拟编码（drop_first=True）
df_encoded = pd.get_dummies(df, columns=['cbwd'], drop_first=True)

# （5）输出结果
print(f"\n虚拟编码后的新数据集列数：{df_encoded.shape[1]}")
print("新数据集的列名：")
print(df_encoded.columns)

import pandas as pd
import numpy as np

# 设置文件路径
csv_file_path = r"D:\原神文件\creditcard.csv"

# （1）加载数据集
try:
    df = pd.read_csv(csv_file_path)
    print("数据集成功导入！")
except Exception as e:
    print(f"导入数据时出错：{e}")
    exit()

# （2）检查列名是否包含 'AGE'
if 'AGE' not in df.columns:
    print("错误：数据集中缺少 'AGE' 列，请检查列名是否正确。")
    exit()

# （3）保存原始 AGE 列的平均值（用于对比）
original_mean_age = df['AGE'].mean()
print(f"\n原始 AGE 列的平均值：{original_mean_age:.3f}")

# （4）随机选择 100 个样本点并将 AGE 置换为缺失值
np.random.seed(42)  # 设置随机种子以确保结果可复现
random_indices = np.random.choice(df.index, size=100, replace=False)  # 随机选择 100 个索引
df.loc[random_indices, 'AGE'] = np.nan  # 将这些索引对应的 AGE 置为 NaN

# （5）计算置换后的平均值
new_mean_age = df['AGE'].mean()  # 忽略缺失值
print(f"\n置换后 AGE 列的平均值：{new_mean_age:.3f}")

# （6）输出缺失值数量（验证）
missing_count = df['AGE'].isnull().sum()
print(f"AGE 列中缺失值的数量：{missing_count}")




import pandas as pd
# 设置文件路径
csv_file_path = r"D:\原神文件\hs300_basic.csv"

# 假设 hs300 是一个包含历史数据的 DataFrame
# 示例数据结构：
# hs300 = pd.DataFrame({
#     'date': ['2025-03-01', '2025-03-02', '2025-03-03'],
#     'pct_chg': [1.2, -0.5, 0.8]
# })

# 筛选出上涨交易日（pct_chg > 0）
up_days = hs300_basic[hs300_basic['pct_chg'] > 0]

# 计算上涨交易日的 pct_chg 平均值
average_up_pct_chg = up_days['pct_chg'].mean()

# 输出结果
print(f"上涨交易日的 pct_chg 平均值为: {average_up_pct_chg:.3f}%")






















import pandas as pd

# 设置文件路径
csv_file_path = r"D:\原神文件\PM25.csv"  # 使用原始字符串避免转义问题

# 读取 CSV 文件
pm25 = pd.read_csv(csv_file_path)

# 去除缺失值
pm25_cleaned = pm25.dropna()

# 提取 pm2.5 列
pm2_5_values = pm25_cleaned['pm2.5']

# 0-1 标准化公式：(x - min) / (max - min)
pm2_5_normalized = (pm2_5_values - pm2_5_values.min()) / (pm2_5_values.max() - pm2_5_values.min())

# 获取索引位置为 100 的标准化值
normalized_value_at_100 = pm2_5_normalized.iloc[100]

# 输出结果
print(f"索引位置为 100 的 pm2.5 标准化数值为: {normalized_value_at_100:.4f}")



import pandas as pd
import numpy as np

# 1. 读取CSV文件
file_path = r"D:\原神文件\PM25.csv"
data = pd.read_csv(file_path)

# 2. 检查PM2.5列名
PM25_col = 'pm2.5'  # 如果列名不同，请修改（如 'PM2.5'）
if PM25_col not in data.columns:
    raise ValueError(f"列名 '{PM25_col}' 不存在于数据中，请检查列名。")
print("数据前5行：\n", data.head())

# 3. 去除缺失值
clean_data = data.dropna(subset=[PM25_col]).copy()

# 4. 等频分箱（5箱），标签设为0~4
clean_data['bin_label'], bins = pd.qcut(
    clean_data[PM25_col],
    q=5,                # 分5箱
    labels=False,       # 标签设为0~4
    retbins=True,       # 返回分箱边界
    duplicates='drop'   # 如果数据有重复边界，自动调整分箱
)

# 5. 查看分箱边界
print("\n等频分箱边界（5箱）：")
for i in range(len(bins) - 1):
    print(f"箱 {i}: {bins[i]:.2f} ~ {bins[i+1]:.2f}")

# 6. 重置索引以确保loc定位准确
clean_data = clean_data.reset_index(drop=True)

# 7. 查询指定索引的分箱结果
indices = [100, 300, 10000, 30000]
print("\n查询指定索引的分箱结果：")
for idx in indices:
    if idx < len(clean_data):
        PM25_value = clean_data.iloc[idx][PM25_col]
        bin_label = clean_data.iloc[idx]['bin_label']
        print(f"索引 {idx}: PM2.5值 = {PM25_value:.2f}, 分箱标签 = {bin_label}")
    else:
        print(f"索引 {idx}: 超出数据范围（清洗后数据共 {len(clean_data)} 行）")







import pandas as pd
import numpy as np

# 1. 读取CSV文件
file_path = r"D:\原神文件\PM25.csv"
data = pd.read_csv(file_path)

# 2. 检查PM2.5列名（假设列名为'PM25'或'PM2.5'）
PM25_col = 'pm2.5'  # 如果列名不同，请修改（如 'PM2.5'）
print("数据前5行：\n", data.head())

# 3. 去除缺失值
clean_data = data.dropna(subset=[PM25_col]).copy()  # 使用copy避免SettingWithCopyWarning

# 4. 等频分箱（5箱），标签设为0~4
clean_data['bin_label'], bins = pd.qcut(
    clean_data[PM25_col],
    q=5,                # 分5箱
    labels=False,       # 标签设为0~4
    duplicates='drop'   # 如果数据有重复边界，自动调整分箱
)

# 5. 查看分箱边界（等频分箱的5个分位数）
quantiles = np.percentile(clean_data[PM25_col], [0, 20, 40, 60, 80, 100])
print("\n等频分箱边界（5箱）：")
for i in range(5):
    print(f"箱 {i}: {quantiles[i]:.2f} ~ {quantiles[i+1]:.2f}")

# 6. 重置索引以确保loc定位准确
clean_data = clean_data.reset_index(drop=True)

# 7. 查询指定索引的分箱结果
indices = [100, 300, 10000, 30000]
print("\n查询指定索引的分箱结果：")
for idx in indices:
    try:
        PM25_value = clean_data.loc[idx, PM25_col]
        bin_label = clean_data.loc[idx, 'bin_label']
        print(f"索引 {idx}: PM2.5值 = {PM25_value:.2f}, 分箱标签 = {bin_label}")
    except KeyError:
        print(f"索引 {idx}: 超出数据范围（清洗后数据共 {len(clean_data)} 行）")
clean_data = clean_data.reset_index(drop=True)  # 确保索引从0开始连续     











import pandas as pd
import numpy as np

# 1. 读取CSV文件
file_path = r"D:\原神文件\PM25.csv"
data = pd.read_csv(file_path)

# 2. 检查PM2.5列名（假设列名为'PM2.5'或'pm2.5'）
print("数据前5行：\n", data.head())
print("数据集的列名：", data.columns)

PM25_col = 'pm2.5'  # 如果列名不同，请修改（如 'pm2.5'）
if PM25_col not in data.columns:
    raise ValueError(f"列名 '{PM25_col}' 不存在，请检查列名拼写")

# 3. 去除缺失值
clean_data = data.dropna(subset=[PM25_col]).copy()  # 使用copy避免SettingWithCopyWarning

# 4. 等频分箱（5箱），标签设为0~4
clean_data['bin_label'] = pd.qcut(
    clean_data[PM25_col],
    q=5,                # 分5箱
    labels=False,       # 标签设为0~4
    duplicates='drop'   # 如果数据有重复边界，自动调整分箱
)

# 获取分箱边界
bins = pd.qcut(clean_data[PM25_col], q=5, retbins=True, duplicates='drop')[1]

# 5. 查看分箱边界（等频分箱的5个分位数）
print("\n等频分箱边界（5箱）：")
for i in range(len(bins) - 1):
    print(f"箱 {i}: {bins[i]:.2f} ~ {bins[i+1]:.2f}")

# 6. 重置索引以确保loc定位准确
clean_data = clean_data.reset_index(drop=True)

# 7. 查询指定索引的分箱结果
indices = [100, 300, 10000, 30000]
print("\n查询指定索引的分箱结果：")
for idx in indices:
    if idx < len(clean_data):
        PM25_value = clean_data.loc[idx, PM25_col]
        bin_label = clean_data.loc[idx, 'bin_label']
        print(f"索引 {idx}: PM2.5值 = {PM25_value:.2f}, 分箱标签 = {bin_label}")
    else:
        print(f"索引 {idx}: 超出数据范围（清洗后数据共 {len(clean_data)} 行）")

# 8. 最终确认索引连续性
clean_data = clean_data.reset_index(drop=True)  # 确保索引从0开始连续







import pandas as pd
import numpy as np

# 1. 读取CSV文件
file_path = r"D:\原神文件\PM25.csv"
data = pd.read_csv(file_path)

# 2. 检查PM2.5列名（假设列名为'pm2.5'或'PM2.5'）
pm25_col = 'pm2.5'  # 如果列名不同，请修改（如 'PM2.5'）
print("数据前5行：\n", data.head())

# 3. 去除缺失值
clean_data = data.dropna(subset=[pm25_col]).copy()  # 使用copy避免SettingWithCopyWarning

# 4. 等频分箱（5箱），标签设为0~4
clean_data['bin_label'] = pd.qcut(
    clean_data[pm25_col],
    q=5,                # 分5箱
    labels=False,       # 标签设为0~4
    duplicates='drop'   # 如果数据有重复边界，自动调整分箱
)

# 5. 查看分箱边界（等频分箱的5个分位数）
bins = pd.qcut(clean_data[pm25_col], q=5, retbins=True, duplicates='drop')[1]
print("\n等频分箱边界（5箱）：")
for i in range(len(bins) - 1):
    print(f"箱 {i}: {bins[i]:.2f} ~ {bins[i+1]:.2f}")

# 6. 重置索引以确保loc定位准确
clean_data = clean_data.reset_index(drop=True)

# 7. 查询指定索引的分箱结果
indices = [100, 300, 10000, 30000]
print("\n查询指定索引的分箱结果：")
for idx in indices:
    try:
        pm25_value = clean_data.loc[idx, pm25_col]  # 注意变量名保持一致
        bin_label = clean_data.loc[idx, 'bin_label']
        print(f"索引 {idx}: PM2.5值 = {pm25_value:.2f}, 分箱标签 = {bin_label}")
    except KeyError:
        print(f"索引 {idx}: 超出数据范围（清洗后数据共 {len(clean_data)} 行）")

# 8. 最终确认索引连续性
clean_data = clean_data.reset_index(drop=True)  # 确保索引从0开始连续


import pandas as pd
import numpy as np

# 1. 读取CSV文件
file_path = r"D:\原神文件\PM25.csv"
data = pd.read_csv(file_path)

# 2. 检查PM2.5列名（假设列名为'pm2.5'或'PM2.5'）
print("数据前5行：\n", data.head())
print("数据集的列名：", data.columns)

PM25_col = 'pm2.5'  # 如果列名不同，请修改（如 'PM2.5'）
if PM25_col not in data.columns:
    raise ValueError(f"列名 '{PM25_col}' 不存在，请检查列名拼写")

# 3. 去除缺失值
clean_data = data.dropna(subset=[PM25_col]).copy()  # 使用copy避免SettingWithCopyWarning

# 4. 等频分箱（5箱），标签设为0~4
clean_data['bin_label'], bins = pd.qcut(
    clean_data[PM25_col],
    q=5,                # 分5箱
    labels=False,       # 标签设为0~4
    retbins=True,       # 返回分箱边界
    duplicates='drop'   # 如果数据有重复边界，自动调整分箱
)

# 5. 查看分箱边界（等频分箱的5个分位数）
print("\n等频分箱边界（5箱）：")
for i in range(len(bins) - 1):
    print(f"箱 {i}: {bins[i]:.2f} ~ {bins[i+1]:.2f}")

# 6. 重置索引以确保loc定位准确
clean_data = clean_data.reset_index(drop=True)

# 7. 查询指定索引的分箱结果
indices = [100, 300, 10000, 30000]
print("\n查询指定索引的分箱结果：")
for idx in indices:
    if idx < len(clean_data):
        PM25_value = clean_data.loc[idx, PM25_col]
        bin_label = clean_data.loc[idx, 'bin_label']
        print(f"索引 {idx}: PM2.5值 = {PM25_value:.2f}, 分箱标签 = {bin_label}")
    else:
        print(f"索引 {idx}: 超出数据范围（清洗后数据共 {len(clean_data)} 行）")

# 8. 最终确认索引连续性
clean_data = clean_data.reset_index(drop=True)  # 确保索引从0开始连续
















import pandas as pd

# 1. 读取数据
file_path = r"D:\原神文件\PM25.csv"
df = pd.read_csv(file_path)

# 2. 检查列名是否存在
if 'pm2.5' not in df.columns:
    raise ValueError("列名 'pm2.5' 不存在，请检查列名拼写")

# 3. 处理缺失值，将缺失值标记为特殊值 -1
df['pm2.5'] = df['pm2.5'].fillna(-1)

# 4. 进行等频分箱，箱数为5，标签为0~4
# 使用 dropna=True 排除缺失值（-1）进行分箱
df['pm2.5_bin'] = pd.qcut(
    df[df['pm2.5'] != -1]['pm2.5'],  # 排除 -1 的值
    q=5,
    labels=range(5)
)

# 5. 将缺失值对应的分箱标签设为 -1
df['pm2.5_bin'] = df['pm2.5_bin'].cat.add_categories(-1)  # 添加 -1 类别
df['pm2.5_bin'] = df['pm2.5_bin'].fillna(-1).astype(int)  # 填充 -1 并转换为整数

# 6. 查看索引位置分别为100，300，10000，30000的分箱结果
indexes = [100, 300, 10000, 30000]
print("\n查询指定索引的分箱结果：")
for idx in indexes:
    if idx < len(df):  # 检查索引是否在范围内
        pm25_bin_value = df.loc[idx, 'pm2.5_bin']
        print(f"索引 {idx}: 分箱标签 = {pm25_bin_value}")
    else:
        print(f"索引 {idx}: 超出数据范围（数据共 {len(df)} 行）")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
import pandas as pd

# 加载数据集（请替换为你的实际文件路径）
# 假设 hs300 数据集是一个 CSV 文件，包含 'date' 和 'pct_chg' 列
file_path =r"D:\原神文件\hs300_basic.csv" # 替换为你的文件路径
df = pd.read_csv(file_path)

# 查看前几行数据，了解数据结构
print("原始数据预览：")
print(df.head())

# 将 'date' 列转换为 datetime 类型
df['date'] = pd.to_datetime(df['date'])

# 筛选出 2010 年的数据
df_2010 = df[df['date'].dt.year == 2010]

# 检查是否有缺失值，并决定如何处理
if df_2010['pct_chg'].isnull().any():
    print("警告: 'pct_chg' 列中存在缺失值，这些记录将会被忽略。")
    df_2010 = df_2010.dropna(subset=['pct_chg'])

# 计算 2010 年全年 'pct_chg' 的平均值
average_pct_chg_2010 = df_2010['pct_chg'].mean()

# 输出结果
print(f"\n2010 年全年的 pct_chg 平均值为: {average_pct_chg_2010:.4f}")

# 可选：绘制 'pct_chg' 在 2010 年的变化趋势图
import matplotlib.pyplot as plt

plt.figure(figsize=(14,7))
plt.plot(df_2010['date'], df_2010['pct_chg'], label='Daily Pct_Chg')
plt.title('HS300 Daily Pct_Chg in 2010')
plt.xlabel('Date')
plt.ylabel('Pct_Chg')
plt.legend()
plt.show()




import pandas as pd
from scipy.stats import spearmanr, pearsonr

df = pd.read_csv("D:\原神文件\wine.csv")

# 计算Pearson相关系数
pearson_corr = df.corr(method='pearson')
print("Pearson相关系数:\n", pearson_corr)

# 计算Spearman相关系数
spearman_corr = df.corr(method='spearman')
print("Spearman相关系数:\n", spearman_corr)




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("D:\原神文件\PM25.csv")

# 示例列名（根据你的数据调整）
# 假设数据中包含 'cbwd', 'hour', 'pm25' 等字段

# 虚拟变量编码
df_encoded = pd.get_dummies(df, columns=['cbwd', 'hour'], drop_first=True)

# 特征与目标变量
X = df_encoded.drop(columns=['pm.25'])
y = df_encoded['pm.25']

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 评估指标
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

# 输出结果
print("训练集 R²:", train_r2)
print("测试集 R²:", test_r2)
print("训练集 RMSE:", train_rmse)
print("测试集 RMSE:", test_rmse)



import pandas as pd
from scipy.stats import ttest_ind
df = pd.read_csv(r"D:\原神文件\hs300_basic.csv")
# 假设你已经加载了数据，并包含以下列：
# 'date'：日期
# 'volume'：成交量
# 'return'：当日收益率（正表示上涨，负表示下跌）

# 仅保留2017年以后的数据
df['date'] = pd.to_datetime(df['date'])
df_recent = df[df['date'] >= '2017-01-01']

# 划分上涨/下跌交易日
up_volume = df_recent[df_recent['close'] > 0]['volume']
down_volume = df_recent[df_recent['close'] < 0]['volume']

# 独立样本 t 检验
t_stat, p_value = ttest_ind(up_volume, down_volume, equal_var=False)

print("t 统计量:", t_stat)
print("p 值:", p_value)





import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv(r"D:\原神文件\hs300_basic.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date').reset_index(drop=True)

df['return'] = df['close'].pct_change()
df_recent = df[df['date'] >= '2017-01-01']

up_volume = df_recent[df_recent['return'] > 0]['volume'].dropna()
down_volume = df_recent[df_recent['return'] < 0]['volume'].dropna()

t_stat, p_value = ttest_ind(up_volume, down_volume, equal_var=False)
print("t 统计量:", t_stat)
print("p 值:", p_value)



import pandas as pd
from scipy.stats import ttest_1samp

# 读取数据（请根据你的实际路径修改）
df = pd.read_csv(r"D:\原神文件\hs300_basic.csv")

# 查看列名（调试用）
print("列名:", df.columns.tolist())

# 确保日期是 datetime 类型，并排序
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date').reset_index(drop=True)

# 计算每日收益率（如果 close 列不存在，请替换为你实际的收盘价列名）
df['return'] = df['close'].pct_change()

# 只保留2017年以后的数据
df_recent = df[df['date'] >= '2017-01-01']

# 删除缺失值
df_recent = df_recent.dropna(subset=['return'])

# 进行单样本 t 检验：H0: mean(return) = 0 vs H1: mean(return) > 0
t_stat, p_value = ttest_1samp(df_recent['return'], popmean=0, alternative='greater')

# 输出结果
print("t 统计量:", t_stat)
print("p 值:", p_value)

# 判断是否显著
alpha = 0.05
if p_value < alpha:
    print("结论：可以认为收益率的均值显著大于0")
else:
    print("结论：不可以认为收益率的均值显著大于0")




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("D:\原神文件\PM25.csv")

# 示例：假设数据包含 'pm2.5', 'year', 'Iws', 'cbwd' 等列
# df = pd.read_csv('your_pm25_data.csv')

# 对 cbwd 进行虚拟变量编码
df_encoded = pd.get_dummies(df, columns=['cbwd'], drop_first=True)

# 特征与目标变量
X = df_encoded.drop(columns=['pm2.5'])
y = df_encoded['pm2.5']

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 输出回归系数
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coefficients)

# 检查特定变量的回归系数
specific_coefficients = {
    "year": coefficients[coefficients['Feature'] == 'year']['Coefficient'].values,
    "Iws": coefficients[coefficients['Feature'] == 'Iws']['Coefficient'].values,
    "cbwd_NW": coefficients[coefficients['Feature'] == 'cbwd_NW']['Coefficient'].values,
    "cbwd_cv": coefficients[coefficients['Feature'] == 'cbwd_cv']['Coefficient'].values
}

for key, value in specific_coefficients.items():
    if len(value) > 0:
        print(f"变量 {key} 的回归系数是: {value[/XMLSchema warning suppressed]}", )
    else:
        print(f"未找到变量 {key}")




import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 加载数据集（请替换为你自己的路径）
df = pd.read_csv("D:\原神文件\PM25.csv")

# 对分类变量 cbwd 进行虚拟变量编码
df_encoded = pd.get_dummies(df, columns=['cbwd'], drop_first=True)

# 特征和目标变量（假设 pm2.5 是你要预测的目标）
X = df_encoded.drop(columns=['pm2.5'])
y = df_encoded['pm2.5']

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 输出所有特征的回归系数
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coefficients)

# 检查特定变量的回归系数
for feature in ['year', 'Iws', 'cbwd_NW', 'cbwd_cv']:
    row = coefficients[coefficients['Feature'] == feature]
    if not row.empty:
        coef = row.iloc[0]['Coefficient']
        print(f"变量 {feature} 的回归系数是: {coef:.4f}")
    else:
        print(f"未找到变量 {feature}")


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 加载数据集（请替换为你自己的路径）
df = pd.read_csv("D:\原神文件\PM25.csv")

# 检查是否有缺失值
print("原始数据中的缺失值情况:")
print(df.isnull().sum())

# 删除含有缺失值的行
df = df.dropna(subset=['pm2.5'])

# 或者，你也可以选择用特定值填充缺失值，例如均值、中位数等
# df['pm2.5'].fillna(df['pm2.5'].mean(), inplace=True)

# 对分类变量 cbwd 进行虚拟变量编码
df_encoded = pd.get_dummies(df, columns=['cbwd'], drop_first=True)

# 特征和目标变量（假设 pm2.5 是你要预测的目标）
X = df_encoded.drop(columns=['pm2.5'])
y = df_encoded['pm2.5']

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 输出所有特征的回归系数
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coefficients)

# 检查特定变量的回归系数
for feature in ['year', 'Iws', 'cbwd_NW', 'cbwd_cv']:
    row = coefficients[coefficients['Feature'] == feature]
    if not row.empty:
        coef = row.iloc[0]['Coefficient']
        print(f"变量 {feature} 的回归系数是: {coef:.4f}")
    else:
        print(f"未找到变量 {feature}")





import pandas as pd
from scipy.stats import f_oneway

# 假设你已经加载了二手车数据，并包含 'price' 和 'make' 字段
df = pd.read_csv(r"D:\原神文件\usedcar.csv")

# 筛选三个品牌
bmw_prices = df[(df['make'] == 'bmw')]['price']
infiniti_prices = df[(df['make'] == 'infiniti')]['price']
lexus_prices = df[(df['make'] == 'lexus')]['price']

# 方差分析
f_stat, p_value = f_oneway(bmw_prices, infiniti_prices, lexus_prices)

print("F统计量:", f_stat)
print("p值:", p_value)




import pandas as pd
from scipy.stats import f_oneway

# 读取数据（注意路径和列名）
df = pd.read_csv(r"D:\原神文件\usedcar.csv")

# 查看列名（调试用）
print("列名:", df.columns.tolist())

# 假设品牌列实际是 'brand'，而不是 'make'
bmw_prices = df[df['manufacturer'] == 'bmw']['price']
infiniti_prices = df[df['manufacturer'] == 'infiniti']['price']
lexus_prices = df[df['manufacturer'] == 'lexus']['price']

# 方差分析
f_stat, p_value = f_oneway(bmw_prices, infiniti_prices, lexus_prices)

print("F统计量:", f_stat)
print("p值:", p_value)

if p_value < 0.05:
    print("结论：三种车的价格均值存在显著差异")
else:
    print("结论：三种车的价格均值不存在显著差异")




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# 加载数据
data = pd.read_csv(r"D:\原神文件\PM25.csv")

# 对cbwd进行虚拟变量编码
data = pd.get_dummies(data, columns=['cbwd'], drop_first=True)

# 分离特征和目标变量
X = data.drop('pm2.5', axis=1)
y = data['pm2.5']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算并打印评估指标
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse}, Test R²: {r2}")



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 加载数据
data = pd.read_csv(r"D:\原神文件\PM25.csv")


# 检查缺失值
print("Missing values before processing:")
print(data.isnull().sum())

# 处理缺失值：填充 pm2.5 的 NaN 为中位数
data['pm2.5'] = data['pm2.5'].fillna(data['pm2.5'].median())

# 对分类变量 cbwd 进行虚拟变量编码（one-hot encoding）
data = pd.get_dummies(data, columns=['cbwd'], drop_first=True)

# 分离特征与目标变量
X = data.drop('pm2.5', axis=1)
y = data['pm2.5']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.2f}, Test R²: {r2:.4f}")


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 训练集预测
y_train_pred = model.predict(X_train)

# 计算训练集的RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print(f"Train RMSE: {train_rmse:.2f}")

# 计算训练集的R²
train_r2 = r2_score(y_train, y_train_pred)
print(f"Train R²: {train_r2:.4f}")


import pandas as pd
from scipy import stats

# 加载数据（请替换为你的文件路径）
df = pd.read_csv(r"D:\原神文件\usedcar.csv")

# 提取两个子集的价格
y1 = df[df['condition'] == 'good']['price']
y2 = df[df['condition'] == 'like new']['price']

# 独立双样本 t 检验（不假设方差相等）
t_stat, p_value = stats.ttest_ind(y1, y2, equal_var=False)

print(f"t统计量: {t_stat:.4f}")
print(f"双侧p值: {p_value:.4f}")

# 单侧检验：H1: E(y1) > E(y2)
if t_stat > 0:
    one_sided_p = p_value / 2
else:
    one_sided_p = 1 - p_value / 2

print(f"单侧p值 (H1: E(y1) > E(y2)): {one_sided_p:.4f}")



import pandas as pd
from scipy import stats

# 加载数据（请替换为你的文件路径）
df = pd.read_csv(r"D:\原神文件\hs300_basic.csv")

# 将 date 列转为 datetime 类型，并提取年份
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# 提取2021年和2022年的 pct_chg 数据
pct_chg_2021 = df[df['year'] == 2021]['pct_chg']
pct_chg_2022 = df[df['year'] == 2022]['pct_chg']

# 进行两样本KS检验
ks_stat, p_value = stats.ks_2samp(pct_chg_2021, pct_chg_2022)

print(f"KS统计量: {ks_stat:.4f}")
print(f"p值: {p_value:.4f}")

# 根据显著性水平判断是否存在显著差异
alpha = 0.05
if p_value < alpha:
    print("结论：可以认为存在显著差异")
else:
    print("结论：不可以认为存在显著差异")



import pandas as pd
from scipy.stats import pearsonr, spearmanr

# 假设你已经加载了数据集并且它包含'odometer'和'price'两列
df = pd.read_csv(r"D:\原神文件\usedcar.csv")

# 示例数据创建（实际使用时请替换为真实的df）
# df = pd.DataFrame({'odometer': [...], 'price': [...]}) 

# 计算Pearson相关系数
pearson_corr, _ = pearsonr(df['odometer'], df['price'])
print(f"Pearson相关系数: {pearson_corr:.4f}")

# 计算Spearman相关系数
spearman_corr, _ = spearmanr(df['odometer'], df['price'])
print(f"Spearman相关系数: {spearman_corr:.4f}")



import numpy as np
from scipy.spatial.distance import euclidean, cityblock

# 假设 wine_data 是你的数据集，numpy 数组形式
wine_data = np.loadtxt("D:\原神文件\wine.csv", delimiter=',')

# 获取索引500的数据点
point_500 = wine_data[500]

# 初始化变量以存储最短距离和对应的索引
min_euclidean_distance = float('inf')
min_manhattan_distance = float('inf')
closest_euclidean_index = -1
closest_manhattan_index = -1

# 遍历数据集，计算每个点到索引500的距离
for i, point in enumerate(wine_data):
    if i != 500: # 跳过索引500本身
        euclidean_dist = euclidean(point_500, point)
        manhattan_dist = cityblock(point_500, point)
        
        # 更新最近邻的索引和最小距离
        if euclidean_dist < min_euclidean_distance:
            min_euclidean_distance = euclidean_dist
            closest_euclidean_index = i
        
        if manhattan_dist < min_manhattan_distance:
            min_manhattan_distance = manhattan_dist
            closest_manhattan_index = i

print(f"在欧几里得距离下，与索引位置500距离最近的样本索引是 {closest_euclidean_index}")
print(f"在曼哈顿距离下，与索引位置500距离最近的样本索引是 {closest_manhattan_index}")


import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cityblock

# 1. 加载数据（假设你的文件名为 'wine.csv'）
df = pd.read_csv("D:\原神文件\wine.csv")

# 2. 删除第一列（如ID列）或其他非数值列（根据你的数据结构调整）
# 如果所有列都是特征，可以跳过这步
# df = df.drop(columns=['ID'])  # 示例：删除ID列

# 3. 转换为 numpy 数组（仅保留数值型列）
data = df.select_dtypes(include=[np.number]).values  # 只保留数值型列

# 4. 获取索引为500的样本
point_500 = data[500]

# 5. 初始化变量用于保存最近邻信息
min_euclidean_dist = float('inf')
min_manhattan_dist = float('inf')
closest_euclidean_idx = -1
closest_manhattan_idx = -1

# 6. 遍历所有样本，计算距离
for i in range(data.shape[0]):
    if i == 500:
        continue  # 跳过自己
    current_point = data[i]
    
    # 欧几里得距离
    euc_dist = euclidean(point_500, current_point)
    if euc_dist < min_euclidean_dist:
        min_euclidean_dist = euc_dist
        closest_euclidean_idx = i
    
    # 曼哈顿距离
    man_dist = cityblock(point_500, current_point)
    if man_dist < min_manhattan_dist:
        min_manhattan_dist = man_dist
        closest_manhattan_idx = i

# 7. 输出结果
print(f"在欧几里得距离下，与索引位置500距离最近的样本索引是 {closest_euclidean_idx}")
print(f"在曼哈顿距离下，与索引位置500距离最近的样本索引是 {closest_manhattan_idx}")







from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# 生成一个随机回归问题
X, y = make_regression(n_samples=100, n_features=30, noise=0.1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化Lasso回归模型，设置惩罚因子为1
lasso = Lasso(alpha=1.0)

# 拟合模型
lasso.fit(X_train, y_train)

# 预测
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

# 计算R²
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# 计算非零系数的数量，即保留的特征数量
num_nonzero_coeffs = sum(lasso.coef_ != 0)

print(f"训练集的R²: {r2_train:.4f}")
print(f"测试集的R²: {r2_test:.4f}")
print(f"保留的特征数量: {num_nonzero_coeffs}")









import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("D:\原神文件\hs300_basic.csv")


# 上涨交易日的成交量
up_volume = np.random.normal(loc=1000, scale=200, size=100)

# 下跌交易日的成交量
down_volume = np.random.normal(loc=950, scale=200, size=100)

# 进行t检验
t_statistic, p_value = stats.ttest_ind(up_volume, down_volume)

print(f"t统计量: {t_statistic:.3f}")
print(f"p值: {p_value:.4f}")

# 给定的t统计量和p值
given_t_statistic_1 = 2.531
given_t_statistic_2 = 2.532
given_p_value = 0.0114

# 检查计算得到的t统计量和p值是否与给定的数值相符
is_close_to_given_t1 = np.isclose(t_statistic, given_t_statistic_1, atol=0.001)
is_close_to_given_t2 = np.isclose(t_statistic, given_t_statistic_2, atol=0.001)
is_close_to_given_p = np.isclose(p_value, given_p_value, atol=0.0001)

print(f"计算得到的t统计量是否接近给定的{given_t_statistic_1}: {is_close_to_given_t1}")
print(f"计算得到的t统计量是否接近给定的{given_t_statistic_2}: {is_close_to_given_t2}")
print(f"计算得到的p值是否接近给定的{given_p_value}: {is_close_to_given_p}")

# 根据p值得出检验结论
alpha = 0.05
if p_value < alpha:
    conclusion = "存在显著差异"
else:
    conclusion = "不存在显著差异"

print(f"检验结论: {conclusion}")






from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import numpy as np

# 设置随机种子，确保结果一致
np.random.seed(42)

# 加载数据
data = load_wine()
X, y = data.data, data.target

# 进行KMeans聚类
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# 计算评估指标
ari = adjusted_rand_score(y, labels)
silhouette = silhouette_score(X, labels)

print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Silhouette Coefficient: {silhouette:.4f}")



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 加载数据（假设bank.csv是标准bank数据集）
df = pd.read_csv(r"D:\原神文件\bank.csv")

# 简单预处理（只选取部分特征，可进一步优化）
X = df.drop(columns=['y'])
y = df['y']

# 标签编码：no -> 0, yes -> 1
le = LabelEncoder()
y = le.fit_transform(y)

# 对类别型特征进行独热编码
X = pd.get_dummies(X, drop_first=True)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建KNN模型，设置k=6，基于距离加权
knn = KNeighborsClassifier(n_neighbors=6, weights='distance')
knn.fit(X_train, y_train)

# 预测
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

# 计算训练集准确率
train_acc = accuracy_score(y_train, y_pred_train)
print(f"训练集准确率: {train_acc:.4f}")

# 测试集分类报告
report = classification_report(y_test, y_pred_test, target_names=['y=0', 'y=1'], output_dict=False)
print("测试集分类报告：")
print(report)

# 获取 recall for y=0
test_report = classification_report(y_test, y_pred_test, output_dict=True)
recall_y0 = test_report['0']['recall']
print(f"测试集在 y=0 上的召回率: {recall_y0:.2f}")




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn import tree

# 加载数据（假设bank.csv是标准bank数据集）
df = pd.read_csv(r"D:\原神文件\bank.csv")

# 数据预处理
X = df.drop(columns=['y'])
y = df['y']

# 标签编码：no -> 0, yes -> 1
le = LabelEncoder()
y = le.fit_transform(y)

# 对类别型特征进行独热编码
X = pd.get_dummies(X, drop_first=True)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建决策树模型
clf = DecisionTreeClassifier(max_depth=4, min_samples_split=100, random_state=42)
clf.fit(X_train, y_train)

# 输出模型的一些基本信息
print("Feature importances: ", clf.feature_importances_)
print("Number of leaves: ", clf.get_n_leaves())
print(classification_report(y_test, clf.predict(X_test)))

# 如果需要查看具体的叶子节点信息，可以输出整个决策树结构或特定部分
# 这里以文本形式展示决策树
text_representation = tree.export_text(clf)
print(text_representation)

# 获取某个特定叶子的信息（例如第4个叶子的概率等信息），可以通过遍历树结构来实现
# 注意：这里的“第几个叶子”取决于你如何定义计数方式（例如按广度优先顺序）

# 示例：计算每个叶子中y=1的概率
leaf_id = clf.apply(X_test) # 获取每个测试样本所属的叶子ID
leaf_stats = {}
for leaf in set(leaf_id):
    mask = leaf_id == leaf
    if sum(mask) > 0:
        proba_y1 = sum(y_test[mask]) / sum(mask)
        leaf_stats[leaf] = {'samples': sum(mask), 'proba_y1': proba_y1}

# 打印每个叶子的相关统计信息
for leaf, stats in leaf_stats.items():
    print(f"Leaf {leaf} - Samples: {stats['samples']}, P(y=1): {stats['proba_y1']:.3f}")






from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import pandas as pd

# 加载数据
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# KMeans 聚类
kmeans = KMeans(n_clusters=7, random_state=42)
clusters = kmeans.fit_predict(X)

# 添加簇标签
X['cluster'] = clusters

# 查看每个簇的样本数量
cluster_sizes = X.groupby('cluster').size()
print("各簇样本数量：")
print(cluster_sizes)

# 找到最大和最小簇的编号
min_cluster_id = cluster_sizes.idxmin()
max_cluster_id = cluster_sizes.idxmax()

# 查看最大簇的 X1 的平均值
X1_mean_in_max_cluster = X[X['cluster'] == max_cluster_id]['alcohol'].mean()
print(f"\n最大簇中 'alcohol'（X1） 的平均值: {X1_mean_in_max_cluster:.3f}")

# 输出最大簇大小
print(f"最大簇样本数量: {cluster_sizes.max()}")



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv(r"D:\原神文件\bank.csv")

# 标签编码 y: 'no' -> 0, 'yes' -> 1
le = LabelEncoder()
df['y'] = le.fit_transform(df['y'])

# 特征和标签
X = df.drop(columns=['y'])
y = df['y']

# 对类别型特征进行独热编码
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 构建决策树模型
model = DecisionTreeClassifier(max_depth=4, min_samples_split=100, random_state=42)
model.fit(X_train, y_train)

# 计算置换特征重要性（重复20次）
result = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42)
perm_importances = result.importances_mean


features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_features = list(numerical_features) + list(features)

# 绘制直方图风格的条形图（可选）
importance_df = pd.DataFrame({'Feature': all_features, 'Importance': perm_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

import seaborn as sns
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title("Permutation Feature Importance")
plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd

# 假设你已经有 SHAP 值矩阵 (n_samples × n_features)
# 示例：生成一个随机 SHAP 矩阵作为演示
df = pd.read_csv(r"D:\原神文件\shapvalue.csv")


# 转为 DataFrame 更直观
feature_names = [f'x{i}' for i in range(21)]
df_shap = pd.DataFrame(shap_values, columns=feature_names)

# 变量重要性 = 平均绝对 SHAP 值
feature_importance = df_shap.abs().mean()

# 边际效应 = 平均 SHAP 值
marginal_effects = df_shap.mean()

# A. x3的变量重要性最高
A_correct = feature_importance.idxmax() == 'x3'

# B. x16的边际效应为0.00212
B_correct = np.isclose(marginal_effects['x16'], 0.00212, atol=1e-5)

# C. x0的变量重要性小于x2
C_correct = feature_importance['x0'] < feature_importance['x2']

# D. 正向和负向边际效应各4个
positive_count = (marginal_effects > 0).sum()
negative_count = (marginal_effects < 0).sum()
D_correct = (positive_count == 4) and (negative_count == 4)

# 输出结果
print("A. x3的变量重要性最高:", A_correct)
print("B. x16的边际效应为0.00212:", B_correct)
print("C. x0的变量重要性小于x2:", C_correct)
print("D. 正向和负向边际效应均为4个:", D_correct)



from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

# 加载数据集
data = load_wine()
X = data.data

# 进行 KMeans 聚类，类别数为 7
kmeans = KMeans(n_clusters=7, random_state=42)
kmeans.fit(X)

# 计算每个样本到最近簇中心的距离（S）
_, distances = pairwise_distances_argmin_min(X, kmeans.cluster_centers_)
S = distances

# 计算各项统计指标
mean_S = np.mean(S)
quantile_95_S = np.percentile(S, 95)
outliers_count = np.sum(S > 30)

# 输出结果
print("S 的平均值:", mean_S)
print("S 的 95% 分位数:", quantile_95_S)
print("S > 30 的异常样本数:", outliers_count)






import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1. 加载数据集（请确保 bank.csv 文件在当前目录）
df = pd.read_csv(r"D:\原神文件\bank.csv")

# 2. 特征与目标变量
X = df.drop('y', axis=1)
y = df['y'].map({'no': 0, 'yes': 1})

# 3. 分割数值型和类别型特征
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 4. 预处理：标准化 + 独热编码
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 构建完整 pipeline（预处理 + 神经网络）
model = Sequential([
    Dense(200, activation='relu', input_shape=(X_train_processed.shape[1],)),
    Dense(200, activation='relu'),
    Dense(200, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 7. 编译模型（Adam，学习率 0.001）
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 8. 训练模型
history = model.fit(X_train_processed, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=0)

# 9. 获取最终损失
final_loss = history.history['loss'][-1]
print("迭代结束后损失函数值约为: {:.4f}".format(final_loss))

# 10. 测试集准确率
test_loss, test_acc = model.evaluate(X_test_processed, y_test, verbose=0)
print("测试集的准确率约为: {:.4f}".format(test_acc))





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

# 1. 加载数据
df = pd.read_csv(r"D:\原神文件\bank.csv", sep=';')

# 2. 特征和目标变量
X = df.drop('y', axis=1)
y = df['y'].map({'no': 0, 'yes': 1})

# 3. 类别型特征列
categorical_cols = X.select_dtypes(include=['object']).columns

# 4. 预处理：对类别变量做 One-Hot 编码
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

X_processed = preprocessor.fit_transform(X)

# 5. 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 6. 建立决策树模型
model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=100,
    random_state=42
)
model.fit(X_train, y_train)

# 7. 获取特征重要性
feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_features = list(feature_names) + list(df.drop('y', axis=1).select_dtypes(include=['int64', 'float64']).columns)

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 8. 输出结果
print("Top 3 features:")
print(feature_importance_df.head(3))

print("\nFeature Importances:")
print(feature_importance_df)

# 9. 判断选项是否成立
def check_option(option, feature, importance):
    actual_imp = feature_importance_df[feature_importance_df['Feature'] == feature]['Importance'].values[0]
    is_correct = np.isclose(actual_imp, importance, atol=1e-3)
    print(f"{option}: {feature} 的重要性是 {actual_imp:.3f}，预期 {importance} → {'✅ 正确' if is_correct else '❌ 错误'}")

# A. 如果仅选择3个属性建立决策树，则应该选择balance、housing和age
top3 = feature_importance_df.iloc[:3]['Feature'].str.contains('balance|housing|age').sum()
A_correct = top3 >= 3
print(f"A: Top 3 是否是 balance、housing 和 age？{'✅ 是' if A_correct else '❌ 否'}")

# B. housing相对于balance更重要
imp_housing = feature_importance_df[feature_importance_df['Feature'] == 'housing_yes']['Importance'].values[0]
imp_balance = feature_importance_df[feature_importance_df['Feature'] == 'balance']['Importance'].values[0]
B_correct = imp_housing > imp_balance
print(f"B: housing 相对于 balance 更重要？{'✅ 是' if B_correct else '❌ 否'}")

# C. age的属性重要性是0.213
check_option("C", 'age', 0.213)

# D. balance的属性重要性是0.147
check_option("D", 'balance', 0.147)


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 读取数据
df = pd.read_csv(r"D:\原神文件\bank.csv", sep=';')

# 特征和目标变量
X = df.drop(columns=['y'])
y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

# 分离数值特征和类别特征
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# 预处理器：对数值特征标准化，对类别特征做独热编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 创建管道，将预处理器和KNN分类器组合在一起
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', KNeighborsClassifier())])

# 设置参数网格，仅考虑k的值从3到10
param_grid = {'classifier__n_neighbors': range(3, 11)}

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用网格搜索进行超参数调优
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 输出最佳参数和最佳得分
print(f"最优参数为k={grid_search.best_params_['classifier__n_neighbors']}")
print(f"最佳验证分数为{grid_search.best_score_:.4f}")

# 在测试集上评估最终模型
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"测试集上的准确率为{test_accuracy:.4f}")




import pandas as pd

# 加载数据集
df = pd.read_csv(r"D:\原神文件\bank.csv", sep=';')

# 假设我们要分析的是'balance'这一列
balance = df['balance']

# 计算描述性统计量
mean_balance = balance.mean()
median_balance = balance.median()
std_balance = balance.std()
quantile_95_balance = balance.quantile(0.95)

print(f"平均数是: {mean_balance:.2f}")
print(f"中位数为: {median_balance:.2f}")
print(f"标准差是: {std_balance:.2f}")
print(f"95%分位数是: {quantile_95_balance:.2f}")


import pandas as pd

# 加载 bank 数据集（确保文件在当前目录下）
df = pd.read_csv(r"D:\原神文件\bank.csv", sep=';')

# 查看前几行数据，确认结构
print("数据前几行：")
print(df.head())

# 查看数据基本信息：列名、非空数量、数据类型
print("\n数据基本信息：")
print(df.info())

# 对数值型列进行描述性统计
print("\n数值型字段的描述性统计（平均值、标准差、分位数等）：")
print(df.describe(include='number'))

# 对类别型列查看各列唯一值数量和常见值
print("\n类别型字段的统计信息：")
for col in df.select_dtypes(include='object').columns:
    print(f"\n【{col}】:")
    print(f"唯一值数量: {df[col].nunique()}")
    print(f"常见取值及频率:\n{df[col].value_counts(normalize=True).head()}")



from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# 加载数据
wine = load_wine()
X = wine.data

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans 聚类
kmeans = KMeans(n_clusters=7, random_state=42)
kmeans.fit(X_scaled)
centers = kmeans.cluster_centers_
inertia = kmeans.inertia_

print("簇中心维度:", centers.shape)
print("簇中心 X1 列（第一个特征）:", centers[:, 0])
print("距离平方和（inertia）:", inertia)


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 加载bank数据集
bank_data = fetch_openml(data_id=your_bank_dataset_id, as_frame=True)
X = bank_data.data
y = bank_data.target

# 数据预处理（根据实际情况进行）
# 这里假设已经进行了必要的预处理，例如处理缺失值、编码分类变量等

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树模型
dt_model = DecisionTreeClassifier(max_depth=4, min_samples_split=100, random_state=42)
dt_model.fit(X_train, y_train)

# 预测
train_preds = dt_model.predict(X_train)
test_preds = dt_model.predict(X_test)

# 计算准确率
train_accuracy = accuracy_score(y_train, train_preds)
test_accuracy = accuracy_score(y_test, test_preds)

# 获取叶子节点数量
leaf_nodes = dt_model.get_n_leaves()

print(f"训练集准确率: {train_accuracy}")
print(f"测试集准确率: {test_accuracy}")
print(f"叶子数量: {leaf_nodes}")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 读取数据
df = pd.read_csv(r"D:\原神文件\bank.csv", sep=";")  # 注意分隔符是分号";"

# 查看前几行数据（可选）
# print(df.head())

# 特征和标签
X = df.drop("y", axis=1)  # 假设目标列名为 "y"
y = df["y"]

# 对类别特征进行编码（将字符串转为数值）
X = X.apply(LabelEncoder().fit_transform)
y = LabelEncoder().fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建决策树模型
model = DecisionTreeClassifier(max_depth=4, min_samples_split=100, random_state=42)
model.fit(X_train, y_train)

# 预测与评估
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

leaf_count = model.get_n_leaves()

# 输出结果
print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")
print(f"叶子数量: {leaf_count}")
