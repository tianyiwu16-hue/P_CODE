import pandas as pd

# 加载数据集，假设creditcard.csv是你的数据文件
creditcard = pd.read_csv("D:\原神文件\creditcard.csv")

# （1）抽取SEX=2的所有行，记为sp1
sp1 = creditcard[creditcard['SEX'] == 2]

# （2）抽取SEX=2且LIMIT_BAL>50000的行，记为sp2
sp2 = creditcard[(creditcard['SEX'] == 2) & (creditcard['LIMIT_BAL'] > 50000)]

# （3）将sp1和sp2合并（这里可能是求并集还是交集取决于需求）
# 这里我们做并集操作
merged_sp1_sp2 = pd.concat([sp1, sp2]).drop_duplicates().reset_index(drop=True)

# （4）在creditcard数据集中创建一个新列id，并等于其索引值
creditcard['id'] = creditcard.index

# 筛选出LIMIT_BAL>50000的行，并只保留id、SEX、EDUCATION列，记为sp3
sp3 = creditcard[creditcard['LIMIT_BAL'] > 50000][['id', 'SEX', 'EDUCATION']]

# 筛选出LIMIT_BAL>200000的行，并只保留id、SEX、MARRIAGE列，记为sp4
sp4 = creditcard[creditcard['LIMIT_BAL'] > 200000][['id', 'SEX', 'MARRIAGE']]

# （5）将sp3和sp4合并
merged_sp3_sp4 = pd.concat([sp3, sp4]).drop_duplicates().reset_index(drop=True)

# 查看前几行以确认数据抽取是否正确
print("SP1 - First few rows:")
print(sp1.head())

print("\nSP2 - First few rows:")
print(sp2.head())


# 导入creditcard数据集
df_creditcard = pd.read_csv("D:\原神文件\creditcard.csv")

# 计算各个变量的均值、中位数、频数（离散变量）、众数（离散变量）和标准差
mean_values = df_creditcard.mean()
median_values = df_creditcard.median()
mode_values = df_creditcard.mode().iloc[0] # 只取第一个模式
std_values = df_creditcard.std()

# 抽取样本大小为1000的样本，计算均值和标准差，并与原始数据比较
sample = df_creditcard.sample(1000)
sample_mean = sample.mean()
sample_std = sample.std()

# 重复抽样过程若干次，分析样本均值与总体均值之间的差异
def sample_analysis(df, n=1000, iterations=10):
    means = []
    for _ in range(iterations):
        sample_means = df.sample(n).mean()
        means.append(sample_means)
    return pd.DataFrame(means)

# 执行分析
analysis_result = sample_analysis(df_creditcard)






# 抽取SEX=2的所有行
sp1 = creditcard[creditcard['SEX'] == 2]
# 抽取SEX=2且LIMIT_BAL>50000的所有行
sp2 = creditcard[(creditcard['SEX'] == 2) & (creditcard['LIMIT_BAL'] > 50000)]
# 合并 sp1 和 sp2
merged_sp = pd.concat([sp1, sp2])
# 创建新列 id 并等于索引值
creditcard['id'] = creditcard.index
# 筛选 LIMIT_BAL > 50000 的行，并保留 id、SEX、EDUCATION 列
sp3 = creditcard[creditcard['LIMIT_BAL'] > 50000][['id', 'SEX', 'EDUCATION']]
# 筛选 LIMIT_BAL > 200000 的行，并保留 id、SEX、MARRIAGE 列
sp4 = creditcard[creditcard['LIMIT_BAL'] > 200000][['id', 'SEX', 'MARRIAGE']]
# 内连合并 sp3 和 sp4，以 id 列为唯一标识符
merged_inner = pd.merge(sp3, sp4, on='id', how='inner')
# 外连合并 sp3 和 sp4，以 id 列为唯一标识符
merged_outer = pd.merge(sp3, sp4, on='id', how='outer')


##############
# 计算平均数、中位数和标准差
mean_age = creditcard['AGE'].mean()
median_age = creditcard['AGE'].median()
std_age = creditcard['AGE'].std()
# 定义需要计算的统计量
statistics = ['mean', 'median', 'std']
# 使用 agg() 函数同时计算 AGE 和 LIMIT_BAL 的统计量
result = creditcard[['AGE', 'LIMIT_BAL']].agg(statistics)
# 输出结果
print(result)
# 计算频数
frequency = creditcard['EDUCATION'].value_counts()
# 计算众数
mode_value = creditcard['EDUCATION'].mode()[0]  # mode() 返回一个 Series，取第一个值
# 抽取样本大小为 1000 的样本
sample = creditcard.sample(n=1000, random_state=42)
# 计算原始数据集的 AGE 均值和标准差
original_mean = creditcard['AGE'].mean()
original_std = creditcard['AGE'].std()
# 计算样本的 AGE 均值和标准差
sample_mean = sample['AGE'].mean()
sample_std = sample['AGE'].std()

# 输出结果
print("原始数据集 AGE 的均值和标准差:")
print(f"均值: {original_mean:.2f}, 标准差: {original_std:.2f}")
print("\n样本 AGE 的均值和标准差:")
print(f"均值: {sample_mean:.2f}, 标准差: {sample_std:.2f}")
# 总体真实值
true_mean = creditcard['AGE'].mean()
# 初始化存储结果
sample_means = []  # 存储每次抽样的样本均值
errors = []        # 存储每次抽样均值与总体真实值的误差
# 重复抽样 1000 次
for i in range(1000):
    # 抽取样本大小为 1000 的样本
    sample = creditcard['AGE'].sample(n=4000, replace=True, random_state=i)  
    # 计算样本均值
    sample_mean = sample.mean()    
    # 计算误差（样本均值与总体真实值的绝对差）
    error = sample_mean - true_mean    
    # 记录结果
    sample_means.append(sample_mean)
    errors.append(error)

# 将结果转换为 DataFrame 以便分析
results = pd.DataFrame({
    'Sample_Mean': sample_means,
    'Error': errors
})
# 输出误差的统计量
print("\n误差的统计量:")
print(results['Error'].describe())
















