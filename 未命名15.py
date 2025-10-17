import json
import pandas as pd

# 使用Python内置的json模块读取JSON文件
with open('D:/学习工作文件夹/原神表格文件.json"', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 或者使用pandas读取JSON文件为DataFrame
df = pd.read_json('D:/学习工作文件夹/原神表格文件.json')

# 显示内容
print(data)
print(df.head())



import pandas as pd
import os

# 使用原始字符串或替换为双反斜杠或正斜杠来避免转义序列问题
# 注意这里directory只包含目录部分，file_name是单独的文件名
directory = r"D:\原神文件"  # 或者使用 "D:\\原神文件" 或 "D:/原神文件"
file_name = "creditcard.csv"
# Directory 只包含了目录部分（例如 "D:\\原神文件" 或 r"D:\原神文件" 或 "D:/原神文件"）。
# file_name 是单独的文件名（例如 "creditcard.csv"）。 • 使用 os.path.join() 来构建完整的文件路径，
# 这样可以确保路径分隔符的一致性，并且兼容不同操作系统。
# 构建完整路径
file_path = os.path.join(directory, file_name)

# 检查文件是否存在
if os.path.exists(file_path):
    try:
        # 加载数据
        creditcard = pd.read_csv(file_path)
        
        # Z值标准化
        mean_limit_bal = creditcard['LIMIT_BAL'].mean()
        std_limit_bal = creditcard['LIMIT_BAL'].std()
        creditcard['LIMIT_BAL_ZScore'] = (creditcard['LIMIT_BAL'] - mean_limit_bal) / std_limit_bal
        
        # 0-1归一化
        min_ = creditcard['LIMIT_BAL'].min()
        max_ = creditcard['LIMIT_BAL'].max()
        creditcard['LIMIT_BAL_Normalized'] = (creditcard['LIMIT_BAL'] - min_) / (max_ - min_)
        
        # 打印处理后的结果以验证
        print(creditcard[['LIMIT_BAL', 'LIMIT_BAL_ZScore', 'LIMIT_BAL_Normalized']].head())
    
    except Exception as e:
        print(f"读取或处理文件时出错: {e}")
else:
    print(f"找不到文件: {file_path}. 请检查文件路径是否正确。")