import pandas as pd

# 1. 定义你的目标排序顺序
target_order = [
    "巩义市", "新郑市", "中牟县", "新密市", "荥阳市", "新安县", "栾川县", "淇县", 
    "新乡县", "沁阳市", "林州市", "长葛市", "鄢陵县", "义马市", "渑池县", "尉氏县", 
    "兰考县", "洛宁县", "宜阳县", "伊川县", "温县", "武陟县", "宝丰县", "舞钢市", 
    "范县", "襄城县", "舞阳县", "内乡县", "淅川县", "桐柏县", "叶县", "鲁山县", 
    "滑县", "内黄县", "封丘县", "方城县", "镇平县", "社旗县", "宁陵县", "柘城县", 
    "夏邑县", "西华县", "商水县", "太康县", "项城市"
]

# 2. 读取 Excel 文件
# 如果你的文件在当前目录下，直接写文件名即可
input_file =  "D:\桌面应用\Henan_Remote_Sensing_Analysis_2020_2024.xlsx"
df = pd.read_excel(input_file)

# 3. 指定包含县市名称的列名（请确保与你的Excel表头名称一致）
county_column = 'name' 

# 4. 设置顺序并排序
# 这一步保证整行数据会随着县市名称的移动而移动
df[county_column] = pd.Categorical(df[county_column], categories=target_order, ordered=True)
df_sorted = df.sort_values(by=['Year', county_column])

# 5. 保存结果
output_file = 'Henan_Sorted_Result.xlsx'
df_sorted.to_excel(output_file, index=False)

print(f"✅ 处理完成！已按你的顺序排列整行数据，结果已保存至: {output_file}")











import pandas as pd

# 1. 你的目标排序列表
target_order = [
    "巩义市", "新郑市", "中牟县", "新密市", "荥阳市", "新安县", "栾川县", "淇县", 
    "新乡县", "沁阳市", "林州市", "长葛市", "鄢陵县", "义马市", "渑池县", "尉氏县", 
    "兰考县", "洛宁县", "宜阳县", "伊川县", "温县", "武陟县", "宝丰县", "舞钢市", 
    "范县", "襄城县", "舞阳县", "内乡县", "淅川县", "桐柏县", "叶县", "鲁山县", 
    "滑县", "内黄县", "封丘县", "方城县", "镇平县", "社旗县", "宁陵县", "柘城县", 
    "夏邑县", "西华县", "商水县", "太康县", "项城市"
]

# 2. 读取 Excel
input_file =  r"D:\桌面应用\Henan_Remote_Sensing_Analysis_2020_2024.xlsx"
df = pd.read_excel(input_file)

# 3. 指定你的列名（请核对 Excel 表头）
# 假设年份列叫 'Year'，县名列叫 'District_CN'
county_col = 'name'
year_col = 'Year'

# 4. 执行自定义排序
df[county_col] = pd.Categorical(df[county_col], categories=target_order, ordered=True)
df_sorted = df.sort_values(by=[year_col, county_col])

# 5. 拆分年份并存入不同的 Sheet
output_file = 'Henan_RSEI_Split_2020_2024.xlsx'

with pd.ExcelWriter(output_file) as writer:
    # 提取 2020 数据并保存
    df_2020 = df_sorted[df_sorted[year_col] == 2020]
    df_2020.to_excel(writer, sheet_name='2020年数据', index=False)
    
    # 提取 2024 数据并保存
    df_2024 = df_sorted[df_sorted[year_col] == 2024]
    df_2024.to_excel(writer, sheet_name='2024年数据', index=False)

print(f"✅ 处理完成！")
print(f"📂 结果文件：{output_file}")
print(f"💡 文件内已自动按年份拆分为两个工作表（Sheet），且每个年份内均已按你的顺序排好。")

























