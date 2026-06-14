import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ---------------------------
# 1. 加载并合并数据
# ---------------------------
# 读取完整指标数据 (自变量 X)
master_data = pd.read_excel(r"D:\桌面应用\总统计表2024.xlsx")
# 读取得分数据 (因变量 Y)
score_data = pd.read_excel(r"D:\桌面应用\三省135县宜居性评价_最终版.xlsx")

# 清洗县名以确保对齐
master_data['县名_clean'] = master_data['县名'].str.strip()
score_data['县名_clean'] = score_data['县名'].str.strip()

# 合并：只保留两边都有的县
reg_data = pd.merge(score_data[['县名_clean', '综合得分']], 
                    master_data, 
                    on='县名_clean', 
                    how='inner')

# ---------------------------
# 2. 筛选回归变量 (示例变量，请根据你的研究调整)
# ---------------------------
# 选出你认为最可能影响宜居性的 6-8 个核心指标
# 建议涵盖：经济、公共服务、生态、基础设施
features = [
    '医疗POI密度',
    'RSEI_mean',
    '路网密度',
    '一般公共预算收入',
    'POI综合覆盖度',
    '医疗POI基尼系数',
    'slope_mean'
]

# 准备 X 和 y
X = reg_data[features]
y = reg_data['综合得分']

# 💡 重要：OLS 必须手动添加常数项 (截距)
X_with_constant = sm.add_constant(X)

# ---------------------------
# 3. 拟合 OLS 模型
# ---------------------------
model = sm.OLS(y, X_with_constant).fit()

# ---------------------------
# 4. 输出诊断报告
# ---------------------------
print("\n" + "📊" + " OLS 回归分析报告 ".center(50, "="))
print(model.summary())
print("=" * 55)

# ---------------------------
# 5. 结果导出
# ---------------------------
# 将回归系数存入 Excel 方便写论文
ols_results = pd.DataFrame({
    "系数 (Coef.)": model.params,
    "标准误 (Std.Err.)": model.bse,
    "t值": model.tvalues,
    "P值 (P>|t|)": model.pvalues
})
ols_results.to_excel("OLS回归系数详情1.xlsx")
print("\n✅ OLS 结果已导出至：OLS回归系数详情.xlsx")




























import geopandas as gpd
import pandas as pd
import numpy as np
from libpysal.weights import Queen
from spreg import ML_Lag

# --- 1. 数据加载与对齐 ---
# 请确保此处指向具体的 .shp 文件
gdf = gpd.read_file(r"D:\桌面应用\ship1\贵州省.shp", encoding='gbk')

master_df = pd.read_excel(r"D:\桌面应用\总统计表2024.xlsx")
scores_df = pd.read_excel(r"D:\桌面应用\三省135县宜居性评价_最终版.xlsx")

def clean(x): return str(x).strip().replace("县", "").replace("区", "").replace("市", "")

gdf['name_c'] = gdf['name'].apply(clean)
scores_df['name_c'] = scores_df['县名'].apply(clean)
master_df['name_c'] = master_df['县名'].apply(clean)

# 合并数据
temp_data = pd.merge(scores_df[['name_c', '综合得分']], master_df, on='name_c', how='inner')
full_data = gdf.merge(temp_data, left_on='name_c', right_on='name_c', how='inner')

# 💡 检查是否有数据丢失
print(f"合并后的样本量: {len(full_data)}")
if len(full_data) == 0:
    print("❌ 合并失败，请检查 'name_c' 列的数据是否对齐")
    exit()

# --- 2. 准备变量 ---
features = [
    '医疗POI密度', 'RSEI_mean', '路网密度', '一般公共预算收入',
    'POI综合覆盖度', '医疗POI基尼系数', 'slope_mean'
]

Y = full_data['综合得分'].values
X = full_data[features].values

# --- 3. 空间权重矩阵 ---
w = Queen.from_dataframe(full_data)
w.transform = 'R'

# --- 4. 运行 SLM ---
slm = ML_Lag(Y, X, w=w, name_y='宜居性得分', name_x=features)

print(slm.summary)





import geopandas as gpd
import pandas as pd

# 1. 尝试读取地图 (如果ship1是文件夹，请确保里面有.shp文件)
# 请将 '你的地图文件名.shp' 替换为你 ship1 文件夹下实际的 .shp 文件名
map_file = r"D:\桌面应用\ship1\贵州省.shp" 
gdf = gpd.read_file(map_file, engine='fiona', encoding='gbk')

# 2. 读取 Excel 结果
scores_df = pd.read_excel(r"D:\桌面应用\三省135县宜居性评价_最终版.xlsx")

# 3. 简单的清洗并打印 (先不合并，直接看名字)
def clean_name(x):
    # 转换为字符串并去除首尾空格
    x = str(x).strip()
    return x

# 打印前 5 个原始名字，看看长什么样
print("--- 地图原始名字前5个 ---")
print(gdf['name'].head(5).tolist())

print("\n--- Excel原始名字前5个 ---")
print(scores_df['县名'].head(5).tolist())

























import geopandas as gpd
import pandas as pd

def load_maps():
    # 路径列表 (确保这三个文件都在 D:\桌面应用\ship1\ 目录下)
    file_paths = [
        r"D:\桌面应用\贵州省\贵州省.shp",
        r"D:\桌面应用\浙江省\浙江省.shp",
        r"D:\桌面应用\河南省\河南省.shp"
    ]
    
    gdfs = []
    for path in file_paths:
        try:
            # 使用 fiona 引擎读取，并指定编码
            temp_gdf = gpd.read_file(path, engine='fiona', encoding='gbk')
            gdfs.append(temp_gdf)
            print(f"✅ 成功加载: {path}")
        except Exception as e:
            print(f"❌ 无法加载 {path}: {e}")
            
    if not gdfs:
        return None
    
    # 将三个省的地图合并为一个整体
    combined_gdf = pd.concat(gdfs, ignore_index=True)
    return combined_gdf

# 加载地图
gdf = load_maps()

if gdf is not None:
    # --- 只有加载成功后，才执行后续的清洗和合并 ---
    def robust_clean(x):
        if pd.isna(x): return ""
        x = str(x).strip()
        for s in ["省", "市", "自治县", "县", "区", "特区"]:
            if len(x) > 2: x = x.replace(s, "")
        return x

    # 假设你的地图中存放地名的列名是 'name'
    gdf['name_c'] = gdf['name'].apply(robust_clean)
    
    # 读取你的 Excel
    scores_df = pd.read_excel(r"D:\桌面应用\三省135县宜居性评价_最终版.xlsx")
    master_df = pd.read_excel(r"D:\桌面应用\总统计表2024.xlsx")
    
    scores_df['name_c'] = scores_df['县名'].apply(robust_clean)
    master_df['name_c'] = master_df['县名'].apply(robust_clean)
    
    # 合并
    excel_data = pd.merge(scores_df, master_df, on='name_c', how='inner')
    full_data = gdf.merge(excel_data, left_on='name_c', right_on='name_c', how='inner')
    
    print(f"📊 最终匹配进入空间分析的县数量: {len(full_data)}")
else:
    print("❌ 地图合并失败，请检查 ship1 目录下是否真的存在这三个 .shp 文件。")







# 打印清洗后的前 5 个名字进行比对
print("--- 地图中的名字 (name_c) ---")
print(gdf['name_c'].head(5).tolist())

print("\n--- Excel 中的名字 (name_c) ---")
# 假设你的 scores_df 已经读入
print(scores_df['name_c'].head(5).tolist())

# 检查是否存在重叠
map_names = set(gdf['name_c'])
excel_names = set(scores_df['name_c'])
common = map_names.intersection(excel_names)

print(f"\n✅ 共有 {len(common)} 个名字完全匹配。")
if len(common) == 0:
    print("❌ 警告：没有找到任何匹配的名字。请检查是否一方叫“贵阳”，另一方叫“贵阳市”等情况。")





















