import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local
from splot.esda import moran_scatterplot, lisa_cluster
from spreg import ML_Lag, ML_Error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------
# 1. 数据对齐与合并
# ---------------------------
# 读取三省合并后的边界数据 (确保 shp 包含 135 个县)
# 注意：如果 shp 是分省的，请先用 gpd.pd.concat([gdf1, gdf2, gdf3]) 合并
counties_gdf = gpd.read_file(r"D:\桌面应用\ship1") 

# 读取之前的评价得分结果
scores_df = pd.read_excel(r"D:/桌面应用/三省135县宜居性评价_最终版.xlsx")

# 关键：确保连接字段没有多余空格
counties_gdf['name'] = counties_gdf['name'].astype(str).str.strip()
scores_df['县名'] = scores_df['县名'].astype(str).str.strip()

# 合并数据
full_data = counties_gdf.merge(scores_df, left_on='name', right_on='县名', how='inner')

# 检查样本量是否为 135 (或你的 279 条记录，取决于你分析的年度维度)
print(f"✅ 数据合并成功，共有 {len(full_data)} 个空间单元进入分析。")

# ---------------------------
# 2. 构建空间权重矩阵 (W)
# ---------------------------
# 移除可能导致孤岛的无几何数据
full_data = full_data[full_data.geometry.notnull()]

# 构建 Queen 邻接矩阵
w = Queen.from_dataframe(full_data)

# 💡 稳健性处理：处理孤岛（Island）
if len(w.islands) > 0:
    print(f"⚠️ 警告：检测到 {len(w.islands)} 个孤岛县。正在切换为 KNN (k=4) 权重以保证连通性...")
    from libpysal.weights import KNN
    w = KNN.from_dataframe(full_data, k=4)

w.transform = 'R'  # 行标准化

# ---------------------------
# 3. 全局空间自相关 (Global Moran's I)
# ---------------------------
y = full_data['综合得分'].values
moran = Moran(y, w)

print("\n" + "="*30)
print(f"🌍 全局 Moran's I 指数: {moran.I:.4f}")
print(f"Z-score: {moran.z_sim:.4f}")
print(f"P-value: {moran.p_sim:.4f}")
print("="*30)

# 绘制 Moran 散点图
fig, ax = moran_scatterplot(moran, aspect_equal=True)
ax.set_title("宜居性综合得分 Moran 散点图", fontsize=15)
plt.show()

# ---------------------------
# 4. 局部空间自相关 (LISA)
# ---------------------------
lisa = Moran_Local(y, w)

# 绘制 LISA 聚类地图
# p=0.05 代表 95% 置信区间
fig, ax = plt.subplots(figsize=(12, 10))
lisa_cluster(lisa, full_data, p=0.05, ax=ax, 
             legend_kwds={'loc': 'lower right', 'title': 'LISA 聚类类型'})
full_data.boundary.plot(ax=ax, linewidth=0.5, color='gray') # 叠加边界线
ax.set_title("三省县域宜居性 LISA 聚类地图", fontsize=18)
plt.show()

# ---------------------------
# 5. 空间可视化：综合得分分布
# ---------------------------
fig, ax = plt.subplots(figsize=(12, 10))
full_data.plot(column='综合得分', 
               cmap='YlGnBu', 
               legend=True, 
               ax=ax, 
               edgecolor='black', 
               linewidth=0.3,
               legend_kwds={'label': "宜居性综合得分", 'orientation': "horizontal", 'shrink': 0.8})
ax.set_title("2024年三省县域宜居性空间分布图", fontsize=18)
ax.set_axis_off()
plt.show()

# ---------------------------
# 6. 空间计量回归分析 (SLM / SEM)
# ---------------------------
# 选择核心解释变量 (请确保列名与 Excel 一致)
indep_vars = ['NDVI', 'RSEI_mean', 'slope_mean', '教育POI密度', '医疗POI密度', '人均GDP']
X = full_data[indep_vars].values
y_reg = y.reshape((-1, 1))

print("\n🚀 正在拟合空间滞后模型 (SLM)...")
slm = ML_Lag(y_reg, X, w=w, name_y='Livability_Score', name_x=indep_vars)
print(slm.summary)

print("\n🚀 正在拟合空间误差模型 (SEM)...")
sem = ML_Error(y_reg, X, w=w, name_y='Livability_Score', name_x=indep_vars)
print(sem.summary)

# ---------------------------
# 7. 导出分析数据
# ---------------------------
# 将 LISA 结果（1-HH, 2-LH, 3-LL, 4-HL）存入原表
full_data['lisa_q'] = lisa.q
full_data.to_file("D:/TJJM/04_分析结果/Spatial_Analysis_Result.shp", encoding='utf-8')
print("\n✅ 空间分析结果已导出至 shp 文件。")





















# 查看地图里的前5个名称
print("🗺️ 地图中的县名示例:", counties_gdf['name'].head(5).tolist())

# 查看Excel里的前5个县名
print("📊 Excel中的县名示例:", scores_df['县名'].head(5).tolist())

# 检查是否有重合
common = set(counties_gdf['name']).intersection(set(scores_df['县名']))
print(f"🔗 两个表中完全一致的名称数量: {len(common)}")









import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local
from splot.esda import moran_scatterplot, lisa_cluster

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------
# 1. 核心：修正乱码读取地图
# ---------------------------
# 💡 提示：'encoding="utf-8"' 通常能解决你刚才看到的 'å\x8d\x97æ\x98\x8e' 乱码问题
shp_path = r"D:\桌面应用\ship1"
counties_gdf = gpd.read_file(shp_path, encoding='utf-8') 

# 如果 utf-8 还是乱码，请尝试改为 encoding='gbk'
if '南明' not in str(counties_gdf['name'].iloc[0]):
    counties_gdf = gpd.read_file(shp_path, encoding='gbk')

# ---------------------------
# 2. 读取并对齐贵州数据
# ---------------------------
# 请确保这个 Excel 已经更新为贵州 88 个县/区的数据
excel_path = r"D:\桌面应用\三省135县宜居性评价_最终版.xlsx"
scores_df = pd.read_excel(excel_path)

# 清洗函数：统一去掉“县、区、特区、自治县”后缀，防止对不上
def clean_gz_name(x):
    if pd.isna(x): return ""
    x = str(x).strip()
    for s in ["省", "市", "自治县", "县", "区", "特区"]:
        if len(x) > 2: # 保护如“开阳县”不变成“开阳”，但“南明区”变成“南明”
            x = x.replace(s, "")
    return x

counties_gdf['name_clean'] = counties_gdf['name'].apply(clean_gz_name)
scores_df['县名_clean'] = scores_df['县名'].apply(clean_gz_name)

# 合并数据
full_data = counties_gdf.merge(scores_df, left_on='name_clean', right_on='县名_clean', how='inner')

print(f"📊 贵州地图原有单元: {len(counties_gdf)} 个")
print(f"🔗 成功匹配进入分析: {len(full_data)} 个县区")

if len(full_data) == 0:
    print("❌ 依旧无法匹配！请检查 Excel 里的县名是否是贵州的（如南明、云岩、遵义等）。")
    # 打印前几个出来看看，人工对一下
    print("地图名示例:", counties_gdf['name_clean'].head(3).tolist())
    print("Excel名示例:", scores_df['县名_clean'].head(3).tolist())
else:
    # ---------------------------
    # 3. 空间权重矩阵构建
    # ---------------------------
    w = Queen.from_dataframe(full_data, use_index=True)
    w.transform = 'R'

    # ---------------------------
    # 4. 全局与局部空间自相关
    # ---------------------------
    y = full_data['综合得分'].values
    
    # 全局 Moran's I
    moran = Moran(y, w)
    print(f"\n🌍 全局 Moran's I: {moran.I:.4f} (P-value: {moran.p_sim:.4f})")

    # 局部 LISA
    lisa = Moran_Local(y, w)

    # ---------------------------
    # 5. 绘图展示
    # ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 左图：综合得分分布
    full_data.plot(column='综合得分', cmap='YlGnBu', legend=True, 
                   ax=axes[0], edgecolor='black', linewidth=0.3)
    axes[0].set_title("贵州省县域宜居性得分分布", fontsize=15)
    axes[0].axis('off')

    # 右图：LISA 聚类图
    lisa_cluster(lisa, full_data, p=0.05, ax=axes[1])
    axes[1].set_title("贵州省宜居性 LISA 聚类图 (HH/LL)", fontsize=15)
    
    plt.tight_layout()
    plt.show()





























