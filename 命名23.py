import geopandas as gpd
import pandas as pd
import numpy as np
from libpysal.weights import Queen
from spreg import ML_Lag

# ---------------------------
# 1. 地图加载与三省合并 (解决乱码)
# ---------------------------
def load_maps_final():
    paths = [
        (r"D:\桌面应用\贵州省\贵州省.shp", "贵州"),
        (r"D:\桌面应用\浙江省\浙江省.shp", "浙江"),
        (r"D:\桌面应用\河南省\河南省.shp", "河南")
    ]
    gdfs = []
    for path, prov in paths:
        try:
            # 使用 utf-8 强制解决乱码问题
            data = gpd.read_file(path, encoding='utf-8') 
            
            # 自动寻找地名列
            target_col = next((c for c in ['name', 'NAME', '县名', 'COUNTY'] if c in data.columns), data.columns[0])
            data = data.rename(columns={target_col: 'map_name'})
            
            # 统一坐标系到 WGS84
            if data.crs is None: data.set_crs(epsg=4326, inplace=True)
            data = data.to_crs(epsg=4326)
            
            gdfs.append(data[['map_name', 'geometry']])
            print(f"✅ {prov} 读取成功！")
        except Exception as e:
            print(f"❌ {prov} 读取失败: {e}")
            
    return pd.concat(gdfs, ignore_index=True) if gdfs else None

# ---------------------------
# 2. 名字清洗函数 (确保匹配)
# ---------------------------
def ultra_clean(x):
    if pd.isna(x): return ""
    s = "".join(filter(lambda c: '\u4e00' <= c <= '\u9fa5', str(x).strip()))
    for suffix in ["省", "市", "自治县", "县", "区", "特区"]:
        if len(s) > 2: s = s.replace(suffix, "")
    return s

# ---------------------------
# 3. 主程序逻辑
# ---------------------------
gdf = load_maps_final()
if gdf is not None:
    gdf['name_c'] = gdf['map_name'].apply(ultra_clean)
    
    # 读取数据 (请确保路径和文件名与你硬盘中的一致)
    scores_df = pd.read_excel(r"D:\桌面应用\三省135县宜居性评价_最终版.xlsx")
    master_df = pd.read_csv(r"D:\桌面应用\总统计表1(1)(3).csv")
    
    scores_df['name_c'] = scores_df['县名'].apply(ultra_clean)
    master_df['name_c'] = master_df['县名'].apply(ultra_clean)
    
    # 合并
    excel_merged = pd.merge(scores_df, master_df, on='name_c', how='inner')
    full_data = gdf.merge(excel_merged, on='name_c', how='inner')
    
    print(f"📊 最终匹配成功的县单元数量: {len(full_data)}")

    if len(full_data) > 0:
        # 定义已对齐的变量名 (使用了合并后产生的 _x 后缀)
        features = [
            '医疗POI密度_x', 'RSEI_mean_x', '路网密度_x', '一般公共预算收入_x',
            'POI综合覆盖度_x', '医疗POI基尼系数_x', 'slope_mean_x'
        ]
        
        Y = full_data['综合得分'].values.reshape(-1, 1)
        X = full_data[features].values
        
        # 构建空间权重矩阵
        w = Queen.from_dataframe(full_data)
        w.transform = 'R'
        
        # 运行模型
        print("🛰️ 正在拟合 SLM 模型...")
        slm = ML_Lag(Y, X, w=w, name_y='宜居性得分', name_x=features)
        
        print("\n" + "📜" + " SLM 空间滞后模型回归报告 ".center(50, "="))
        print(slm.summary)
        print("=" * 55)
    else:
        print("❌ 数据匹配失败，请检查地图乱码是否已通过 ultra_clean 函数清除。")






# 🔍 照妖镜：打印出合并后表格的所有列名
print("--- 合并后表格的实际列名 ---")
print(full_data.columns.tolist())









from spreg import OLS

# 1. 运行 OLS (普通回归，不考虑空间)
print("📈 正在拟合 OLS 模型...")
ols = OLS(Y, X, name_y='宜居性得分', name_x=features)

# 2. 打印对比报告
print("\n" + "⚖️ OLS 与 SLM 模型对比 ".center(50, "="))
print(f"{'指标':<15} | {'OLS':<15} | {'SLM':<15}")
print("-" * 50)
print(f"{'AIC':<15} | {ols.aic:<15.4f} | {slm.aic:<15.4f}")
print(f"{'拟合优度(R²)':<15} | {ols.r2:<15.4f} | {slm.pr2:<15.4f}") # 注意SLM通常看伪R2
print("=" * 50)

# 3. 结论判断
if slm.aic < ols.aic:
    print("✅ 结论: SLM 模型的 AIC 更小，空间效应显著，模型更优！")
else:
    print("⚠️ 结论: AIC 差异较小，建议进一步检查 SEM 模型。")




















import numpy
print(numpy.__version__)
import sys
print(sys.executable)































