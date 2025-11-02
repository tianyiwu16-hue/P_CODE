import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random

# 设置页面标题
st.set_page_config(page_title="校园生活行为洞察平台", layout="wide")
st.title("🎓 大学生校园生活行为可视化平台")
st.markdown("### 时间范围：2025年9月8日 - 9月14日（本周）")

# --------------------------
# 1. 模拟数据生成（仅运行一次）
# --------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    students = [f"stu_{i:03d}" for i in range(100)]
    locations = ["食堂", "图书馆", "教学楼", "宿舍", "体育场"]
    data = []

    start_date = datetime(2025, 9, 8)
    for day_offset in range(7):  # 7天
        current_day = start_date + timedelta(days=day_offset)
        for stu in students:
            # 每人每天产生3-6条记录
            num_records = random.randint(3, 6)
            for _ in range(num_records):
                hour = random.randint(6, 23)
                minute = random.choice([0, 15, 30, 45])
                record_time = current_day + timedelta(hours=hour, minutes=minute)
                loc = random.choice(locations)
                amount = round(random.uniform(5, 25), 2) if loc == "食堂" else 0
                # 简单行为分类（用于环形图）
                if loc in ["图书馆", "教学楼"]:
                    behavior = "学习"
                elif loc == "食堂":
                    behavior = "饮食"
                elif loc == "体育场":
                    behavior = "运动"
                else:
                    behavior = "休息"
                data.append([stu, record_time, loc, amount, behavior])
    
    df = pd.DataFrame(data, columns=["student_id", "timestamp", "location", "amount", "behavior"])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    return df

df = generate_data()

# --------------------------
# 2. 核心指标计算
# --------------------------
total_spend = df['amount'].sum()
avg_spend = df.groupby('student_id')['amount'].sum().mean()
avg_library_hours = df[df['location'] == '图书馆'].groupby('student_id').size().mean() * 0.5  # 假设每次停留30分钟
early_risers = df[df['hour'] <= 7].groupby('date')['student_id'].nunique().mean() / len(df['student_id'].unique()) * 100

# --------------------------
# 3. 页面布局
# --------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("人均食堂消费（本周）", f"¥{avg_spend:.1f}", delta="↑5.2%")

with col2:
    st.metric("平均图书馆时长", f"{avg_library_hours:.1f} 小时/周")

with col3:
    st.metric("早起学生比例", f"{early_risers:.0f}%")

# --------------------------
# 4. 环形图：行为分布
# --------------------------
col4, col5 = st.columns(2)

with col4:
    st.subheader("行为类型分布")
    behavior_count = df['behavior'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.pie(behavior_count, labels=behavior_count.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("husl"))
    ax1.add_artist(plt.Circle((0,0),0.7,fc='white'))  # 环形图
    st.pyplot(fig1)

# --------------------------
# 5. 热力图：活跃时段
# ------------------
with col5:
    st.subheader("校园活跃热力图")
    pivot = df.groupby(['location', 'hour']).size().unstack(fill_value=0)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax2)
    ax2.set_title("各区域 hourly 活跃度")
    st.pyplot(fig2)

# --------------------------
# 6. 时间轴：个体轨迹
# --------------------------
st.subheader("学生行为轨迹（示例：stu_001）")
sample_student = df[df['student_id'] == 'stu_001'].sort_values('timestamp')
timeline_text = ""
for _, row in sample_student[sample_student['date'] == datetime(2025, 9, 8).date()].iterrows():
    timeline_text += f"- {row['timestamp'].strftime('%H:%M')} {row['location']}（消费¥{row['amount']:.0f}）\n"
st.markdown(timeline_text)

# --------------------------
# 7. 柱状图：学院对比（模拟数据）
# --------------------------
st.subheader("各学院周均图书馆访问次数对比")
# 模拟学院数据
college_map = {f"stu_{i:03d}": np.random.choice(["计算机", "外语", "体育", "经管"], p=[0.4,0.3,0.2,0.1]) for i in range(100)}
df['college'] = df['student_id'].map(college_map)
college_visits = df[df['location'] == '图书馆'].groupby('college').size().sort_values(ascending=False)

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.bar(college_visits.index, college_visits.values, color=sns.color_palette("Set2"))
ax3.set_ylabel("访问次数")
st.pyplot(fig3)

# --------------------------
# 8. 底部说明
# --------------------------
st.markdown("---")
st.caption("数据说明：本数据为模拟生成，仅用于教学演示。真实项目需遵守隐私保护规范。")













































