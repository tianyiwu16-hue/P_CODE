import random
import pandas as pd

# 定义不同维度的短语库
subjects = ["村里的", "家门口的", "咱们村的", "这里的", "配套设施"]
attributes = ["道路", "环境", "绿化", "垃圾处理", "路灯", "卫生室", "网络", "饮水", "养老服务", "文化活动"]
evaluations = ["非常满意", "改善明显", "方便快捷", "环境优美", "需要加强", "挺到位的", "设施齐全", "焕然一新"]
connectors = ["，", "且", "而且", "但是"]

def generate_comment():
    # 随机组合生成一条评论
    s = random.choice(subjects)
    a = random.choice(attributes)
    e = random.choice(evaluations)
    return f"{s}{a}{e}。"

# 生成500条数据
data = [generate_comment() for _ in range(500)]

# 保存为CSV文件方便后续词云分析
df = pd.DataFrame(data, columns=["comment"])
df.to_csv("rural_livability_comments.csv", index=False, encoding='utf_8_sig')

print("已成功生成500条评论数据并保存为 rural_livability_comments.csv")











import random
import pandas as pd

# 核心评价对象 (名词，建议短一点)
objects = ["道路", "环境", "绿化", "垃圾", "路灯", "网络", "饮水", "养老", "厕所", "治安", "教育", "医疗"]
# 核心评价词 (形容词或状态)
states = ["好", "很棒", "方便", "整洁", "待加强", "完善", "一般", "给力", "急需", "优美"]

def generate_short_comment():
    # 随机抽取一个对象和一个评价词
    obj = random.choice(objects)
    sta = random.choice(states)
    # 组合为如“道路好”、“环境很棒”
    return f"{obj}{sta}"

# 生成 500 条数据
# 使用列表推导式生成
data = [generate_short_comment() for _ in range(500)]

# 保存为 CSV
df = pd.DataFrame(data, columns=["comment"])
df.to_csv("rural_livability_comments.csv", index=False, encoding='utf_8_sig')

print("已成功生成500条评论数据（每条不超过5字）并保存为 rural_livability_comments1.csv")












import random
import pandas as pd

# 映射指标的短评论库
# 格式：(指标内容, 评价)
indicators = [
    ("生态", "好"), ("绿化", "足"), ("环境", "优"), 
    ("路平", "坦"), ("路网", "密"), ("设施", "全"), 
    ("诊所", "近"), ("床位", "够"), ("服务", "好"), 
    ("收入", "增"), ("日子", "富"), ("分配", "公"), 
    ("办事", "近"), ("点位", "多"), ("便利", "高")
]

def generate_target_comment():
    # 随机选择一个指标及其描述
    item = random.choice(indicators)
    # 组合为如“生态好”、“绿化足”，确保字数在 2-4 字之间
    return f"{item[0]}{item[1]}"

# 生成 500 条数据
data = [generate_target_comment() for _ in range(500)]

# 保存为 CSV
df = pd.DataFrame(data, columns=["comment"])
df.to_csv("rural_index_comments.csv", index=False, encoding='utf_8_sig')

print("已生成 500 条指标相关评论，保存至 rural_index_comments.csv")














import random
import pandas as pd

# 评价对象（学术指标的口语化映射）
objs = ["生态环境", "路网密度", "公共服务", "人均GDP", "医疗床位", "空间可达", "POI点位", "居住环境", "路网覆盖", "经济水平"]
# 评价状态（4-6字评论的核心逻辑）
states = ["质量非常好", "建设很完善", "可达性很高", "指标很优秀", "整体水平优", "分布很合理", "非常方便达", "环境特别好", "提升空间大", "基本满足需"]

def generate_diverse_comment():
    # 随机组合
    obj = random.choice(objs)
    sta = random.choice(states)
    
    # 拼接并截取，确保总长度在 4-6 字
    comment = f"{obj}{sta}"
    return comment[:6] if len(comment) > 6 else comment

# 生成 500 条数据
data = [generate_diverse_comment() for _ in range(500)]

# 保存为 CSV
df = pd.DataFrame(data, columns=["comment"])
df.to_csv("rural_index_comments_v2.csv", index=False, encoding='utf_8_sig')
print("已成功生成 500 条短评论，保存至 rural_index_comments_v2.csv")



















import random
import pandas as pd

# 评价对象（核心指标）
objs = ["RSEI", "NDVI", "坡度", "POI", "床位", "GDP", "基尼", "路网", "生态", "医疗", "服务"]
# 评价状态（多维度评价）
states = ["好", "优", "高", "足", "低", "差", "完善", "均衡", "密集", "欠缺", "强", "弱"]

def generate_short_comment():
    # 随机组合，保持字数在 3-4 字
    obj = random.choice(objs)
    sta = random.choice(states)
    return f"{obj}{sta}"

# 生成 500 条数据
data = [generate_short_comment() for _ in range(500)]

# 保存为 CSV
df = pd.DataFrame(data, columns=["comment"])
df.to_csv("rural_short_comments.csv", index=False, encoding='utf_8_sig')
print("已成功生成 500 条 3-4 字评论，保存至 rural_short_comments.csv")











import random
import pandas as pd

# 分类定义词库
categories = {
    "战略": ["粮食安全", "防止返贫", "城乡融合", "千万工程", "实事求是"],
    "宜居": ["宜居宜业", "和美乡村", "乡村振兴", "农业现代化", "人居环境", "民生福祉"],
    "设施": ["基础设施", "宜居农房", "网络覆盖", "节水灌溉", "生态修复", "绿色生产", "土地整治"],
    "服务": ["公共服务", "学校共同体", "基层医疗", "法治乡村", "文明乡风", "移风易俗", "乡村治理"],
    "产业": ["新质生产力", "农文旅融合", "富民产业", "乡村人才", "新农人", "AI农业", "农民增收"]
}

def generate_academic_comments(n=500):
    data = []
    # 获取所有词汇
    all_words = [word for sublist in categories.values() for word in sublist]
    
    for _ in range(n):
        # 随机选择 1-2 个词进行组合，确保字数在 3-7 字
        sample = random.sample(all_words, k=random.randint(1, 2))
        comment = "".join(sample)
        # 截取以符合 3-7 字要求
        data.append(comment[:7])
    
    return data

# 生成并保存
df = pd.DataFrame(generate_academic_comments(), columns=["comment"])
df.to_csv("rural_index_data.csv", index=False, encoding='utf_8_sig')
print("数据生成完毕：rural_index_data.csv")































