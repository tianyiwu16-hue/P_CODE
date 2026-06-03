import pandas as pd
import jieba
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# 1. 东方美学配色方案函数
def oriental_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    """
    自定义颜色函数：提供红色、金色及暖色调
    """
    # 选取的颜色：中国红、朱砂、黛金、琥珀色、胭脂色
    colors = ['#B22222', '#DAA520', '#8B0000', '#FFD700', '#CD8539', '#A52A2A', '#E60012']
    return np.random.choice(colors)

def main():
    # --- 第一步：读取数据 ---
    # 请确保文件名与路径正确
    file_path =  "D:\桌面应用\评论.xlsx"
    try:
        df = pd.read_excel(file_path)
        # 提取评论内容
        text_data = "".join(df['comment'].astype(str).tolist())
    except Exception as e:
        print(f"读取文件出错: {e}")
        return

    # --- 第二步：清洗数据与分词 ---
    # 定义需要剔除的无意义词（停用词）
    # 在实际研究中，你可以通过读取外部 stopwords.txt 加载更多词汇
    stopwords = set(['的', '了', '在', '是', '我', '有', '都', '和', '一个', '这', '也', '就', '很', '于', '吗', '吧', '呢', '啊', '这种'])
    
    # 针对花西子品牌，手动添加jieba词库，防止品牌词被切分
    custom_keywords = ['花西子', '数字代言人', '东方美学', '设计感', '真实感', '建模']
    for word in custom_keywords:
        jieba.add_word(word)

    # 使用正则只保留中文字符（过滤掉标点、数字、英文）
    chinese_only = "".join(re.findall(r'[\u4e00-\u9fa5]+', text_data))
    
    # 分词
    word_list = jieba.lcut(chinese_only)
    
    # 过滤停用词和单字
    filtered_words = [word for word in word_list if word not in stopwords and len(word) > 1]

    # --- 第三步：统计高频词 ---
    word_counts = Counter(filtered_words)
    print("--- 前50个高频词统计 ---")
    top_50 = word_counts.most_common(50)
    for i, (word, count) in enumerate(top_50, 1):
        print(f"{i}. {word}: {count}")

    # --- 第四步：生成词云图 ---
    # 注意：font_path 必须指向你电脑中存在的中文字体文件
    # Windows 一般在 C:/Windows/Fonts/simhei.ttf
    # Mac 一般在 /System/Library/Fonts/STHeiti Light.ttc
    font_path = "C:/Windows/Fonts/msyh.ttc" # 这里以微软雅黑为例

    wc = WordCloud(
        font_path=font_path,
        background_color='white',
        width=1000,
        height=800,
        max_words=100,
        random_state=42,
        color_func=oriental_color_func, # 应用东方美学颜色
        prefer_horizontal=0.7 # 70%的词汇水平显示，增加设计感
    )

    # 基于词频生成
    wc.generate_from_frequencies(word_counts)

    # --- 第五步：显示与保存 ---
    plt.figure(figsize=(12, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off') # 隐藏坐标轴
    
    # 保存图片
    wc.to_file("huaxizi_comment_wordcloud.png")
    print("\n词云图已保存为: huaxizi_comment_wordcloud.png")
    
    plt.show()

if __name__ == "__main__":
    main()
    
    
    

    



import pandas as pd
import jieba
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # 新增：用于处理图像形状

# 1. 东方美学配色方案函数
def oriental_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = ['#B22222', '#DAA520', '#8B0000', '#FFD700', '#CD8539', '#A52A2A', '#E60012']
    return np.random.choice(colors)

def main():
    # --- 第一步：读取数据 ---
    # 💡 提示：Windows路径建议加 r 防止转义字符报错
    file_path = r"D:\桌面应用\评论.xlsx" 
    mask_path = r"D:\桌面应用\ChatGPT Image 2026年3月11日 19_34_18.png"  # 💡 在这里输入你的并蒂莲形状图片路径
    
    try:
        df = pd.read_excel(file_path)
        text_data = "".join(df['comment'].astype(str).tolist())
    except Exception as e:
        print(f"读取文件出错: {e}")
        return

    # --- 第二步：清洗数据与分词 ---
    stopwords = set(['的', '了', '在', '是', '我', '有', '都', '和', '一个', '这', '也', '就', '很', '于', '吗', '吧', '呢', '啊', '这种', '真的', '太', '还'])
    
    custom_keywords = ['花西子', '数字代言人', '东方美学', '设计感', '真实感', '建模']
    for word in custom_keywords:
        jieba.add_word(word)

    chinese_only = "".join(re.findall(r'[\u4e00-\u9fa5]+', text_data))
    word_list = jieba.lcut(chinese_only)
    filtered_words = [word for word in word_list if word not in stopwords and len(word) > 1]

    # --- 第三步：统计高频词 ---
    word_counts = Counter(filtered_words)
    print("--- 前50个高频词统计 ---")
    top_50 = word_counts.most_common(50)
    for i, (word, count) in enumerate(top_50, 1):
        print(f"{i}. {word}: {count}")

    # --- 第四步：形状遮罩处理 (新增部分) ---
    try:
        # 读取并蒂莲形状图
        mask_image = np.array(Image.open(mask_path))
    except Exception as e:
        print(f"读取遮罩图片出错，请检查路径: {e}")
        return

    # --- 第五步：生成词云图 ---
    font_path = "C:/Windows/Fonts/msyh.ttc" 

    wc = WordCloud(
        font_path=font_path,
        background_color='white',
        # 注意：有了mask后，width和height会由图片尺寸决定，但为了清晰度建议保留
        width=1000,
        height=800,
        max_words=200,                # 形状遮罩建议增加词量，否则填不满形状
        mask=mask_image,              # 【核心：加入形状】
        random_state=42,
        color_func=oriental_color_func, 
        contour_width=2,              # 【核心：增加外轮廓宽度】
        contour_color='#DAA520',      # 【核心：外轮廓颜色设为金色】
        prefer_horizontal=0.7 
    )

    wc.generate_from_frequencies(word_counts)

    # --- 第六步：显示与保存 ---
    plt.figure(figsize=(12, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off') 
    
    wc.to_file("huaxizi_shape_wordcloud.png")
    print("\n带形状的词云图已保存为: huaxizi_shape_wordcloud.png")
    
    plt.show()

if __name__ == "__main__":
    main()

    
    
    
    
    
    
import pandas as pd
import jieba
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def oriental_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = ['#B22222', '#DAA520', '#8B0000', '#FFD700', '#CD8539', '#A52A2A']
    return np.random.choice(colors)

def main():
    # --- 1. 路径设置 (请确保路径正确) ---
    file_path = r"D:\桌面应用\评论.xlsx" 
    mask_path = r"D:\桌面应用\ChatGPT Image 2026年3月11日 19_34_18.png"  # 你的并蒂莲图片
    font_path = "C:/Windows/Fonts/msyh.ttc" 

    # --- 2. 读取并处理遮罩图片 (关键改进) ---
    try:
        # 打开图片并转为 RGBA
        img = Image.open(mask_path).convert("RGBA")
        
        # 创建一个纯白背景的新图
        canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
        canvas.paste(img, (0, 0), img)
        
        # 强制将背景设为 255 (纯白)，非背景设为 0 (填色区)
        mask_image = np.array(canvas.convert("L")) # 转为灰度图
        # 阈值处理：让接近白色的地方变成 255，不填字
        mask_image = np.where(mask_image > 230, 255, 0) 
        
    except Exception as e:
        print(f"遮罩图片处理失败: {e}")
        return

    # --- 3. 读取评论数据 ---
    try:
        df = pd.read_excel(file_path)
        text_data = "".join(df['comment'].astype(str).tolist())
    except Exception as e:
        print(f"Excel读取失败: {e}")
        return

    # --- 4. 分词与清洗 ---
    jieba.add_word('花西子')
    jieba.add_word('东方美学')
    stopwords = set(['的', '了', '在', '是', '我', '有', '都', '和', '很', '就', '也', '太', '真的'])
    
    clean_text = "".join(re.findall(r'[\u4e00-\u9fa5]+', text_data))
    words = [w for w in jieba.lcut(clean_text) if w not in stopwords and len(w) > 1]
    word_counts = Counter(words)

    # --- 5. 生成词云 (参数优化) ---
    wc = WordCloud(
        font_path=font_path,
        background_color='white',
        mask=mask_image,            # 应用强制二值化的遮罩
        max_words=500,              # 增加词数以便填满花瓣细节
        max_font_size=150,
        min_font_size=5,
        random_state=42,
        color_func=oriental_color_func,
        contour_width=3,            # 描边加粗
        contour_color='#DAA520',    # 金色边框
        repeat=True                 # 如果词不够多，自动重复填充，确保形状完整
    )

    wc.generate_from_frequencies(word_counts)

    # --- 6. 绘图展示 ---
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    wc.to_file("huaxizi_shape_wordcloud.png")
    plt.show()

if __name__ == "__main__":
    main()
    
    
    
    
    
    
import pandas as pd
import jieba
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
from collections import Counter

# 1. 定义东方美学配色方案
def oriental_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    """
    自定义颜色函数：选取中国红、朱砂、琥珀、黛金等色调
    """
    colors = ['#B22222', '#DAA520', '#8B0000', '#FFD700', '#CD8539', '#A52A2A', '#E60012']
    return np.random.choice(colors)

def main():
    # --- 第一步：设置文件路径 (根据实际情况修改) ---
    file_path = r"D:\桌面应用\评论.xlsx"      # Excel文件路径
    mask_path = r"D:\桌面应用\ChatGPT Image 2026年3月11日 19_46_49.png"      # 并蒂莲Logo遮罩图路径（需白底黑图）
    font_path = 'C:/Windows/Fonts/simhei.ttf' # Windows字体路径，Mac请改为系统支持的字体路径

    # --- 第二步：读取数据 ---
    try:
        df = pd.read_excel(file_path)
        # 提取 comment 列并转换为长文本
        raw_text = "".join(df['comment'].astype(str).tolist())
    except Exception as e:
        print(f"读取文件出错: {e}")
        return

    # --- 第三步：数据清洗与分词 ---
    # 定义简单的停用词列表
    stopwords = {'的', '了', '在', '是', '我', '有', '都', '和', '这', '也', '很', '你', '说', '就', '人', '吧', '吗', '呢'}
    
    # 针对品牌特征，手动添加词库，防止核心词被切碎
    custom_keywords = ['花西子', '数字代言人', '东方美学', '设计感', '真实感', '建模']
    for word in custom_keywords:
        jieba.add_word(word)

    # 使用正则只保留中文字符（过滤掉标点、数字、英文）
    chinese_only = "".join(re.findall(r'[\u4e00-\u9fa5]+', raw_text))
    
    # 分词
    word_list = jieba.lcut(chinese_only)
    
    # 过滤词：去除停用词、去除长度小于2的词
    filtered_words = [w for w in word_list if w not in stopwords and len(w) >= 2]

    # --- 第四步：词频统计 ---
    word_counts = Counter(filtered_words)
    print("--- 出现频率最高的前50个词 ---")
    top_50 = word_counts.most_common(50)
    for i, (word, count) in enumerate(top_50, 1):
        print(f"{i}. {word}: {count}")

    # --- 第五步：图像遮罩处理 ---
    try:
        # 读取 Logo 形状图片并转化为 numpy 数组
        # 提示：图片背景必须为纯白色，Logo部分为深色/黑色
        mask_img = np.array(Image.open(mask_path))
    except Exception as e:
        print(f"读取 mask 图片出错: {e}")
        return

    # --- 第六步：生成词云图 ---
    wc = WordCloud(
        font_path=font_path,
        background_color='white',
        width=1000,
        height=800,
        mask=mask_img,              # 设置 Logo 形状遮罩
        max_words=200,              # 最大词语数量
        random_state=42,
        color_func=oriental_color_func, # 应用东方美学颜色
        contour_width=2,            # 增加轮廓线宽度
        contour_color='#DAA520'     # 轮廓线颜色（金色）
    )

    # 基于词频生成词云
    wc.generate_from_frequencies(word_counts)

    # --- 第七步：输出与保存 ---
    plt.figure(figsize=(12, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off') # 隐藏坐标轴
    
    # 保存图片
    wc.to_file("huaxizi_comment_wordcloud.png")
    print("\n✅ 词云图已成功生成并保存为: huaxizi_comment_wordcloud.png")
    
    plt.show()

if __name__ == "__main__":
    main()  
    
    
    
    
    
import pandas as pd
import jieba
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
from collections import Counter

# 1. 东方美学配色
def oriental_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = ['#B22222', '#DAA520', '#8B0000', '#FFD700', '#CD8539', '#A52A2A']
    return np.random.choice(colors)

def main():
    # --- 路径配置 (请确保路径正确) ---
    file_path = r"D:\桌面应用\评论.xlsx" 
    mask_path = r"D:\桌面应用\ChatGPT Image 2026年3月11日 19_46_49.png" # 你的并蒂莲形状图
    font_path = "C:/Windows/Fonts/msyh.ttc"   # 微软雅黑

    # --- 第一步：强制形状修复 (关键改进) ---
    try:
        # 打开图片并转为灰度图 ("L" 模式)
        mask_raw = Image.open(mask_path).convert("L")
        # 将接近白色的部分(>230)强制设为255(纯白)，其余设为0(填词区)
        mask_array = np.array(mask_raw)
        mask_final = np.where(mask_array > 230, 255, 0) 
    except Exception as e:
        print(f"处理形状图片出错: {e}")
        return

    # --- 第二步：读取数据 ---
    try:
        df = pd.read_excel(file_path)
        text_data = "".join(df['comment'].astype(str).tolist())
    except Exception as e:
        print(f"读取Excel出错: {e}")
        return

    # --- 第三步：清洗与分词 ---
    stopwords = {'的', '了', '在', '是', '我', '有', '都', '和', '很', '就', '也', '太', '真的', '感觉'}
    jieba.add_word('花西子')
    jieba.add_word('东方美学')
    
    clean_text = "".join(re.findall(r'[\u4e00-\u9fa5]+', text_data))
    words = [w for w in jieba.lcut(clean_text) if w not in stopwords and len(w) >= 2]
    word_counts = Counter(words)

    # --- 第四步：生成词云 (优化参数) ---
    wc = WordCloud(
        font_path=font_path,
        background_color='white',
        mask=mask_final,             # 使用修复后的遮罩矩阵
        max_words=500,               # 调高词数以便填满花瓣细节
        repeat=True,                 # 如果词不够，自动重复填充以填满形状
        random_state=42,
        color_func=oriental_color_func,
        contour_width=3,             # 描边加粗，让形状更明显
        contour_color='#DAA520'      # 金色边框
    )

    wc.generate_from_frequencies(word_counts)

    # --- 第五步：展示与保存 ---
    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    wc.to_file("huaxizi_shape_wordcloud.png")
    print("\n✅ 带形状的词云图已生成！")
    plt.show()

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    