import requests
from bs4 import BeautifulSoup
import json
import time

# 基本URL
base_url = "https://finance.eastmoney.com/a/ccjdd.html?page="

# 结果存储
all_news = []

# 循环遍历50页
for page in range(1, 51):
    print(f"正在抓取第{page}页...")
    url = base_url + str(page)
    
    # 发送请求并获取页面内容
    response = requests.get(url)
    if response.status_code != 200:
        print(f"请求失败，状态码: {response.status_code}")
        continue
    
    # 解析HTML
    soup = BeautifulSoup(response.content, 'lxml')
    
    # 提取新闻条目
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))
    
    for item in news_items:
        # 获取新闻标题
        title = item.find('p', class_='title').get_text().strip()  # 提取标题
        # 获取新闻内容（摘要）
        content = item.find('p', class_='info').get_text().strip()  # 提取内容
        # 获取发布时间
        publish_time = item.find('p', class_='time').get_text().strip()  # 提取时间
        # 获取新闻的链接
        link = item.find('a')['href']  # 获取新闻的详细链接
        
        # 默认评论人数为0（若有评论页可以进一步提取）
        comments_count = 0
        
        # 保存数据到字典
        news_data = {
            "title": title,
            "content": content,
            "publish_time": publish_time,
            "source": "EastMoney",  # 来源固定为“EastMoney”
            "comments_count": comments_count,
            "link": link  # 可选：如果你想保存新闻详情链接
        }
        
        all_news.append(news_data)
    
    # 暂停1秒，防止过于频繁的请求
    time.sleep(1)

# 保存为JSON文件
with open('news_data.json', 'w', encoding='utf-8') as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

print("抓取完成，数据已保存为 news_data.json")


        
        
        
        
        
import json

# 读取并查看数据
with open('news_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 打印输出前10条新闻数据
for news in data[:10]:
    print(news)
       
        
        
import os

# 打印当前工作目录
print("当前工作目录:", os.getcwd())
      











import requests
from bs4 import BeautifulSoup
import json
import time

# 基本URL
base_url = "https://finance.eastmoney.com/a/ccjdd.html?page="

# 结果存储
all_news = []

# 循环遍历50页
for page in range(1, 51):
    print(f"正在抓取第{page}页...")
    url = base_url + str(page)
    
    # 发送请求并获取页面内容
    response = requests.get(url)
    print(f"请求 URL: {url}, 状态码: {response.status_code}")
    
    if response.status_code != 200:
        print(f"请求失败，状态码: {response.status_code}")
        continue
    
    # 解析HTML
    soup = BeautifulSoup(response.content, 'lxml')
    
    # 提取新闻条目
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))

    if not news_items:
        print(f"没有找到新闻条目。检查选择器是否正确。")
        continue
    
    print(f"成功找到新闻条目数量: {len(news_items)}")
    
    for item in news_items:
        # 获取新闻标题
        title = item.find('p', class_='title').get_text().strip()  # 提取标题
        # 获取新闻内容（摘要）
        content = item.find('p', class_='info').get_text().strip()  # 提取内容
        # 获取发布时间
        publish_time = item.find('p', class_='time').get_text().strip()  # 提取时间
        # 获取新闻的链接
        link = item.find('a')['href']  # 获取新闻的详细链接
        
        # 默认评论人数为0（若有评论页可以进一步提取）
        comments_count = 0
        
        # 保存数据到字典
        news_data = {
            "title": title,
            "content": content,
            "publish_time": publish_time,
            "source": "EastMoney",  # 来源固定为“EastMoney”
            "comments_count": comments_count,
            "link": link  # 可选：如果你想保存新闻详情链接
        }
        
        all_news.append(news_data)
    
    # 暂停1秒，防止过于频繁的请求
    time.sleep(1)

# 确认是否有数据，保存为JSON文件
if all_news:
    with open('news_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_news, f, ensure_ascii=False, indent=4)
    print("数据已成功保存为 news_data.json")
else:
    print("没有抓取到任何数据")








from selenium import webdriver
from bs4 import BeautifulSoup

# 使用 Selenium 模拟浏览器获取页面内容
driver = webdriver.Chrome()  # 或使用 Firefox、Edge
driver.get("https://finance.eastmoney.com/a/ccjdd.html?page=1")

# 获取页面源代码
html = driver.page_source

# 使用 BeautifulSoup 解析 HTML
soup = BeautifulSoup(html, 'lxml')

# 查找新闻条目
news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))

if news_items:
    print(f"成功找到新闻条目数量: {len(news_items)}")
else:
    print("没有找到新闻条目")




for item in news_items[:3]:  # 打印前三条新闻内容
    title = item.find('p', class_='title').get_text().strip()
    content = item.find('p', class_='info').get_text().strip()
    publish_time = item.find('p', class_='time').get_text().strip()
    link = item.find('a')['href']
    
    print(f"标题: {title}")
    print(f"内容: {content}")
    print(f"发布时间: {publish_time}")
    print(f"链接: {link}")
    print("="*50)






import time
import json
from selenium import webdriver
from bs4 import BeautifulSoup

# 设置 Selenium WebDriver（确保 chromedriver 可用）
driver = webdriver.Chrome()

# 用于保存所有新闻的列表
all_news = []

# 抓取50页的数据
for page in range(1, 51):  # 循环抓取50页
    url = f"https://finance.eastmoney.com/a/ccjdd.html?page={page}"
    driver.get(url)
    
    # 等待页面加载（根据需要调整等待时间，或使用 WebDriverWait）
    driver.implicitly_wait(5)  # 等待 5 秒
    
    # 解析页面内容
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    # 查找所有新闻条目
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))
    
    if news_items:
        print(f"第{page}页抓取成功，共抓取到 {len(news_items)} 条新闻。")
        
        # 遍历每条新闻并提取数据
        for item in news_items:
            title = item.find('p', class_='title').get_text().strip()
            content = item.find('p', class_='info').get_text().strip()
            publish_time = item.find('p', class_='time').get_text().strip()
            link = item.find('a')['href']
            
            # 构建新闻数据字典
            news_data = {
                "title": title,
                "content": content,
                "publish_time": publish_time,
                "source": "EastMoney",  # 可以根据实际需要调整
                "comments_count": 0,  # 评论数默认为0，可能需要进一步处理
                "link": link
            }
            
            # 将新闻数据添加到总列表中
            all_news.append(news_data)
    
    else:
        print(f"第{page}页没有找到新闻条目，可能页面结构发生了变化。")
    
    # 每页抓取完成后等待 1 秒
    time.sleep(1)

# 完成所有页面抓取后，将数据保存为 JSON 文件
with open('news_data.json', 'w', encoding='utf-8') as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

# 打印完成消息
print("所有数据已保存到 news_data.json")

# 关闭浏览器
driver.quit()











import time
import json
from selenium import webdriver
from bs4 import BeautifulSoup

# 设置 Selenium WebDriver（确保 chromedriver 可用）
driver = webdriver.Chrome()

# 用于保存所有新闻的列表
all_news = []

# 抓取50页的数据
for page in range(1, 51):  # 循环抓取50页
    url = f"https://finance.eastmoney.com/a/ccjdd.html?page={page}"
    driver.get(url)
    
    # 等待页面加载（根据需要调整等待时间，或使用 WebDriverWait）
    driver.implicitly_wait(5)  # 等待 5 秒
    
    # 解析页面内容
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    # 查找所有新闻条目
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))
    
    if news_items:
        print(f"第{page}页抓取成功，共抓取到 {len(news_items)} 条新闻。")
        
        # 遍历每条新闻并提取数据
        for item in news_items:
            title = item.find('p', class_='title').get_text().strip()
            content = item.find('p', class_='info').get_text().strip()
            publish_time = item.find('p', class_='time').get_text().strip()
            link = item.find('a')['href']
            
            # 尝试抓取评论数
            comment_count = 0  # 默认评论数为0
            comment_element = item.find('div', class_='level1_item')  # 根据评论条目的 div 查找
            if comment_element:
                # 从 data-reply_count 获取评论数
                comment_count = int(comment_element.get('data-reply_count', 0))
            
            # 构建新闻数据字典
            news_data = {
                "title": title,
                "content": content,
                "publish_time": publish_time,
                "source": "EastMoney",  # 可以根据实际需要调整
                "comments_count": comment_count,  # 更新评论数
                "link": link
            }
            
            # 将新闻数据添加到总列表中
            all_news.append(news_data)
    
    else:
        print(f"第{page}页没有找到新闻条目，可能页面结构发生了变化。")
    
    # 每页抓取完成后等待 1 秒
    time.sleep(1)

# 完成所有页面抓取后，将数据保存为 JSON 文件
with open('news_data.json', 'w', encoding='utf-8') as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

# 打印完成消息
print("所有数据已保存到 news_data.json")

# 关闭浏览器
driver.quit()











import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# 设置 Selenium WebDriver（确保 chromedriver 可用）
driver = webdriver.Chrome()

# 用于保存所有新闻的列表
all_news = []

# 等待页面加载的函数
def wait_for_page_to_load():
    try:
        # 等待页面中的某个特定元素加载完毕
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'li[id^="newsTr"]'))
        )
    except Exception as e:
        print(f"页面加载超时或出错: {e}")

# 抓取评论的函数
def get_comments(news_url):
    # 尝试访问新闻详情页面并抓取评论
    driver.get(news_url)
    time.sleep(2)  # 等待页面加载

    # 解析页面内容
    soup = BeautifulSoup(driver.page_source, 'lxml')

    # 查找评论区域，评论的具体结构可能会有所不同，这里假设评论在某个特定的 div 中
    comments = []
    comment_elements = soup.find_all('div', class_='comment-item')  # 假设评论项在 class='comment-item' 的 div 中
    for comment in comment_elements:
        username = comment.find('span', class_='comment-author').get_text(strip=True)
        content = comment.find('p', class_='comment-content').get_text(strip=True)
        comments.append({
            "username": username,
            "content": content
        })

    # 获取评论总数
    comment_count = len(comments)
    
    return comments, comment_count

# 抓取50页的数据
for page in range(1, 51):  # 循环抓取50页
    url = f"https://finance.eastmoney.com/a/ccjdd.html?page={page}"
    driver.get(url)
    
    # 等待页面加载（等待特定的元素出现）
    wait_for_page_to_load()
    
    # 解析页面内容
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    # 查找所有新闻条目
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))
    
    if news_items:
        print(f"第{page}页抓取成功，共抓取到 {len(news_items)} 条新闻。")
        
        # 遍历每条新闻并提取数据
        for item in news_items:
            title = item.find('p', class_='title').get_text().strip()
            content = item.find('p', class_='info').get_text().strip()
            publish_time = item.find('p', class_='time').get_text().strip()
            link = item.find('a')['href']
            
            # 尝试抓取评论数
            comment_count = 0  # 默认评论数为0
            comments = []
            comment_element = item.find('div', class_='level1_item')  # 根据评论条目的 div 查找
            if comment_element:
                # 从 data-reply_count 获取评论数
                comment_count = int(comment_element.get('data-reply_count', 0))
                if comment_count > 0:
                    # 如果有评论数，则抓取评论
                    comments, comment_count = get_comments(link)

            # 构建新闻数据字典
            news_data = {
                "title": title,
                "content": content,
                "publish_time": publish_time,
                "source": "EastMoney",  # 可以根据实际需要调整
                "comments_count": comment_count,  # 更新评论数
                "link": link,
                "comments": comments  # 保存评论内容
            }
            
            # 将新闻数据添加到总列表中
            all_news.append(news_data)
    
    else:
        print(f"第{page}页没有找到新闻条目，可能页面结构发生了变化。")
    
    # 每页抓取完成后等待 1 秒
    time.sleep(1)

# 完成所有页面抓取后，将数据保存为 JSON 文件
with open('news_data_with_comments.json', 'w', encoding='utf-8') as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

# 打印完成消息
print("所有数据已保存到 news_data_with_comments.json")

# 关闭浏览器
driver.quit()













import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# 设置 Selenium WebDriver（确保 chromedriver 可用）
driver = webdriver.Chrome()

# 用于保存所有新闻的列表
all_news = []

# 等待页面加载的函数
def wait_for_page_to_load():
    try:
        # 等待页面中的某个特定元素加载完毕
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'li[id^="newsTr"]'))
        )
    except Exception as e:
        print(f"页面加载超时或出错: {e}")

# 用 Selenium 获取并点击“加载更多评论”按钮
def load_all_comments(news_url):
    driver.get(news_url)
    time.sleep(2)  # 等待页面加载
    
    # 假设评论区域有“加载更多”按钮，模拟点击按钮
    while True:
        try:
            # 找到“加载更多”按钮并点击
            load_more_button = driver.find_element(By.CSS_SELECTOR, '.load-more-comments')  # 假设是这个类名
            load_more_button.click()
            time.sleep(2)  # 等待更多评论加载
        except Exception as e:
            print(f"没有更多评论按钮或加载完毕: {e}")
            break

    # 返回加载完的页面内容
    return driver.page_source

# 抓取评论的函数
def get_comments(news_url):
    # 获取完整的页面并加载所有评论
    page_source = load_all_comments(news_url)
    
    # 解析页面内容
    soup = BeautifulSoup(page_source, 'lxml')
    
    comments = []
    comment_elements = soup.find_all('div', class_='comment-item')  # 根据实际页面结构调整
    for comment in comment_elements:
        username = comment.find('span', class_='comment-author').get_text(strip=True)
        content = comment.find('p', class_='comment-content').get_text(strip=True)
        comments.append({
            "username": username,
            "content": content
        })

    # 获取评论总数
    comment_count = len(comments)
    
    return comments, comment_count

# 抓取50页的数据
for page in range(1, 51):  # 循环抓取50页
    url = f"https://finance.eastmoney.com/a/ccjdd.html?page={page}"
    driver.get(url)
    
    # 等待页面加载（等待特定的元素出现）
    wait_for_page_to_load()
    
    # 解析页面内容
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    # 查找所有新闻条目
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))
    
    if news_items:
        print(f"第{page}页抓取成功，共抓取到 {len(news_items)} 条新闻。")
        
        # 遍历每条新闻并提取数据
        for item in news_items:
            title = item.find('p', class_='title').get_text().strip()
            content = item.find('p', class_='info').get_text().strip()
            publish_time = item.find('p', class_='time').get_text().strip()
            link = item.find('a')['href']
            
            # 尝试抓取评论数
            comment_count = 0  # 默认评论数为0
            comments = []
            comment_element = item.find('div', class_='level1_item')  # 根据评论条目的 div 查找
            if comment_element:
                # 从 data-reply_count 获取评论数
                comment_count = int(comment_element.get('data-reply_count', 0))
                if comment_count > 0:
                    # 如果有评论数，则抓取评论
                    comments, comment_count = get_comments(link)

            # 构建新闻数据字典
            news_data = {
                "title": title,
                "content": content,
                "publish_time": publish_time,
                "source": "EastMoney",  # 可以根据实际需要调整
                "comments_count": comment_count,  # 更新评论数
                "link": link,
                "comments": comments  # 保存评论内容
            }
            
            # 将新闻数据添加到总列表中
            all_news.append(news_data)
    
    else:
        print(f"第{page}页没有找到新闻条目，可能页面结构发生了变化。")
    
    # 每页抓取完成后等待 1 秒
    time.sleep(1)

# 完成所有页面抓取后，将数据保存为 JSON 文件
with open('news_data_with_comments.json', 'w', encoding='utf-8') as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

# 打印完成消息
print("所有数据已保存到 news_data_with_comments1.json")

# 关闭浏览器
driver.quit()








import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# 设置 Selenium WebDriver（确保 chromedriver 可用）
driver = webdriver.Chrome()

# 用于保存所有新闻的列表
all_news = []

# 等待页面加载的函数
def wait_for_page_to_load():
    try:
        # 等待页面中的某个特定元素加载完毕
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'li[id^="newsTr"]'))
        )
    except Exception as e:
        print(f"页面加载超时或出错: {e}")

# 抓取评论的函数
def get_comments(news_url):
    # 尝试访问新闻详情页面并抓取评论
    driver.get(news_url)
    time.sleep(2)  # 等待页面加载

    # 解析页面内容
    soup = BeautifulSoup(driver.page_source, 'lxml')

    # 查找评论区域，评论的具体结构可能会有所不同，这里假设评论在某个特定的 div 中
    comments = []
    comment_elements = soup.find_all('div', class_='comment-item')  # 假设评论项在 class='comment-item' 的 div 中
    for comment in comment_elements:
        username = comment.find('span', class_='comment-author').get_text(strip=True)
        content = comment.find('p', class_='comment-content').get_text(strip=True)
        comments.append({
            "username": username,
            "content": content
        })

    # 获取评论总数
    comment_count = len(comments)
    
    return comments, comment_count

# 抓取50页的数据
for page in range(1, 51):  # 循环抓取50页
    url = f"https://finance.eastmoney.com/a/ccjdd.html?page={page}"
    driver.get(url)
    
    # 等待页面加载（等待特定的元素出现）
    wait_for_page_to_load()
    
    # 解析页面内容
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    # 查找所有新闻条目
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))
    
    if news_items:
        print(f"第{page}页抓取成功，共抓取到 {len(news_items)} 条新闻。")
        
        # 遍历每条新闻并提取数据
        for item in news_items:
            title = item.find('p', class_='title').get_text().strip()
            content = item.find('p', class_='info').get_text().strip()
            publish_time = item.find('p', class_='time').get_text().strip()
            link = item.find('a')['href']
            
            # 尝试抓取评论数
            comment_count = 0  # 默认评论数为0
            comments = []
            comment_element = item.find('div', class_='level1_item')  # 根据评论条目的 div 查找
            if comment_element:
                # 从 data-reply_count 获取评论数
                comment_count = int(comment_element.get('data-reply_count', 0))
                if comment_count > 0:
                    # 如果有评论数，则抓取评论
                    comments, comment_count = get_comments(link)

            # 构建新闻数据字典
            news_data = {
                "title": title,
                "content": content,
                "publish_time": publish_time,
                "source": "EastMoney",  # 可以根据实际需要调整
                "comments_count": comment_count,  # 更新评论数
                "link": link,
                "comments": comments  # 保存评论内容
            }
            
            # 将新闻数据添加到总列表中
            all_news.append(news_data)
    
    else:
        print(f"第{page}页没有找到新闻条目，可能页面结构发生了变化。")
    
    # 每页抓取完成后等待 1 秒
    time.sleep(1)

# 完成所有页面抓取后，将数据保存为 JSON 文件
with open('news_data_with_comments.json', 'w', encoding='utf-8') as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

# 打印完成消息
print("所有数据已保存到 news_data_with_comments.json")

# 关闭浏览器
driver.quit()
















import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
import time

# 设置 WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# 新闻列表页 URL
base_url = "https://finance.eastmoney.com/a/ccjdd.html?page="

# 存储抓取的数据
news_data = []

# 抓取前 5 页的新闻数据（可以修改为 50 页）
for page in range(1, 6):
    print(f"正在抓取第{page}页...")
    url = base_url + str(page)
    driver.get(url)
    time.sleep(2)  # 等待页面加载

    # 等待新闻列表加载
    WebDriverWait(driver, 30).until(
    EC.element_to_be_clickable((By.XPATH, '//element_xpath')))



    # 获取新闻列表
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    news_items = soup.find_all('li', class_='list_item')

    # 遍历每条新闻
    for item in news_items:
        title = item.find('p', class_='title').get_text()
        content = item.find('p', class_='info').get_text()
        publish_time = item.find('p', class_='time').get_text()
        source = "EastMoney"  # 固定值，或根据页面提取
        link = item.find('a')['href']
        
        # 访问新闻详情页
        driver.get(link)
        time.sleep(2)  # 等待页面加载

        # 获取评论数
        soup_detail = BeautifulSoup(driver.page_source, 'html.parser')
        comment_count = 0
        comment_section = soup_detail.find('div', class_='level1_item')
        if comment_section:
            comment_count = int(comment_section.get('data-reply_count', 0))

        # 保存新闻数据
        news_data.append({
            "title": title,
            "content": content,
            "publish_time": publish_time,
            "source": source,
            "comments_count": comment_count,
            "link": link
        })
    
    print(f"第{page}页抓取完成")

# 关闭 WebDriver
driver.quit()

# 保存数据为 JSON 文件
with open('news_data_with_comments.json', 'w', encoding='utf-8') as f:
    json.dump(news_data, f, ensure_ascii=False, indent=4)

print("抓取完成，数据已保存到 news_data_with_comments.json")











import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# 设置 Selenium WebDriver（确保 chromedriver 可用）
driver = webdriver.Chrome()

# 用于保存所有新闻的列表
all_news = []

# 抓取50页的数据
for page in range(1, 51):  # 循环抓取50页
    url = f"https://finance.eastmoney.com/a/ccjdd.html?page={page}"
    driver.get(url)
    
    # 等待页面完全加载（等待特定元素的出现）
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'li#newsTr')))
    except Exception as e:
        print(f"第{page}页加载失败：{e}")
        continue  # 跳过此页，继续抓取下一页
    
    # 解析页面内容
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    # 查找所有新闻条目
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))
    
    if news_items:
        print(f"第{page}页抓取成功，共抓取到 {len(news_items)} 条新闻。")
        
        # 遍历每条新闻并提取数据
        for item in news_items:
            title = item.find('p', class_='title').get_text().strip()
            content = item.find('p', class_='info').get_text().strip()
            publish_time = item.find('p', class_='time').get_text().strip()
            link = item.find('a')['href']
            
            # 尝试抓取评论数
            comment_count = 0  # 默认评论数为0
            comment_element = item.find('div', class_='level1_item')  # 根据评论条目的 div 查找
            if comment_element:
                # 从 data-reply_count 获取评论数
                comment_count = int(comment_element.get('data-reply_count', 0))
            
            # 进入每篇新闻详情页抓取评论数据
            driver.get(link)  # 进入新闻详情页
            time.sleep(2)  # 等待页面加载
            
            # 等待评论区加载（等待特定元素的出现）
            try:
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.comment-list')))
            except Exception as e:
                print(f"新闻 {title} 评论加载失败：{e}")
                continue  # 如果评论区加载失败，则跳过此新闻
            
            # 解析新闻详情页
            soup_detail = BeautifulSoup(driver.page_source, 'lxml')
            comments = []
            
            # 获取评论内容和作者
            comment_elements = soup_detail.find_all('div', class_='comment-item')
            for comment in comment_elements:
                comment_text = comment.find('div', class_='comment-text').get_text().strip()
                comment_author = comment.find('span', class_='user-name').get_text().strip()
                comment_time = comment.find('span', class_='comment-time').get_text().strip()
                
                comments.append({
                    'author': comment_author,
                    'text': comment_text,
                    'time': comment_time
                })
            
            # 构建新闻数据字典
            news_data = {
                "title": title,
                "content": content,
                "publish_time": publish_time,
                "source": "EastMoney",  # 可以根据实际需要调整
                "comments_count": comment_count,  # 更新评论数
                "link": link,
                "comments": comments  # 包含评论数据
            }
            
            # 将新闻数据添加到总列表中
            all_news.append(news_data)
    
    else:
        print(f"第{page}页没有找到新闻条目，可能页面结构发生了变化。")
    
    # 每页抓取完成后等待 1 秒
    time.sleep(1)

# 完成所有页面抓取后，将数据保存为 JSON 文件
with open('news_data_with_comments.json', 'w', encoding='utf-8') as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

# 打印完成消息
print("所有数据已保存到 news_data_with_comments.json")

# 关闭浏览器
driver.quit()













import time
import json
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# 设置无头浏览器（headless）
chrome_options = Options()
chrome_options.add_argument("--headless")  # 启用无头模式
chrome_options.add_argument("--disable-gpu")  # 禁用GPU加速（在某些系统中可提高稳定性）
chrome_options.add_argument("--no-sandbox")  # 对于某些Linux环境可能需要

# 设置 Selenium WebDriver（确保 chromedriver 可用）
driver = webdriver.Chrome(options=chrome_options)

# 用于保存所有新闻的列表
all_news = []

# 定义一个重试函数
def load_page_with_retry(url, retries=3):
    """尝试加载页面并在失败时重试"""
    for attempt in range(retries):
        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'li#newsTr')))
            return True  # 页面加载成功
        except Exception as e:
            print(f"加载失败，尝试第{attempt + 1}次：{e}")
            if attempt < retries - 1:
                time.sleep(random.uniform(2, 5))  # 随机等待2到5秒，避免过快重试
            else:
                print(f"加载页面失败：{url}")
                return False  # 重试次数超过限制，返回加载失败

# 抓取50页的数据
for page in range(1, 51):  # 循环抓取50页
    url = f"https://finance.eastmoney.com/a/ccjdd.html?page={page}"
    
    # 使用重试机制加载页面
    if not load_page_with_retry(url):
        continue  # 如果加载失败，跳过当前页面，继续下一个页面
    
    # 解析页面内容
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    # 查找所有新闻条目
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))
    
    if news_items:
        print(f"第{page}页抓取成功，共抓取到 {len(news_items)} 条新闻。")
        
        # 遍历每条新闻并提取数据
        for item in news_items:
            title = item.find('p', class_='title').get_text().strip()
            content = item.find('p', class_='info').get_text().strip()
            publish_time = item.find('p', class_='time').get_text().strip()
            link = item.find('a')['href']
            
            # 尝试抓取评论数
            comment_count = 0  # 默认评论数为0
            comment_element = item.find('div', class_='level1_item')  # 根据评论条目的 div 查找
            if comment_element:
                # 从 data-reply_count 获取评论数
                comment_count = int(comment_element.get('data-reply_count', 0))
            
            # 进入每篇新闻详情页抓取评论数据
            driver.get(link)  # 进入新闻详情页
            time.sleep(2)  # 等待页面加载
            
            # 等待评论区加载（等待特定元素的出现）
            try:
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.comment-list')))
            except Exception as e:
                print(f"新闻 {title} 评论加载失败：{e}")
                continue  # 如果评论区加载失败，则跳过此新闻
            
            # 解析新闻详情页
            soup_detail = BeautifulSoup(driver.page_source, 'lxml')
            comments = []
            
            # 获取评论内容和作者
            comment_elements = soup_detail.find_all('div', class_='comment-item')
            for comment in comment_elements:
                comment_text = comment.find('div', class_='comment-text').get_text().strip()
                comment_author = comment.find('span', class_='user-name').get_text().strip()
                comment_time = comment.find('span', class_='comment-time').get_text().strip()
                
                comments.append({
                    'author': comment_author,
                    'text': comment_text,
                    'time': comment_time
                })
            
            # 构建新闻数据字典
            news_data = {
                "title": title,
                "content": content,
                "publish_time": publish_time,
                "source": "EastMoney",  # 可以根据实际需要调整
                "comments_count": comment_count,  # 更新评论数
                "link": link,
                "comments": comments  # 包含评论数据
            }
            
            # 将新闻数据添加到总列表中
            all_news.append(news_data)
    
    else:
        print(f"第{page}页没有找到新闻条目，可能页面结构发生了变化。")
    
    # 每页抓取完成后等待 1 秒
    time.sleep(1)

# 完成所有页面抓取后，将数据保存为 JSON 文件
with open('news_data_with_comments.json', 'w', encoding='utf-8') as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

# 打印完成消息
print("所有数据已保存到 news_data_with_comments.json")

# 关闭浏览器
driver.quit()







import time
import json
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# 设置无头浏览器（headless）
chrome_options = Options()
chrome_options.add_argument("--headless")  # 启用无头模式
chrome_options.add_argument("--disable-gpu")  # 禁用GPU加速（在某些系统中可提高稳定性）
chrome_options.add_argument("--no-sandbox")  # 对于某些Linux环境可能需要

# 设置Selenium的浏览器日志捕获
chrome_options.add_argument("--log-level=3")  # 仅输出错误日志

# 设置 Selenium WebDriver（确保 chromedriver 可用）
driver = webdriver.Chrome(options=chrome_options)

# 用于保存所有新闻的列表
all_news = []

# 定义一个重试函数
def load_page_with_retry(url, retries=3):
    """尝试加载页面并在失败时重试"""
    for attempt in range(retries):
        try:
            driver.get(url)
            # 尝试更长的等待时间，等待页面加载
            WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'ul#newsList')))  # 更稳定的选择器
            return True  # 页面加载成功
        except Exception as e:
            print(f"加载失败，尝试第{attempt + 1}次：{e}")
            if attempt < retries - 1:
                time.sleep(random.uniform(5, 10))  # 增加更长的等待时间，避免过快重试
            else:
                print(f"加载页面失败：{url}")
                return False  # 重试次数超过限制，返回加载失败

# 抓取50页的数据
for page in range(1, 51):  # 循环抓取50页
    url = f"https://finance.eastmoney.com/a/ccjdd.html?page={page}"
    
    # 使用重试机制加载页面
    if not load_page_with_retry(url):
        continue  # 如果加载失败，跳过当前页面，继续下一个页面
    
    # 解析页面内容
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    # 查找所有新闻条目
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))
    
    if news_items:
        print(f"第{page}页抓取成功，共抓取到 {len(news_items)} 条新闻。")
        
        # 遍历每条新闻并提取数据
        for item in news_items:
            title = item.find('p', class_='title').get_text().strip()
            content = item.find('p', class_='info').get_text().strip()
            publish_time = item.find('p', class_='time').get_text().strip()
            link = item.find('a')['href']
            
            # 尝试抓取评论数
            comment_count = 0  # 默认评论数为0
            comment_element = item.find('div', class_='level1_item')  # 根据评论条目的 div 查找
            if comment_element:
                # 从 data-reply_count 获取评论数
                comment_count = int(comment_element.get('data-reply_count', 0))
            
            # 进入每篇新闻详情页抓取评论数据
            driver.get(link)  # 进入新闻详情页
            time.sleep(3)  # 等待页面加载
            
            # 等待评论区加载（等待特定元素的出现）
            try:
                WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.comment-list')))
            except Exception as e:
                print(f"新闻 {title} 评论加载失败：{e}")
                continue  # 如果评论区加载失败，则跳过此新闻
            
            # 解析新闻详情页
            soup_detail = BeautifulSoup(driver.page_source, 'lxml')
            comments = []
            
            # 获取评论内容和作者
            comment_elements = soup_detail.find_all('div', class_='comment-item')
            for comment in comment_elements:
                comment_text = comment.find('div', class_='comment-text').get_text().strip()
                comment_author = comment.find('span', class_='user-name').get_text().strip()
                comment_time = comment.find('span', class_='comment-time').get_text().strip()
                
                comments.append({
                    'author': comment_author,
                    'text': comment_text,
                    'time': comment_time
                })
            
            # 构建新闻数据字典
            news_data = {
                "title": title,
                "content": content,
                "publish_time": publish_time,
                "source": "EastMoney",  # 可以根据实际需要调整
                "comments_count": comment_count,  # 更新评论数
                "link": link,
                "comments": comments  # 包含评论数据
            }
            
            # 将新闻数据添加到总列表中
            all_news.append(news_data)
    
    else:
        print(f"第{page}页没有找到新闻条目，可能页面结构发生了变化。")
    
    # 每页抓取完成后等待 1 秒
    time.sleep(random.uniform(3, 5))  # 随机等待

# 完成所有页面抓取后，将数据保存为 JSON 文件
with open('news_data_with_comments.json', 'w', encoding='utf-8') as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

# 打印完成消息
print("所有数据已保存到 news_data_with_comments.json")

# 关闭浏览器
driver.quit()














import requests
from bs4 import BeautifulSoup
import json
import re

# 抓取新闻信息
def get_news_data(news_url):
    response = requests.get(news_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 安全地获取标题
    title_tag = soup.find('p', class_='title')
    title = title_tag.find('a').text.strip() if title_tag else '未找到标题'
    
    # 同样处理其他信息
    content_tag = soup.find('div', class_='content')
    content = content_tag.text.strip() if content_tag else '未找到内容'
    
    return {
        'title': title,
        'content': content,
        # 继续抓取其他信息
    }



# 抓取评论数据
def get_comments(news_id):
    # 拼接评论接口 URL
    comment_url = f'https://6.newspush.eastmoney.com/sse?cb=icomet_cb_0&cname={news_id}&seq=92850&noop=237&token=&_={str(int(time.time() * 1000))}'

    # 请求评论接口
    response = requests.get(comment_url)
    
    # 解析 JSONP 格式的数据 (需要提取 JSON 部分)
    jsonp_data = response.text
    json_data = re.search(r'icomet_cb_0\((.*)\)', jsonp_data).group(1)
    
    # 转换为字典
    comments_data = json.loads(json_data)

    # 提取评论内容
    comments = []
    for comment in comments_data.get('data', {}).get('list', []):
        comment_text = comment.get('content', '')
        comment_time = comment.get('create_time', '')
        comments.append({
            'text': comment_text,
            'time': comment_time
        })

    return comments

# 综合抓取新闻和评论
def get_news_and_comments(news_url, news_id):
    # 获取新闻信息
    news_data = get_news_data(news_url)
    
    # 获取评论信息
    comments = get_comments(news_id)
    
    # 将评论数据添加到新闻数据中
    news_data['comments'] = comments
    
    return news_data

# 示例新闻 URL
news_url = 'https://finance.eastmoney.com/a/202511173566113864.html'

# 从新闻页面中提取出 news_id 或者可以通过 URL 提取
news_id = 'bdc02c361aab973818f3583fb8b5e6d5'  # 假设你已经提取了正确的 ID

# 获取新闻和评论数据
news_and_comments = get_news_and_comments(news_url, news_id)

# 打印抓取到的新闻和评论数据
print(json.dumps(news_and_comments, ensure_ascii=False, indent=4))










import requests
from bs4 import BeautifulSoup

# 请求头，模拟浏览器访问
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# 目标新闻页面 URL
url = 'https://finance.eastmoney.com/a/ccjdd.html'

# 发送请求获取页面内容
response = requests.get(url, headers=headers)
response.raise_for_status()  # 检查请求是否成功

# 解析 HTML 内容
soup = BeautifulSoup(response.content, 'html.parser')

# 提取新闻标题
title = soup.find('h1', class_='article-title').get_text(strip=True)

# 提取新闻发布时间
publish_time = soup.find('div', class_='time-source').get_text(strip=True)

# 提取新闻来源
source = soup.find('div', class_='source').get_text(strip=True)

# 提取新闻内容
content = soup.find('div', class_='article').get_text("\n", strip=True)

# 提取评论人数
comments_section = soup.find('span', class_='comment-num')
comments_count = int(comments_section.get_text(strip=True)) if comments_section else 0

# 打印抓取的结果
print(f"标题: {title}")
print(f"发布时间: {publish_time}")
print(f"来源: {source}")
print(f"内容: {content}")
print(f"评论人数: {comments_count}")



