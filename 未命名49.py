import requests
import csv


f = open('b.csv',mode='a',newline='',encoding='utf-8')
csv_writer = csv.writer(f)

csv_writer.writerow(['jobName','salaryDesc', 'bossName','brandName','cityName', 'areaDistrict', 'jobExperience','jobDegree'])

headers={
    'Cookie':'lastCity=101030100; ab_guid=f15d4127-6c3a-49c0-be9a-2cb0f549f738; wt2=DFGOALq23w-0wEMOPVsjnzAo8HrBjl6MxBNQg1skbrHVT4ynFedk2tYDcy2jodvK0ZqO854IAfD-s1cTAkvp6NQ~~; wbg=0; zp_at=uVevtNEGmGxDPGeJetYlNbRZMNwWwomqP0kAYloGAHs~; __zp_seo_uuid__=30a4f1d1-ff2d-4a34-89cc-23cc22bad3ed; __l=r=https%3A%2F%2Fwww.google.com%2F&l=%2F&s=1; __g=-; Hm_lvt_194df3105ad7148dcf2b98a91b5e727a=1766238834,1766504279; HMACCOUNT=6D5DCEC5E098F33C; Hm_lpvt_194df3105ad7148dcf2b98a91b5e727a=1766545565; __zp_stoken__=cdcfgOj3DjsK7woHCvUghejs6IUkwPDo9Oz46Oj14LkgxPjAzOx%2FCmcK6VcO6dFXDgBDCrzsgOzAwOj1JMDM3EDs0SDE9MDFIxLDCvkgxSMORw709IB7ClcK8VsSJfVLDjgx5wrMNw6rCsA%2FDrcKxCsOQwrsEw6PCviErAAoFBwELW10GVV9aBQwEGA1bXA5SVgsADhgNWwdfDw0eGzbEiMK8PsK2w7vCtTHDiMOwwr5Iwrc%2BMDtIITowwqs4SUk6MTNJO8SxxLrFicSwxLDEscS6w5nDmMO%2Fw5XEusWJxLDCv8SxxLrFicOww5%2FEscS6xYnEsMO%2BwrA5M8KCwr7CscKmxITCsMOvxJrCmkPCp1HDssKrw7XCscKmTMOyXMK6SsKiwqHCnE5LwrrDiUPClGhfUVJFwqp5wqtPwqFZwrhkXGNpQsK6WVZBBmlbV2k8A1kRw44%3D; bst=V2RtkvFeL6311sVtRuyRocLy247DrfzCg~|RtkvFeL6311sVtRuyRocLy247DrQxCk~; __c=1766504278; __a=79240322.1766238834.1766238834.1766504278.53.2.21.53',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    
    }

total_pages = 20

for page in range(1, total_pages + 1):
    print(f"--- 正在爬取第 {page} 页 ---")
    payload = {
        'page': str(page),
        'pageSize': '15',
        'city': '101030100',
        'query': '网络爬虫工程师',
        'expectInfo': '',
        'multiSubway': '',
        'multiBusinessDistrict': '',
        'position': '',
        'jobType': '',
        'salary': '',
        'experience': '',
        'degree': '',
        'industry': '',
        'scale': '',
        'stage': '',
        'scene': '1'
    }


    url = 'https://www.zhipin.com/wapi/zpgeek/search/joblist.json?_=1766549959184'
    #1.访问数据来源
    response = requests.get(url=url,headers=headers,params=payload)
    #2.从其中将整个内容拿到


    json_data = response.json()
    #3.将信息提取出来
    jobList = json_data['zpData']['jobList']

    for job in jobList:
        jobName = job['jobName']
        bossName = job['bossName']
        brandName = job['brandName']
        salaryDesc = job['salaryDesc']
        cityName = job['cityName']
        areaDistrict = job['areaDistrict']
        jobExperience = job['jobExperience']
        jobDegree = job['jobDegree']
        print(jobName,salaryDesc, bossName,brandName,cityName, areaDistrict, jobExperience,jobDegree)
        csv_writer.writerow([jobName,salaryDesc, bossName,brandName,cityName, areaDistrict, jobExperience,jobDegree])







print(response)
print(response.text)


import os
print("当前工作目录:", os.getcwd())















import requests
import csv


f = open('b.csv',mode='a',newline='',encoding='utf-8')
csv_writer = csv.writer(f)

csv_writer.writerow(['jobName','salaryDesc', 'bossName','brandName','cityName', 'areaDistrict', 'jobExperience','jobDegree'])

headers={
    'Cookie':'lastCity=101030100; ab_guid=f15d4127-6c3a-49c0-be9a-2cb0f549f738; wt2=DFGOALq23w-0wEMOPVsjnzAo8HrBjl6MxBNQg1skbrHVT4ynFedk2tYDcy2jodvK0ZqO854IAfD-s1cTAkvp6NQ~~; wbg=0; zp_at=uVevtNEGmGxDPGeJetYlNbRZMNwWwomqP0kAYloGAHs~; __zp_seo_uuid__=30a4f1d1-ff2d-4a34-89cc-23cc22bad3ed; __l=r=https%3A%2F%2Fwww.google.com%2F&l=%2F&s=1; __g=-; Hm_lvt_194df3105ad7148dcf2b98a91b5e727a=1766238834,1766504279; HMACCOUNT=6D5DCEC5E098F33C; Hm_lpvt_194df3105ad7148dcf2b98a91b5e727a=1766545565; __zp_stoken__=cdcfgPjvDj8K9woLCsTYgLzM%2BL0g%2BNz47OjBJPjt5IDs9MDE1SBvDm8KxwpTDrcKIU8OBw6nCtz8uOj4zPjtIPjAzHjoyOz07MT87xLzCsEk%2FO8Odw7s8LhXDp8KwwqjDq3ZWw4ANZ8KwAcOswrEBw6bCvQzDkcK9D8OnwrAgLQMOAwYPGF9bB1NUXgMNAgsBXV0AUVINAQALAV0GUQQBEBpIw7vCsDDCt8O9wr49wrbDscKwO8KzMDE9Oy08McKtKzU3Oz8wNT3EsMS8xLrEvMS%2BxLDEvMOKw4TDscOUxLzEusS8wrHEsMS8xLrDvMORxLDEvMS6xLzDsMKxJzDChsKwQMO5xIPCvMOhxJvCnF%2FDtMK4w7vCpcKSecKsXsKSwrzCoMKiwqJEwoDCiMKgwqvCk27Ck1ZeX1FZwqx4wq1Ewq1HwrliV2dXQ8K8SlJPB1doU1c9BX7DjMOP; bst=V2RtkvFeL6311sVtRuyRocLy247DrRxi8~|RtkvFeL6311sVtRuyRocLy247DrWwSs~; __c=1766504278; __a=79240322.1766238834.1766238834.1766504278.55.2.23.55',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    
    }

total_pages = 20

for page in range(1, total_pages + 1):
    print(f"--- 正在爬取第 {page} 页 ---")
    payload = {
        'page': str(page),
        'pageSize': '15',
        'city': '101030100',
        'query': '网络爬虫工程师',
        'expectInfo': '',
        'multiSubway': '',
        'multiBusinessDistrict': '',
        'position': '',
        'jobType': '',
        'salary': '',
        'experience': '',
        'degree': '',
        'industry': '',
        'scale': '',
        'stage': '',
        'scene': '1'
    }


    url = 'https://www.zhipin.com/wapi/zpgeek/search/joblist.json?_=1766549959184'
    #1.访问数据来源
    response = requests.get(url=url,headers=headers,params=payload)
    #2.从其中将整个内容拿到
    print(response)
    print(response.text)








import requests
import csv
import time  # 1. 导入时间模块

f = open('b.csv', mode='a', newline='', encoding='utf-8')
csv_writer = csv.writer(f)

csv_writer.writerow(['jobName','salaryDesc', 'bossName','brandName','cityName', 'areaDistrict', 'jobExperience','jobDegree'])

headers={
    'Cookie':'lastCity=101030100; ab_guid=f15d4127-6c3a-49c0-be9a-2cb0f549f738; wt2=DFGOALq23w-0wEMOPVsjnzAo8HrBjl6MxBNQg1skbrHVT4ynFedk2tYDcy2jodvK0ZqO854IAfD-s1cTAkvp6NQ~~; wbg=0; zp_at=uVevtNEGmGxDPGeJetYlNbRZMNwWwomqP0kAYloGAHs~; __zp_seo_uuid__=30a4f1d1-ff2d-4a34-89cc-23cc22bad3ed; __l=r=https%3A%2F%2Fwww.google.com%2F&l=%2F&s=1; __g=-; Hm_lvt_194df3105ad7148dcf2b98a91b5e727a=1766238834,1766504279; HMACCOUNT=6D5DCEC5E098F33C; Hm_lpvt_194df3105ad7148dcf2b98a91b5e727a=1766545565; __zp_stoken__=cdcfgMz%2FDjMK%2FwofCtDIrbjYzKzM8MjM%2FMTo0Mz9iKj5IPDo3NRbCqcKxLcOqfVfDisKPwq8yKjE8NjM%2FMzw9PhoxSD5IPzo9PsWJwrwyPT7DqMO%2FPywowqjCvyDDq3Nbw4wOZcK9BMOgwroLw6PDiADDmsK%2FCsOqwrwrLwYDBwUNBVJfBGlpUwcOGA4EUV4KXF8BCgoOBFEFWxkEHBEyw77CtTzCtMO%2FwrtIwrLDusK6PsK%2BPDo%2FPjgwOsKvLjAzMD09MDHEu8S%2BxL%2FFicS6xLvEvsOPw4HDvcOXxL7Ev8WJwr3Eu8S%2BxL%2FEicOdxLvEvsS%2FxYnDvMK6JT3Ci8K8Q3bFsMOJw63EkMKew4nDusKjwp1Kw7LCq8KHwqHCjELCj0JEacKAwq5KYMKiwrjCm1JdXVxMwqBjwq9ZwrhDwqJ4UmpTWMK%2BT19LBFVVXlM%2BB8KBwoXDjw%3D%3D; bst=V2RtkvFeL6311sVtRuyRocLy247DrSwik~|RtkvFeL6311sVtRuyRocLy247DrSxig~; __c=1766504278; __a=79240322.1766238834.1766238834.1766504278.61.2.29.61',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }

total_pages = 20

for page in range(1, total_pages + 1):
    print(f"--- 正在爬取第 {page} 页 ---")
    payload = {
        'page': str(page),
        'pageSize': '15',
        'city': '101030100',
        'query': '网络爬虫工程师',
        'expectInfo': '',
        'multiSubway': '',
        'multiBusinessDistrict': '',
        'position': '',
        'jobType': '',
        'salary': '',
        'experience': '',
        'degree': '',
        'industry': '',
        'scale': '',
        'stage': '',
        'scene': '1'
    }
    
    current_timestamp = int(time.time() * 1000)

    url = f'https://www.zhipin.com/wapi/zpgeek/search/joblist.json?_={current_timestamp}'
    
    # 访问数据来源
    response = requests.get(url=url, headers=headers, params=payload)
    
    # 打印状态和结果
    print(response)
    print(response.text)

    # 在此处可以将你的 json 提取和 csv 写入逻辑补上...
    # json_data = response.json()
    # ... (省略中间的数据提取代码) ...

    # 2. 爬取完一页后歇息 5 秒
    print(f"第 {page} 页爬取完成，正在休眠 5 秒...")
    time.sleep(5)

f.close()
















import requests


url = 'https://www.zhipin.com/wapi/zpgeek/search/joblist.json?_=1766545667585'

headers = {
    'Cookie':'lastCity=101030100; ab_guid=f15d4127-6c3a-49c0-be9a-2cb0f549f738; wt2=DFGOALq23w-0wEMOPVsjnzAo8HrBjl6MxBNQg1skbrHVT4ynFedk2tYDcy2jodvK0ZqO854IAfD-s1cTAkvp6NQ~~; wbg=0; zp_at=uVevtNEGmGxDPGeJetYlNbRZMNwWwomqP0kAYloGAHs~; __zp_seo_uuid__=30a4f1d1-ff2d-4a34-89cc-23cc22bad3ed; __l=r=https%3A%2F%2Fwww.google.com%2F&l=%2F&s=1; __g=-; Hm_lvt_194df3105ad7148dcf2b98a91b5e727a=1766238834,1766504279; HMACCOUNT=6D5DCEC5E098F33C; Hm_lpvt_194df3105ad7148dcf2b98a91b5e727a=1766545565; __zp_stoken__=cdcfgPTPDgMK%2FwoPCuj4vHDI9Jzc8Nj0zPTpIPTNmKjo%2BSD43SRDChMK%2BIMOlc1vDjsOswqs8Jj08Mj0zNzwxSBY9SDo%2BMz49OsS%2Fw4g2PTrDnsOzOywUwpTDiCHDuHdVw5gKZcKxCsOkwr4Lw6fCvgTDnsK%2FDsOkw4gvLwINCxkNGVxTGGlVXQsKGAoKVVoKUGkFDgoKClUZWwUKKB0yw7rCu0jDiMO%2Fwr8%2Bwr7DvsK6OsOISD4%2FOi40PsKvKjY%2FPD0xNjXEv8S%2BxLvEv8S2xL%2FEvsOLw4fEicOTxL7Eu8S%2Fw4nEv8S%2BxLvDv8OpxL%2FEvsS7xL%2FEiMK%2BJTHChcOIwrR9xbvCv8O5xJzCnlLCmMKgwrPCpsKHTcKtwrPCokLCu03CimBZVsK6wqHCgWvClV5RXVBCwqRnwq9Fwq5PwqZ4VmRfRMK%2BS2lHGFVpaF86B0nCj8OP; bst=V2RtkvFeL6311sVtRuyRocLy247DrSxig~|RtkvFeL6311sVtRuyRocLy247DrfzC0~; __c=1766504278; __a=79240322.1766238834.1766238834.1766504278.62.2.30.62',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }


payload = {
    'page': '20',
    'pageSize': '15',
    'city': '101030100',
    'query': '网络爬虫工程师',
    'expectInfo': '',
    'multiSubway': '',
    'multiBusinessDistrict': '',
    'position': '',
    'jobType': '',
    'salary': '',
    'experience': '',
    'degree': '',
    'industry': '',
    'scale': '',
    'stage': '',
    'scene': '1'
}

response = requests.get(url=url,headers=headers,params=payload)
html_data = response.text
print(html_data)


#数据解析（提取我们想要的数据）css选择器提取数据（专门用于提取HTML数据）
selector = parsel.Selector(html_data)
selector.css('')








import requests
import csv
import time
import random  # 引入随机模块

f = open('b.csv', mode='a', newline='', encoding='utf-8')
csv_writer = csv.writer(f)

# 保持原来的表头
csv_writer.writerow(['jobName','salaryDesc', 'bossName','brandName','cityName', 'areaDistrict', 'jobExperience','jobDegree'])

headers={
    'Cookie':'lastCity=101030100; ab_guid=f15d4127-6c3a-49c0-be9a-2cb0f549f738; wt2=DFGOALq23w-0wEMOPVsjnzAo8HrBjl6MxBNQg1skbrHVT4ynFedk2tYDcy2jodvK0ZqO854IAfD-s1cTAkvp6NQ~~; wbg=0; zp_at=uVevtNEGmGxDPGeJetYlNbRZMNwWwomqP0kAYloGAHs~; __zp_seo_uuid__=30a4f1d1-ff2d-4a34-89cc-23cc22bad3ed; __l=r=https%3A%2F%2Fwww.google.com%2F&l=%2F&s=1; __g=-; Hm_lvt_194df3105ad7148dcf2b98a91b5e727a=1766238834,1766504279; HMACCOUNT=6D5DCEC5E098F33C; Hm_lpvt_194df3105ad7148dcf2b98a91b5e727a=1766545565; __zp_stoken__=cdcfgPDvDjcKxwoPCuzYqBUk8LzI6Njw7MDxIPDtjLDo%2FMDtJSRHDrcK7wpTEvXJTw4vDssKvPS4wOjI8OzI6MUkeMDY6Pzs7OzrEvsKwMzs6w5%2FDuz4qFMO1wrDClsS%2Bd1TDgA9jwrELw6zCuw3Dp8K%2FDMObwrEOw6XCsCohAgwDBAsZXVsFV1VcAw8GCgtdXwxQaA0LDAoLXQRdBQsQEDTDusK6MMK1w7HCvz%2FCtsO7wrw6w4kwOzE6Lzw7wqEqNzcxOzE3PcS6xLDEu8S%2BxL7EusSww4vDhsOxw5bEsMS7xL7CscS6xLDEu8O%2Bw5HEusSwxLvEvsOwwrsjMcKEwrDCq8OQxZPCvsOhxJHCkFnClcKfw7tdwpjCvsK%2FVsO%2FwqzCuk7CgsKqQE%2FCu0DCqlDCnVZcW1BDwqxiwqFFwq9HwqNmVmVXWcKwS2hPBVNpaVc%2FGcKDQcOP; bst=V2RtkvFeL6311sVtRuyRocLy247DrfzC0~|RtkvFeL6311sVtRuyRocLy247DrRwiw~; __c=1766504278; __a=79240322.1766238834.1766238834.1766504278.63.2.31.63',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }

total_pages = 20

# --- 改进1：生成随机页码列表 ---
page_list = list(range(1, total_pages + 1))
random.shuffle(page_list) 
print(f"爬取顺序已打乱: {page_list}")

for page in page_list:
    # 动态生成当前时间戳
    current_ts = int(time.time() * 1000)
    url = f'https://www.zhipin.com/wapi/zpgeek/search/joblist.json?_={current_ts}'
    
    print(f"--- 正在随机爬取第 {page} 页 ---")
    
    payload = {
        'page': str(page),
        'pageSize': '15',
        'city': '101030100',
        'query': '网络爬虫工程师',
        'expectInfo': '',
        'multiSubway': '',
        'multiBusinessDistrict': '',
        'position': '',
        'jobType': '',
        'salary': '',
        'experience': '',
        'degree': '',
        'industry': '',
        'scale': '',
        'stage': '',
        'scene': '1'
    }

    # 1.访问数据来源
    response = requests.get(url=url, headers=headers, params=payload)
    
    # 2.检查返回结果
    print(f"状态码: {response.status_code}")
    print(response.text)
    
    # 如果触发了行为异常，建议直接停止
    if '"code":37' in response.text:
        print("检测到行为异常(code 37)，随机化也救不了当前的 Cookie，请手动去浏览器验证！")
        break

    # --- 改进2：使用随机休眠 (10-20秒之间) ---
    sleep_time = random.uniform(10, 20)
    print(f"第 {page} 页请求完成，模拟人类行为随机休眠 {sleep_time:.2f} 秒...")
    time.sleep(sleep_time)

f.close()






from DrissionPage import ChromiumPage
import csv
import time
import random

# --- 1. 初始化浏览器和文件 ---
# 这会打开一个浏览器窗口，你可以看到它在操作
page = ChromiumPage()
f = open('b.csv', mode='a', newline='', encoding='utf-8')
csv_writer = csv.writer(f)

# 保持你要求的表头结构不变
# csv_writer.writerow(['jobName','salaryDesc', 'bossName','brandName','cityName', 'areaDistrict', 'jobExperience','jobDegree'])

# --- 2. 设置随机页码 ---
total_pages = 20
page_list = list(range(1, total_pages + 1))
random.shuffle(page_list)
print(f"随机爬取顺序: {page_list}")

base_url = "https://www.zhipin.com/web/geek/job?query=%E7%BD%91%E7%BB%9C%E7%88%AC%E8%99%AB%E5%B7%A5%E7%A8%8B%E5%B8%88&city=101030100"

for p in page_list:
    print(f"--- 🚀 正在爬取第 {p} 页 ---")
    
    # 构造带页码的URL
    target_url = f"{base_url}&page={p}"
    page.get(target_url)
    
    # --- 3. 智能等待与反爬处理 ---
    # 等待页面核心元素加载出来
    if page.ele('text:验证码'):
        print("发现验证码！请在浏览器窗口手动完成验证，程序会自动检测通过...")
        page.wait.ele_deletion('text:验证码', timeout=60) # 等待验证码消失
    
    # 稍微等一下列表加载
    time.sleep(random.uniform(2, 4))
    
    # --- 4. 提取数据 (使用 XPath 匹配你需要的字段) ---
    items = page.eles('.job-card-wrapper') # 找到所有的职位卡片
    
    if not items:
        print(f"第 {p} 页未发现数据，可能是被拦截或Cookie失效，建议手动在浏览器点一下。")
        continue

    for item in items:
        try:
            # 提取各个字段
            jobName = item.ele('.job-name').text
            salaryDesc = item.ele('.salary').text
            bossName = item.ele('.boss-info-tag').text # 对应你的 bossName
            brandName = item.ele('.company-name').text
            
            # 城市和地区通常在 .job-area-wrapper 里
            area_text = item.ele('.job-area').text # 例如 "天津·南开区"
            cityName = area_text.split('·')[0] if '·' in area_text else area_text
            areaDistrict = area_text.split('·')[1] if '·' in area_text else ""
            
            # 经验和学历通常在 .job-labels 列表里
            labels = item.eles('.tag-list')
            # 标签通常是 [经验, 学历]
            jobExperience = labels[0].text if len(labels) > 0 else "不限"
            jobDegree = labels[1].text if len(labels) > 1 else "不限"

            # 5. 写入 CSV (保持原来的列顺序)
            csv_writer.writerow([jobName, salaryDesc, bossName, brandName, cityName, areaDistrict, jobExperience, jobDegree])
            
        except Exception as e:
            print(f"某条数据解析出错，跳过: {e}")

    print(f"✅ 第 {p} 页爬取完成，抓取了 {len(items)} 条数据。")

    # --- 6. 随机休眠 (模拟人类看网页的时间) ---
    sleep_time = random.uniform(10, 20)
    print(f"等待 {sleep_time:.2f} 秒后继续...\n")
    time.sleep(sleep_time)

f.close()
page.quit()
print("🎉 数据爬取任务圆满完成！")





from DrissionPage import ChromiumPage
import csv
import time
import random

# 1. 初始化
page = ChromiumPage()
f = open('b.csv', mode='a', newline='', encoding='utf-8')
csv_writer = csv.writer(f)

# 初始访问第一页
base_url = "https://www.zhipin.com/web/geek/job?query=%E7%BD%91%E7%BB%9C%E7%88%AC%E8%99%AB%E5%B7%A5%E7%A8%8B%E5%B8%88&city=101030100"
page.get(base_url)

for p in range(1, 21): # 建议按顺序爬
    print(f"--- 🚀 正在处理第 {p} 页 ---")
    
    # --- 智能验证检测 ---
    if "安全校验" in page.title or page.ele('text:验证码'):
        print("🚨 发现验证码！请在浏览器手动完成验证，完成后程序会自动继续...")
        # 循环等待，直到验证码消失或页面标题改变
        while page.ele('text:验证码'):
            time.sleep(2)
    
    # 等待列表加载完成
    page.wait.ele_displayed('.job-card-wrapper', timeout=10)
    
    # --- 提取数据 ---
    items = page.eles('.job-card-wrapper')
    
    if not items:
        print(f"第 {p} 页未发现数据，请检查浏览器是否正常显示职位列表。")
        # 如果卡住了，建议手动在浏览器里点一下刷新
        input("请确认页面加载正常后，按回车键继续...")
        items = page.eles('.job-card-wrapper')

    for item in items:
        try:
            # 这里的 class 名 BOSS 可能会变，如果爬不到，请反馈
            jobName = item.ele('.job-name').text
            salaryDesc = item.ele('.salary').text
            brandName = item.ele('.company-name').text
            area_text = item.ele('.job-area').text
            
            # 写入 CSV
            csv_writer.writerow([jobName, salaryDesc, brandName, area_text])
        except:
            continue

    print(f"✅ 第 {p} 页抓取成功 ({len(items)} 条)")

    # --- 模拟真人点击“下一页” ---
    if p < 20:
        btn_next = page.ele('@class=ui-icon-arrow-right') # 找到下一页图标按钮
        if btn_next:
            sleep_time = random.uniform(8, 15)
            print(f"休眠 {sleep_time:.2f} 秒后点击下一页...")
            time.sleep(sleep_time)
            btn_next.click()
        else:
            print("找不到下一页按钮，可能已到底或被拦截。")
            break

f.close()
print("任务结束")





from DrissionPage import ChromiumPage
import csv
import time
import random

# 1. 初始化
page = ChromiumPage()
f = open('b.csv', mode='a', newline='', encoding='utf-8')
csv_writer = csv.writer(f)

# 访问第一页
url = "https://www.zhipin.com/web/geek/job?query=%E7%BD%91%E7%BB%9C%E7%88%AC%E8%99%AB%E5%B7%A5%E7%A8%8B%E5%B8%88&city=101030100"
page.get(url)

for p in range(1, 21):
    print(f"--- 🚀 正在处理第 {p} 页 ---")
    
    # 强制等待 3 秒，让 JS 渲染完成
    time.sleep(3)
    
    # 智能检查：如果页面没加载出职位列表，先处理验证码或刷新
    if not page.ele('.job-list-box') and not page.ele('.job-card-wrapper'):
        print("🚨 页面空空如也！可能是触发了验证码，或者需要手动刷新。")
        input("请在浏览器处理好页面（看到职位列表）后，按回车键继续...")

    # --- 提取职位卡片 (使用模糊匹配类名) ---
    # 尝试多种可能的容器类名
    items = page.eles('tag:li@@class^job-card') # 匹配所有 class 以 job-card 开头的 li 标签
    if not items:
        items = page.eles('.job-card-wrapper') # 备选方案

    count = 0
    for item in items:
        try:
            # 内部查找也使用相对路径和模糊匹配，防止 class 变化
            jobName = item.ele('.job-name').text
            salaryDesc = item.ele('.salary').text
            brandName = item.ele('.company-name').text
            # 找到包含“·”的那个元素，通常是地区
            area_text = item.ele('text:·').text if item.ele('text:·') else "未知"
            
            # 写入 CSV
            csv_writer.writerow([jobName, salaryDesc, brandName, area_text])
            count += 1
        except:
            continue

    print(f"✅ 第 {p} 页抓取成功 ({count} 条)")

    # --- 模拟真人查找“下一页” ---
    if p < 20:
        # 寻找带有“右箭头”图标或者标题为“下一页”的按钮
        btn_next = page.ele('tag:i@@class^ui-icon-arrow-right') or page.ele('@@title=下一页') or page.ele('.options-pages').eles('tag:a')[-1]
        
        if btn_next:
            sleep_time = random.uniform(5, 10)
            print(f"休眠 {sleep_time:.2f} 秒后点击下一页...")
            time.sleep(sleep_time)
            # 滚动到按钮位置再点击，更像真人
            page.scroll.to_see(btn_next)
            btn_next.click()
        else:
            print("找不到下一页按钮，尝试直接修改 URL 进入下一页...")
            page.get(f"{url}&page={p+1}")
            time.sleep(3)

f.close()
print("任务结束")






import requests
import csv
import time
import random
from bs4 import BeautifulSoup

# --- 1. 初始化设置 ---
f = open('boss_jobs.csv', mode='a', newline='', encoding='utf-8')
csv_writer = csv.writer(f)
# 写入完整的表头
csv_writer.writerow(['职位名称', '薪资', '公司', '地区', '发布日期', '职位标签', '职位描述', '详情链接'])

headers = {
    'Cookie': 'lastCity=101030100; ab_guid=f15d4127-6c3a-49c0-be9a-2cb0f549f738; wt2=DFGOALq23w-0wEMOPVsjnzAo8HrBjl6MxBNQg1skbrHVT4ynFedk2tYDcy2jodvK0ZqO854IAfD-s1cTAkvp6NQ~~; wbg=0; zp_at=uVevtNEGmGxDPGeJetYlNbRZMNwWwomqP0kAYloGAHs~; __zp_seo_uuid__=30a4f1d1-ff2d-4a34-89cc-23cc22bad3ed; __l=r=https%3A%2F%2Fwww.google.com%2F&l=%2F&s=1; __g=-; Hm_lvt_194df3105ad7148dcf2b98a91b5e727a=1766238834,1766504279; HMACCOUNT=6D5DCEC5E098F33C; Hm_lpvt_194df3105ad7148dcf2b98a91b5e727a=1766593446; __zp_stoken__=c929gPDnDk8K8wo7CvjstGUJGMz45OUc8OUA%2BPDllMUc6PTw%2FQSPCm8K9wpwweFTDij3DgD0kRkc%2FOUY1PDk7Ijw9PTw8Okc8xYbCujVGR8OaxIZBLBzCpcOGwpkveV%2FDlw49w4cTwo7CvQjDlMK5EcKgw4cOfMOHNioFFwQNFw9dWQtaWFkOEAgKEWFjD2dbEgoQDBNfCmAIDh0nOsO6w4A8wrnDusOAPMK5w7rDgDzCuTo9PEcqQTzCtyo1Oz04RjRCxLvEvMS9xYbEtMS8xL3DlsOTw7zDmcWGxLvEvMK9xYbEu8S8w73DpsS7xLzEvcWGw7rCvS48wpHCvcKLw6XCtcK9w63EncKbw4LCt2TClcKnw7vDgMSHScKQwrfCu3HChUnCjFPCpWbCn3JTbcKiwrTDhnpfaMK9ZcKwYWFJY8ODaG5%2FX8KAXgzCggQKXUEEw57CiA%3D%3D; bst=V2RtkvFeL6311sVtRuyRocLy247DrXxCg~|RtkvFeL6311sVtRuyRocLy247DrVzCk~; __c=1766504278; __a=79240322.1766238834.1766238834.1766504278.65.2.33.65', 
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
    'Referer': 'https://www.zhipin.com/web/geek/job'
}

# --- 2. 爬取前三页列表 ---
page_list = [1, 2, 3] # 顺序爬取更稳，不要跳
all_job_links = []

for page in page_list:
    print(f"--- 正在爬取第 {page} 页列表 ---")
    current_ts = int(time.time() * 1000)
    url = f'https://www.zhipin.com/wapi/zpgeek/search/joblist.json?_={current_ts}'
    
    payload = {
        'page': str(page),
        'pageSize': '15',
        'city': '101030100',
        'query': '网络爬虫工程师',
        'scene': '1'
    }

    try:
        res = requests.get(url=url, headers=headers, params=payload)
        data = res.json()
        
        if data.get('code') == 0:
            job_list = data['zpData']['jobList']
            for job in job_list:
                # 记录列表页能拿到的基础信息
                base_info = {
                    'jobName': job.get('jobName'),
                    'salary': job.get('salaryDesc'),
                    'brand': job.get('brandName'),
                    'city': job.get('cityName'),
                    # 拼接详情页URL
                    'detail_url': f"https://www.zhipin.com/job_detail/{job.get('encryptJobId')}.html"
                }
                all_job_links.append(base_info)
            print(f"第 {page} 页获取成功，暂存 {len(job_list)} 个职位。")
        else:
            print(f"列表页被拦截: {data.get('message')}")
            break
            
    except Exception as e:
        print(f"请求出错: {e}")

    time.sleep(random.uniform(5, 10)) # 列表页翻页间隔

# --- 3. 逐个请求详情页 (深度爬取) ---
print(f"\n--- 列表抓取完成，准备深入爬取 {len(all_job_links)} 个职位的详细描述 ---")

for job_info in all_job_links:
    url = job_info['detail_url']
    print(f"正在获取: {job_info['jobName']} - {job_info['brand']}")
    
    try:
        # 详情页请求需增加 Referer 模拟
        headers['Referer'] = 'https://www.zhipin.com/web/geek/job'
        resp = requests.get(url, headers=headers)
        
        if "安全校验" in resp.text:
            print("⚠️ 详情页触发验证！请去浏览器手动点开一个职位滑动验证码后再继续。")
            break
            
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 解析：职位描述 (剔除干扰项)
        desc_element = soup.find('p', class_='desc')
        if desc_element:
            for hidden in desc_element.find_all('span'): hidden.decompose()
            full_desc = desc_element.get_text(separator=" ").strip().replace('\n', ' ')
        else:
            full_desc = "未抓取到描述"

        # 解析：职位标签
        tag_list = soup.find('ul', class_='job-label-list')
        job_tags = ",".join([li.text for li in tag_list.find_all('li')]) if tag_list else ""

        # 解析：发布日期
        active_span = soup.find('span', class_='boss-active-time')
        pub_date = active_span.text if active_span else "未知"

        # 写入 CSV
        csv_writer.writerow([
            job_info['jobName'], job_info['salary'], job_info['brand'], 
            job_info['city'], pub_date, job_tags, full_desc, url
        ])
        
        print(f"✅ 抓取成功！")

    except Exception as e:
        print(f"解析详情页失败: {e}")

    # 重点：详情页的间隔必须长！！
    wait_time = random.uniform(20, 40)
    print(f"等待 {wait_time:.2f} 秒以保护 Cookie...")
    time.sleep(wait_time)

f.close()
print("🎉 任务全部完成！数据已保存至 boss_jobs.csv")






import requests
import re
import time
import random

# --- 1. 配置信息 ---
headers = {
    'Cookie': 'lastCity=101030100; ab_guid=f15d4127-6c3a-49c0-be9a-2cb0f549f738; wt2=DFGOALq23w-0wEMOPVsjnzAo8HrBjl6MxBNQg1skbrHVT4ynFedk2tYDcy2jodvK0ZqO854IAfD-s1cTAkvp6NQ~~; wbg=0; zp_at=uVevtNEGmGxDPGeJetYlNbRZMNwWwomqP0kAYloGAHs~; __zp_seo_uuid__=30a4f1d1-ff2d-4a34-89cc-23cc22bad3ed; __l=r=https%3A%2F%2Fwww.google.com%2F&l=%2F&s=1; __g=-; Hm_lvt_194df3105ad7148dcf2b98a91b5e727a=1766238834,1766504279; HMACCOUNT=6D5DCEC5E098F33C; Hm_lpvt_194df3105ad7148dcf2b98a91b5e727a=1766593446; __zp_stoken__=c929gQznDlsOHwpLCvDwxekBDJTk8R0M5Q0JBQzloMjtAOkA%2FQCYPwr3Cny50YcOQbsK%2BQiRDPENDOTk8OD4UQzg7QDlAPTvFgcK6OD07w6DDuT0sHRjCv8KYMsKHW8OKDEfCuAzCjsOAC8Ogw4MOwpzDhw%2FCgcK5MTcTExEPDRBiWQ5ZXGMRDAgLDF9cEllfDxASCwxfD2MMDBojOsO7wr06wr7Eh8K%2BQMK8xIDDgjvCvjpARzswPkDCtys4PUI9ODg%2FxYHFhsS6xYHEtMWBxYbDisOJw7vDlcWGxLrFgcK7xYHFhsS6xIHDm8WBxYbEusWBw7rDgCU4wovCusKPw6DCtMOAw6vEosKmwrzDv8Ksw7VQwqLCnMKNUcKERMKJbMKwwrROwrPCpMK5SmRWaMKYwr7CuXVfZcOGccKqXl1JYsOGdmnCgmF8Yw7CgBMFXTwPasK7w4w%3D; bst=V2RtkvFeL6311sVtRuyRocLy247DrVzCk~|RtkvFeL6311sVtRuyRocLy247DrUwSo~; __c=1766504278; __a=79240322.1766238834.1766238834.1766504278.66.2.34.66',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.zhipin.com/web/geek/job'
}

def get_detail(url):
    print(f"正在抓取详情: {url}")
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        html = resp.text
        
        if "职位描述" not in html:
            print("⚠️ 未能进入详情页，可能需要更新Cookie或处理验证码")
            return None

        # --- 2. 正则解析逻辑 ---
        
        # (1) 提取职位描述：匹配 <p class="desc">...</p> 之间的内容
        # 使用 [\s\S]*? 是为了匹配包含换行符在内的所有字符
        desc_match = re.search(r'<p data-v-.*? class="desc">([\s\S]*?)</p>', html)
        if desc_match:
            raw_desc = desc_match.group(1)
            # 剔除 HTML 标签（如 <span>, <style> 等）
            clean_desc = re.sub(r'<.*?>', '', raw_desc)
            # 剔除常见的混淆词
            clean_desc = re.sub(r'直聘|boss|来自BOSS|BOSS直聘', '', clean_desc, flags=re.I)
            # 压缩多余空格和换行
            clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
        else:
            clean_desc = "未找到描述"

        # (2) 提取职位标签：匹配 <ul class="job-label-list"> 里的 <li>
        tags_section = re.search(r'<ul data-v-.*? class="job-label-list">([\s\S]*?)</ul>', html)
        if tags_section:
            job_tags = re.findall(r'<li.*?>(.*?)</li>', tags_section.group(1))
            tags_str = ",".join(job_tags)
        else:
            tags_str = "无标签"

        # (3) 提取活跃日期：匹配 <span class="boss-active-time">
        date_match = re.search(r'<span data-v-.*? class="boss-active-time">(.*?)</span>', html)
        pub_date = date_match.group(1) if date_match else "未知"

        return clean_desc, tags_str, pub_date

    except Exception as e:
        print(f"抓取异常: {e}")
        return None

# --- 3. 主程序逻辑 ---
def start():
    # 1. 获取第一页列表
    list_url = 'https://www.zhipin.com/wapi/zpgeek/search/joblist.json'
    params = {'query': '网络爬虫工程师', 'city': '101030100', 'page': '1', 'pageSize': '15'}
    
    res = requests.get(list_url, headers=headers, params=params)
    if res.json().get('code') != 0:
        print("列表页抓取失败，请检查Cookie")
        return

    job_list = res.json()['zpData']['jobList']
    
    for job in job_list:
        e_id = job['encryptJobId']
        detail_url = f"https://www.zhipin.com/job_detail/{e_id}.html"
        
        # 抓取并解析
        result = get_detail(detail_url)
        
        if result:
            desc, tags, date = result
            print(f"✅ 职位: {job['jobName']}")
            print(f"   日期: {date}")
            print(f"   标签: {tags}")
            print(f"   描述: {desc[:50]}...") # 打印前50个字
            print("-" * 30)
        
        # 详情页之间必须有长间隔
        wait = random.uniform(15, 25)
        print(f"等待 {wait:.2f} 秒...")
        time.sleep(wait)

if __name__ == "__main__":
    start()








































