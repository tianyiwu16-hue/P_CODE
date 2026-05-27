#drissionpage模板

from DrissionPage import ChromiumOptions

path=r'C:\Program Files\Google\Chrome\Application\chrome.exe'
ChromiumOptions().set_browser_path(path).save()




import time  #  导入时间模块
#导入自动化模块
from DrissionPage import ChromiumPage
#打开浏览器
dp = ChromiumPage()

#监听（一定要在访问之前）
dp.listen.start('joblist')


#访问网站

current_timestamp = int(time.time() * 1000)

dp.get('https://www.zhipin.com/web/geek/jobs?query=网络爬虫工程师&city=101030100')

#等待数据包的加载
resp = dp.listen.wait()
#获取响应数据
json_data = resp.response.body
jobList = json_data['zpData']['jobList']
for job in jobList:
    print(job)






import time  # 导入时间模块
# 导入自动化模块
from DrissionPage import ChromiumPage

# 打开浏览器
dp = ChromiumPage()

# === 关键修复：监听正确的API路径（用包含关键词的方式，更精准）===
dp.listen.start('wapi/zpgeek/search/joblist.json')  # 推荐这样写，精确匹配职位API

# 访问网站（指定城市：101030100 是沈阳，你可以改成其他）
dp.get('https://www.zhipin.com/web/geek/jobs?query=网络爬虫工程师&city=101030100')

# 等待页面基本加载
time.sleep(5)

# 多次滚动到底部，触发加载更多职位数据（BOSS直聘是无限滚动）
for _ in range(20):  
    dp.scroll.to_bottom()
    print("滚动到底部，触发加载...")
    time.sleep(3)  # 等待新数据包加载

    # 等待数据包响应
    resp = dp.listen.wait(timeout=10)  # 最多等10秒

    if resp and resp.response.body:  # 安全检查，避免None报错
        json_data = resp.response.body  # 已经是字典
        jobList = json_data.get('zpData', {}).get('jobList', [])
        print(f"--- 正在爬取第 {_} 页 ---")
        print(f"本页抓取到 {len(jobList)} 条职位")
        
        for job in jobList:
            print({
                '职位名称': job.get('jobName'),
                '薪资': job.get('salaryDesc'),
                '公司': job.get('brandName'),
                '地点': job.get('cityName') + '-' + job.get('areaDistrict'),
                '经验': job.get('experienceDesc'),
                '学历': job.get('degreeDesc'),
                '福利': job.get('welfareDesc', []),
                '技能标签': job.get('skills', [])
            })
            print("-" * 80)
    else:
        print("未捕获到数据包，可能已加载完毕或网络问题")
        break

# 停止监听并关闭浏览器
dp.listen.stop()
dp.quit()










for _ in range(20):  
    dp.scroll.to_bottom()
    print("滚动到底部，触发加载...")
    time.sleep(3)  












import time
import pandas as pd
from DrissionPage import ChromiumPage

# 打开浏览器（会复用你之前保存的 Chrome 配置）
dp = ChromiumPage()

# 开始监听职位列表 API（精确匹配，成功率最高）
dp.listen.start('wapi/zpgeek/search/joblist.json')

# 访问搜索页面（示例：网络爬虫工程师，沈阳。可自行修改 query 和 city）
dp.get('https://www.zhipin.com/web/geek/jobs?query=网络爬虫工程师&city=101030100')

# 初始等待页面加载
time.sleep(5)

# 存储所有职位完整信息
all_jobs_detail = []

# 抓取前 N 页（建议先设小一点测试，比如 3~5 页）
for page_num in range(1, 6):
    print(f"\n=== 正在抓取第 {page_num} 页职位列表 ===")
    
    # 滚动到底部触发加载更多
    dp.scroll.to_bottom()
    time.sleep(3)
    
    # 等待数据包
    resp = dp.listen.wait(timeout=10)
    
    if not resp or not resp.response.body:
        print("本页无更多数据，抓取结束")
        break
    
    json_data = resp.response.body
    jobList = json_data.get('zpData', {}).get('jobList', [])
    
    print(f"第 {page_num} 页抓到 {len(jobList)} 条职位，正在逐个抓取详情...")
    
    for job in jobList:
        # 基础信息
        basic_info = {
            '职位名称': job.get('jobName'),
            '薪资': job.get('salaryDesc'),
            '公司': job.get('brandName'),
            '地点': f"{job.get('cityName', '')}-{job.get('areaDistrict', '')}",
            '经验': job.get('experienceDesc'),
            '学历': job.get('degreeDesc'),
        }
        
        # 关键：获取加密职位 ID
        encryptJobId = job.get('encryptJobId')
        if not encryptJobId:
            print("缺少 encryptJobId，跳过此职位")
            continue
        
        # 构造详情页 URL（带 lid 和 securityId 更稳定）
        lid = job.get('lid', '')
        securityId = job.get('securityId', '').lower()  # 有时是大写，需要转小写
        
        detail_url = f"https://www.zhipin.com/job_detail/{encryptJobId}.html"
        if lid and securityId:
            detail_url += f"?lid={lid}&securityId={securityId}"
        
        # 新标签页打开详情页
        detail_tab = dp.new_tab(detail_url)
        detail_tab.wait.load_start()
        time.sleep(3)  # 等待 JS 渲染完成
        
        try:
            # 提取详细信息（2025年底有效的选择器）
            job_desc = detail_tab.ele('css:.detail-bottom-text', timeout=8).text.strip()
            company_desc = detail_tab.ele('css:.company-info-box', timeout=5).text.strip() if detail_tab.ele('css:.company-info-box', timeout=1) else ''
            address = detail_tab.ele('css:.job-location .address', timeout=5).text.strip() if detail_tab.ele('css:.job-location .address', timeout=1) else ''
            hr_name = detail_tab.ele('css:.boss-info .name', timeout=5).text.strip() if detail_tab.ele('css:.boss-info .name', timeout=1) else ''
            
            # 合并完整信息
            full_info = {
                **basic_info,
                '职位描述': job_desc,
                '公司介绍': company_desc,
                '详细地址': address,
                'HR姓名': hr_name,
                '详情链接': detail_url
            }
            
            all_jobs_detail.append(full_info)
            print(f"✓ 成功: {basic_info['职位名称']} @ {basic_info['公司']}")
            
        except Exception as e:
            print(f"✗ 详情解析失败（可能风控或页面异常）: {e}")
        
        # 关闭详情标签页
        detail_tab.close()
        
        # 防风控：每个职位间隔 2~4 秒
        time.sleep(3)

# 停止监听，关闭浏览器
dp.listen.stop()
dp.quit()

# 输出结果
print(f"\n抓取完成！共获取 {len(all_jobs_detail)} 个职位的完整详情\n")
for info in all_jobs_detail:
    print(info)
    print("=" * 100)

# 可选：保存到 CSV 文件（方便查看和分析）
if all_jobs_detail:
    df = pd.DataFrame(all_jobs_detail)
    filename = '网络爬虫工程师_沈阳_完整详情.csv'
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"数据已保存到 {filename}")







import time
import pandas as pd
from DrissionPage import ChromiumPage

dp = ChromiumPage()

dp.listen.start('wapi/zpgeek/search/joblist.json')

dp.get('https://www.zhipin.com/web/geek/jobs?query=网络爬虫工程师&city=101030100')

time.sleep(5)

all_jobs_detail = []

for page_num in range(1, 2):
    print(f"\n=== 正在抓取第 {page_num} 页职位列表 ===")
    
    dp.scroll.to_bottom()
    time.sleep(3)
    
    resp = dp.listen.wait(timeout=10)
    
    if not resp or not resp.response.body:
        print("本页无更多数据，抓取结束")
        break
    
    json_data = resp.response.body
    jobList = json_data.get('zpData', {}).get('jobList', [])
    
    print(f"第 {page_num} 页抓到 {len(jobList)} 条职位，正在逐个抓取详情...")
    
    for job in jobList:
        basic_info = {
            '职位名称': job.get('jobName'),
            '薪资': job.get('salaryDesc'),
            '公司': job.get('brandName'),
            '地点': f"{job.get('cityName', '')}-{job.get('areaDistrict', '')}",
            '经验': job.get('experienceDesc'),
            '学历': job.get('degreeDesc'),
        }
        
        encryptJobId = job.get('encryptJobId')
        if not encryptJobId:
            print("缺少 encryptJobId，跳过此职位")
            continue
        
        lid = job.get('lid', '')
        securityId = job.get('securityId', '').lower()
        
        detail_url = f"https://www.zhipin.com/job_detail/{encryptJobId}.html"
        if lid and securityId:
            detail_url += f"?lid={lid}&securityId={securityId}"
        
        detail_tab = dp.new_tab(detail_url)
        detail_tab.wait.load_start()
        time.sleep(4)  # 多等一秒，确保渲染完成
        
        try:
            # 关键修复：职位描述的正确选择器
            job_desc = detail_tab.ele('css:.job-sec-text', timeout=10).text.strip()
            
            # 其他字段（更宽松的容错）
            company_desc = detail_tab.ele('css:.company-info-box', timeout=5).text.strip() if detail_tab.ele('css:.company-info-box', timeout=1) else '未找到公司介绍'
            address = detail_tab.ele('css:.job-location .address', timeout=5).text.strip() if detail_tab.ele('css:.job-location .address', timeout=1) else '未找到地址'
            hr_name = detail_tab.ele('css:.boss-info .name', timeout=5).text.strip() if detail_tab.ele('css:.boss-info .name', timeout=1) else '未找到HR'
            
            full_info = {
                **basic_info,
                '职位描述': job_desc,
                '公司介绍': company_desc,
                '详细地址': address,
                'HR姓名': hr_name,
                '详情链接': detail_url
            }
            
            all_jobs_detail.append(full_info)
            print(f"✓ 成功: {basic_info['职位名称']} @ {basic_info['公司']}")
            
        except Exception as e:
            print(f"✗ 详情解析失败: {e}")
            # 可选：打印页面源码调试
            # print(detail_tab.html)
        
        detail_tab.close()
        time.sleep(3)  # 防风控

dp.listen.stop()
dp.quit()

print(f"\n抓取完成！共获取 {len(all_jobs_detail)} 个职位的完整详情\n")

if all_jobs_detail:
    df = pd.DataFrame(all_jobs_detail)
    filename = '网络爬虫工程师_沈阳_完整详情.csv'
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"数据已保存到 {filename}")





















import time  # 导入时间模块
# 导入自动化模块
from DrissionPage import ChromiumPage

# 你的 headers（直接复制你提供的）
headers = {
    'Cookie': 'lastCity=101030100; ab_guid=f15d4127-6c3a-49c0-be9a-2cb0f549f738; wt2=DFGOALq23w-0wEMOPVsjnzAo8HrBjl6MxBNQg1skbrHVT4ynFedk2tYDcy2jodvK0ZqO854IAfD-s1cTAkvp6NQ~~; wbg=0; zp_at=uVevtNEGmGxDPGeJetYlNbRZMNwWwomqP0kAYloGAHs~; __zp_seo_uuid__=e9324f27-6642-49cc-940f-729c64d043d7; __l=r=https%3A%2F%2Fwww.google.com%2F&l=%2F&s=1; __g=-; Hm_lvt_194df3105ad7148dcf2b98a91b5e727a=1766238834,1766504279,1766817080; HMACCOUNT=6D5DCEC5E098F33C; Hm_lpvt_194df3105ad7148dcf2b98a91b5e727a=1766817083; __zp_stoken__=7afdgwpMnFU8fb2FbFFJWZMKPwoRSUVdsw4Z3wr%2FDgkhbwqvCtsK5asOEUsK9wrbCpk3CiW1VUsKXacKXwrTCnlHCsmbCsmjCqEjEgGPCncSiw67DgMOmwqjDgsODwpZBP8ODw77Fg8S%2BxYTEvcOjw77FhMS9xYPCvsWExL3Fg8OixITDlcOTxL7FhMS8xYPEvsWExL1PRUI4QkQ7LsK0PU8uRD1DPsK%2BPcOAw77Cvj3DgMO%2Bwr49w4DDvkcYIwoNWQ5ZEA0QDh9kYQlgWh8ODQkKZ11jDWdiEBETCQgLLjbCuFEKTAnDg8KTEMK9wp4Kw4HDiRfDjlluw4VNw4E1HTdPw7jDoz5CRcODxL1ER0NHRD0%2BRyZDQkc7OEJGQTg1OMK9w5zDk29zwrtUw4chGzpGRD1DPj91Qkc6OkJHQUFCRTU4O8KDM0PDj8KVw4TDi0JH; bst=V2RtkvFeL6311sVtRuyRocLy247DrfzSU~|RtkvFeL6311sVtRuyRocLy247DrRwSo~; __c=1766817080; __a=79240322.1766238834.1766504278.1766817080.72.3.5.72',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
}

# 打开浏览器
dp = ChromiumPage()

# === 新增部分：注入你的 headers（不修改你原来的代码）===
# 先访问首页，让浏览器建立正常环境
dp.get('https://www.zhipin.com/')

# 设置 User-Agent
dp.set.headers({'User-Agent': headers['User-Agent']})

# 设置 Cookie（关键！带上 __zp_stoken__ 等验证信息）
for cookie_str in headers['Cookie'].split('; '):
    name, value = cookie_str.split('=', 1)
    dp.set.cookies({name: value})

# === 新增结束 ===

# 访问网站（你的原代码）

dp.get('https://www.zhipin.com/web/geek/job?query=网络爬虫工程师')

# 替换你原来的 print(dp.json) 这行
print("页面标题：", dp.title)                  # 应该输出类似 “网络爬虫工程师招聘_网络爬虫工程师招聘信息【BOSS直聘】”
print("当前URL：", dp.url)                     # 确认是不是搜索页面
print(dp.html)                              # 如果想看完整源码（会很长，慎用）












import time
import pandas as pd
import os
import re
from DrissionPage import ChromiumPage

dp = ChromiumPage()

dp.listen.start('wapi/zpgeek/search/joblist.json')

# 搜索关键词（沈阳 + 网络爬虫工程师）
dp.get('https://www.zhipin.com/web/geek/jobs?query=网络爬虫工程师&city=101030100')

time.sleep(5) 

all_jobs_detail = []

for page_num in range(1, 4):  # 只抓1页
    print(f"\n=== 正在抓取第 {page_num} 页职位列表 ===")
    
    dp.scroll.to_bottom()
    time.sleep(3)
    
    resp = dp.listen.wait(timeout=10)
    
    if not resp or not resp.response.body:
        print("本页无更多数据，抓取结束")
        break
    
    json_data = resp.response.body
    jobList = json_data.get('zpData', {}).get('jobList', [])
    
    print(f"第 {page_num} 页抓到 {len(jobList)} 条职位，正在逐个抓取详情...")
    
    for job in jobList:
        basic_info = {
            '招聘岗位名称': job.get('jobName'),
            '招聘企业名称': job.get('brandName'),
            '薪资': job.get('salaryDesc'),
            '地点': f"{job.get('cityName', '')}-{job.get('areaDistrict', '')}",
            '工作经验': job.get('jobExperience', '未知'),
            '学历': job.get('jobDegree', '未知'),
            '职位关键词标签': ', '.join(job.get('skills', [])),
            '发布时间': '未知', # 这个字段将由公司页的“职位更新时间”填充
        }
        
        encryptJobId = job.get('encryptJobId')
        if not encryptJobId:
            continue
        
        # 1. 抓取职位详情页（仅抓取职位描述）
        lid = job.get('lid', '')
        securityId = job.get('securityId', '').lower()
        detail_url = f"https://www.zhipin.com/job_detail/{encryptJobId}.html"
        if lid and securityId:
            detail_url += f"?lid={lid}&securityId={securityId}"
        
        detail_tab = dp.new_tab(detail_url)
        detail_tab.wait.load_start()
        time.sleep(5)
        
        try:
            job_desc = detail_tab.ele('css:.job-sec-text', timeout=10).text.strip()
        except:
            job_desc = '未提取到职位描述'
        
        detail_tab.close()
        
        # 2. 抓取公司主页（提取工商信息 + 职位列表第一个职位更新时间）
        encryptBrandId = job.get('encryptBrandId') or job.get('brandId')
        company_info = {'企业介绍（BOSS直聘）': '未找到', '公司主页链接': '无'}
        gsb_info = {k: '未显示' for k in ['企业名称', '法定代表人', '成立时间', '企业类型', '经营状态', '注册资本', '注册地址', '营业期限', '所属地区', '统一社会信用代码', '核准日期', '曾用名', '登记机关', '所属行业', '经营范围']}
        
        if encryptBrandId:
            company_url = f"https://www.zhipin.com/gongsi/{encryptBrandId}.html"
            company_info['公司主页链接'] = company_url
            
            company_tab = dp.new_tab(company_url)
            company_tab.wait.load_start()
            
            # --- 关键修改：滚动到底部并提取公司页的更新时间 ---
            company_tab.scroll.to_bottom()
            time.sleep(7) # 公司页加载较慢，建议多等一会儿
            
            # 使用正则直接从公司页源码提取
            company_html = company_tab.html
            # 匹配 职位列表第一个职位更新时间:2025-09-29
            time_match = re.search(r'职位列表第一个职位更新时间[:：]\s*(\d{4}-\d{2}-\d{2})', company_html)
            
            if time_match:
                basic_info['发布时间'] = time_match.group(1)
            # --------------------------------------------

            try:
                intro_ele = company_tab.ele('css:.fold-text', timeout=8)
                if intro_ele:
                    company_info['企业介绍（BOSS直聘）'] = intro_ele.text.strip()
                
                gsb_map = {
                    'business-detail-name': '企业名称', 'business-detail-user': '法定代表人',
                    'business-detail-time': '成立时间', 'business-detail-type': '企业类型',
                    'business-detail-status': '经营状态', 'business-detail-money': '注册资本',
                    'business-detail-location': '注册地址', 'business-detail-business-time': '营业期限',
                    'business-detail-belone-location': '所属地区', 'business-detail-id': '统一社会信用代码',
                    'business-detail-check-time': '核准日期', 'business-detail-old-name': '曾用名',
                    'business-detail-orang': '登记机关',
                }
                
                for cls, key in gsb_map.items():
                    try:
                        ele = company_tab.ele(f'css:.{cls}', timeout=3)
                        if ele:
                            value = ele.text.split('：')[-1].strip() if '：' in ele.text else ele.text
                            gsb_info[key] = value
                    except:
                        pass
                
                # 经营范围和所属行业
                try:
                    range_ele = company_tab.ele('css:li.col-auto:last-child', timeout=3)
                    if range_ele and '经营范围' in range_ele.text:
                        gsb_info['经营范围'] = range_ele.text.split('：')[-1].strip()
                    
                    industry_ele = company_tab.ele('css:li.col-auto span.t:contains(所属行业)', timeout=3)
                    if industry_ele:
                        gsb_info['所属行业'] = industry_ele.parent().text.split('：')[-1].strip()
                except:
                    pass
                
            except Exception as e:
                print(f"公司页解析异常: {e}")
            
            company_tab.close()
        
        full_info = {
            **basic_info,
            '职位描述': job_desc,
            **company_info,
            **gsb_info,
            '职位详情链接': detail_url
        }
        
        all_jobs_detail.append(full_info)
        print(f"✓ 成功: {basic_info['招聘岗位名称']} @ {basic_info['招聘企业名称']} [公司页时间: {basic_info['发布时间']}]")
        
        time.sleep(3)

dp.listen.stop()
dp.quit()

if all_jobs_detail:
    df = pd.DataFrame(all_jobs_detail)
    filename = r'D:\BOSS直聘_完整招聘及工商数据22.csv'
    if os.path.exists(filename):
        os.remove(filename)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"数据已保存到 {filename}")














import pandas as pd

# 1. 定义新提取的企业数据 (共 15 条原始记录，去重后 14 家企业)
new_companies_data = [
    {
        "credit_code": "91430121MAD9XN9H85", "company_name": "长沙海予你传媒有限公司",
        "companydescription": "公司成立于3年，主要经营抖音直播。提供安静轻松的工作氛围，福利包含免费提供设备。",
        "legal_representative": "肖露", "establishment_date": "2024/1/10", "company_type": "有限责任公司（自然人投资或控股）",
        "business_status": "存续", "registered_capital": "5万人民币", "registered_address": "中国（湖南）自由贸易试验区长沙片区会展区块黄兴镇香樟东路230号会展配套产业园10栋105-578",
        "operating_period": "2024-01-10至2074-01-09", "region": "湖南省", "approval_date": "2024/1/11", "previous_names": "-",
        "registration_authority": "湖南省长沙市长沙县市场监督管理局", "industry": "未显示", "business_scope": "文艺创作;数字内容制作服务;计算机软硬件及辅助设备零售;体育竞赛组织等"
    },
    {
        "credit_code": "91120104MADHK16H5E", "company_name": "天津焱阳信息咨询服务有限公司",
        "companydescription": "公司成立于2010年。企业发展至今已有15年之久。旗下拥有多家分公司及单体药房，现有员工2000余人。",
        "legal_representative": "孙艳艳", "establishment_date": "2024/4/30", "company_type": "有限责任公司",
        "business_status": "存续", "registered_capital": "100万人民币", "registered_address": "天津市南开区东马路129号仁恒置地国际中心7层7单元0314",
        "operating_period": "2024-04-30至2074-04-29", "region": "天津市", "approval_date": "2024/4/30", "previous_names": "-",
        "registration_authority": "天津市南开区市场监督管理局", "industry": "未显示", "business_scope": "信息咨询服务;企业管理咨询;市场营销策划;食品销售;健康咨询服务等"
    },
    {
        "credit_code": "91120102MADYC9YP2U", "company_name": "天津众安锦合数据有限公司",
        "companydescription": "未找到",
        "legal_representative": "梁祐祯", "establishment_date": "2024/8/27", "company_type": "有限责任公司（法人独资）",
        "business_status": "存续", "registered_capital": "50万人民币", "registered_address": "天津市河东区上杭路街道万达广场B座807室",
        "operating_period": "2024-08-27至-", "region": "天津市", "approval_date": "2025/5/20", "previous_names": "-",
        "registration_authority": "天津市河东区市场监督管理局", "industry": "未显示", "business_scope": "互联网数据服务;大数据服务;信息技术咨询服务;软件开发;互联网安全服务等"
    },
    {
        "credit_code": "91330102MAEM1FXC71", "company_name": "职映（杭州）科技有限公司",
        "companydescription": "职映成立于2025年，原阿里字节谷歌团队，已经获得融资。致力于通过多模态人工智能＋算法提供一站式招聘解决方案。",
        "legal_representative": "王士铭", "establishment_date": "2025/6/9", "company_type": "有限责任公司（台港澳合资）",
        "business_status": "存续", "registered_capital": "100万元", "registered_address": "浙江省杭州市上城区九堡镇九环路9号1幢8楼B801-6室",
        "operating_period": "2025-06-09至-", "region": "浙江省", "approval_date": "2025/8/7", "previous_names": "-",
        "registration_authority": "杭州市上城区市场监督管理局", "industry": "未显示", "business_scope": "软件开发;数据处理和存储支持服务;云计算装备技术服务;信息系统集成服务等"
    },
    {
        "credit_code": "911101026631109996", "company_name": "北京瑞达恒建筑咨询有限公司",
        "companydescription": "瑞达恒公司（RCC Group）成立于2007年，建筑信息行业龙头。旗下包括瑞达恒工程信息网、慧讯网等。",
        "legal_representative": "王德存", "establishment_date": "2007/5/29", "company_type": "有限责任公司（自然人投资或控股）",
        "business_status": "存续", "registered_capital": "1000万人民币", "registered_address": "北京市西城区宣武门外大街6、8、10、12、16、18号10号楼12层1225-1237",
        "operating_period": "2007-05-29至2047-05-28", "region": "北京市", "approval_date": "2024/9/25", "previous_names": "-",
        "registration_authority": "北京市西城区市场监督管理局", "industry": "未显示", "business_scope": "因特网信息服务;建筑信息咨询;市场调查;软件开发;销售计算机软件等"
    },
    {
        "credit_code": "91310118MABR3UWG42", "company_name": "上海易安航空票务服务有限公司",
        "companydescription": "成立于2014年，专业为代理全国各航空公司客运行销的企业。与数十家航空公司达成合作。",
        "legal_representative": "张保军", "establishment_date": "2022/6/28", "company_type": "有限责任公司（自然人投资或控股）",
        "business_status": "存续", "registered_capital": "150万人民币", "registered_address": "上海市青浦区华新镇华腾路1288号1幢",
        "operating_period": "2022-06-28至2052-06-27", "region": "上海市", "approval_date": "2022/6/28", "previous_names": "-",
        "registration_authority": "青浦区市场监督管理局", "industry": "未显示", "business_scope": "旅客票务代理;咨询策划服务;市场营销策划;企业管理咨询;软件开发等"
    },
    {
        "credit_code": "92120102MA071NRA55", "company_name": "天津市河东区亿顺办公家具经营部",
        "companydescription": "主要做办公家具销售，人员关系融洽，没有层级压力。",
        "legal_representative": "曹文亚", "establishment_date": "2020/5/27", "company_type": "个体工商户",
        "business_status": "存续", "registered_capital": "1万元", "registered_address": "天津市河东区春华街道红星路顺驰桥旁124号",
        "operating_period": "2020-05-27至-", "region": "天津市", "approval_date": "2020/5/27", "previous_names": "-",
        "registration_authority": "天津市河东区市场监督管理局", "industry": "未显示", "business_scope": "家具销售;日用百货销售。"
    },
    {
        "credit_code": "91420100MA49NA275K", "company_name": "武汉德瑞森电子商务有限公司",
        "companydescription": "未找到",
        "legal_representative": "郑浩", "establishment_date": "2021/1/11", "company_type": "有限责任公司（自然人投资或控股）",
        "business_status": "存续", "registered_capital": "100万人民币", "registered_address": "武汉经济技术开发区12C2地块武汉经开万达广场B区S5-3栋11层B3-22室",
        "operating_period": "2021-01-11至-", "region": "湖北省", "approval_date": "2023/9/4", "previous_names": "-",
        "registration_authority": "武汉经济技术开发区市场监督管理局", "industry": "未显示", "business_scope": "食品销售;互联网销售;电子产品销售;翻译服务;广告制作等"
    },
    {
        "credit_code": "91110105MA01HE05X9", "company_name": "北京燃数科技有限公司",
        "companydescription": "致力于打造新一代智能数据分析平台，为客户提供在线协作的可视化商业决策工具。",
        "legal_representative": "邢志峰", "establishment_date": "2019/2/27", "company_type": "有限责任公司（法人独资）",
        "business_status": "存续", "registered_capital": "1000万人民币", "registered_address": "北京市朝阳区望京街9号商业楼3层1-325号095室",
        "operating_period": "2019-02-27至2049-02-26", "region": "北京市", "approval_date": "2025/3/24", "previous_names": "-",
        "registration_authority": "北京市朝阳区市场监督管理局", "industry": "未显示", "business_scope": "技术开发;基础软件服务;应用软件服务;软件开发;数据处理等"
    },
    {
        "credit_code": "91120104MAD8M5LEXA", "company_name": "津证云（天津）科技有限公司",
        "companydescription": "未找到",
        "legal_representative": "刘丽丽", "establishment_date": "2023/12/27", "company_type": "有限责任公司（自然人独资）",
        "business_status": "存续", "registered_capital": "200万人民币", "registered_address": "天津滨海高新区华苑产业区梅苑路5号金座广场-2007-1",
        "operating_period": "2023-12-27至-", "region": "天津市", "approval_date": "2025/11/25", "previous_names": "-",
        "registration_authority": "天津滨海高新技术产业开发区市场监督管理局", "industry": "未显示", "business_scope": "数据处理和存储支持服务;信息系统运行维护服务;互联网数据服务;软件开发等"
    },
    {
        "credit_code": "91120111MA0782CE33", "company_name": "天津纳兰云科技有限公司",
        "companydescription": "成立于2021年，以软件开发为核心，目前以信鸽行业APP和公棚ERP系统为主要方向。",
        "legal_representative": "李跃", "establishment_date": "2021/1/8", "company_type": "有限责任公司（自然人独资）",
        "business_status": "正常", "registered_capital": "100万", "registered_address": "天津市南开区南开三马路37号中关村e谷（南开）创想世界5A层5A08室",
        "operating_period": "2021-01-08至-", "region": "天津市", "approval_date": "2025/6/19", "previous_names": "-",
        "registration_authority": "天津市南开区市场监督管理局", "industry": "未显示", "business_scope": "第二类增值电信业务;拍卖业务;兽药经营等"
    },
    {
        "credit_code": "91120223MA05LNW29X", "company_name": "天津三联航空票务代理有限公司",
        "companydescription": "未找到",
        "legal_representative": "曲金环", "establishment_date": "2016/11/29", "company_type": "有限责任公司",
        "business_status": "存续", "registered_capital": "50万人民币", "registered_address": "天津市静海区静海镇旭华道海馨园小区对面30米",
        "operating_period": "2016-11-29至-", "region": "天津市", "approval_date": "2016/11/29", "previous_names": "-",
        "registration_authority": "天津市静海区市场监督管理局", "industry": "未显示", "business_scope": "票务代理;普通货运;仓储服务;汽车租赁;劳务服务;电子商务开发等"
    },
    {
        "credit_code": "91320481MADDRXJJ6J", "company_name": "常州鸣力科技有限公司",
        "companydescription": "未找到",
        "legal_representative": "何杰", "establishment_date": "2024/3/14", "company_type": "有限责任公司（自然人投资或控股）",
        "business_status": "存续", "registered_capital": "1000万元", "registered_address": "溧阳市溧城街道昆仑南路172号2幢1111号",
        "operating_period": "2024-03-14至-", "region": "江苏省", "approval_date": "2025/2/25", "previous_names": "-",
        "registration_authority": "溧阳市行政审批局", "industry": "未显示", "business_scope": "技术服务;信息技术咨询;网络技术服务;广告设计;电子产品销售等"
    },
    {
        "credit_code": "911101086996126161", "company_name": "中科天玑数据科技股份有限公司",
        "companydescription": "中科曙光旗下，专注于大数据智能方向的核心企业。在自然语言处理、知识图谱等领域有深厚积累。",
        "legal_representative": "王元兵", "establishment_date": "2010/1/4", "company_type": "其他股份有限公司（非上市）",
        "business_status": "存续", "registered_capital": "15000万人民币", "registered_address": "北京市海淀区西三旗建材城内3幢三层319号",
        "operating_period": "2010-01-04至2030-01-03", "region": "北京市", "approval_date": "2025/8/7", "previous_names": "北京中科天玑科技有限公司",
        "registration_authority": "北京市海淀区市场监督管理局", "industry": "未显示", "business_scope": "技术开发;数据处理;软件开发;计算机系统服务;广告设计等"
    }
]

# 2. 读取原 CSV 文件
try:
    df_old = pd.read_csv('corpsinfo.csv', encoding='utf-8-sig')
    print("成功读取原文件 'corpsinfo.csv'")
except FileNotFoundError:
    # 如果文件不存在，则创建一个空的 DataFrame，列名与要求一致
    columns = [
        "credit_code", "companydescription", "company_name", "legal_representative",
        "establishment_date", "company_type", "business_status", "registered_capital",
        "registered_address", "operating_period", "region", "approval_date",
        "previous_names", "registration_authority", "industry", "business_scope"
    ]
    df_old = pd.DataFrame(columns=columns)
    print("原文件不存在，已创建新表结构。")

# 3. 将新数据转换为 DataFrame
df_new = pd.DataFrame(new_companies_data)

# 4. 合并数据
# 使用 concat 合并，并根据 credit_code（统一社会信用代码）进行去重，保留最新的记录
df_combined = pd.concat([df_old, df_new], ignore_index=True)
df_combined.drop_duplicates(subset=['credit_code'], keep='last', inplace=True)

# 5. 保存回文件
output_file = 'corpsinfo_updated.csv'
df_combined.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"处理完成！已将新数据合并并保存至：{output_file}")
print(f"当前文件总行数：{len(df_combined)}")








