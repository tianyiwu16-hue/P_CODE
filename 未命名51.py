# -*- coding: utf-8 -*-
import csv
import time
import random
from DrissionPage import ChromiumPage, ChromiumOptions

class BossScraper:
    def __init__(self):
        co = ChromiumOptions()
        # 使用你的 Edge 路径
        edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\143.0.3650.96\msedge.exe"
        co.set_browser_path(edge_path)
        
        self.page = ChromiumPage(co)
        self.job_id_counter = 1
        self.processed_corps = set() 
        
        # 初始化 CSV (使用 'w' 覆盖模式)
        self.init_csv()
        
        # 启动数据包监听
        self.page.listen.start('joblist.json')

    def init_csv(self):
        """初始化并打开 CSV 文件，覆盖原表"""
        self.job_file = open('jobinfo.csv', 'w', encoding='utf-8-sig', newline='')
        self.job_writer = csv.writer(self.job_file)
        self.job_writer.writerow(['id', 'jobname', 'credit_code', 'salary', 'location', 
                                 'exp_year', 'edu_level', 'Description', 'tags', 'publishdate'])
        
        self.corp_file = open('corpsinfo.csv', 'w', encoding='utf-8-sig', newline='')
        self.corp_writer = csv.writer(self.corp_file)
        self.corp_writer.writerow(['credit_code', 'companydescription', 'company_name', 
                                  'legal_representative', 'establishment_date', 'company_type', 
                                  'business_status', 'registered_capital', 'registered_address', 
                                  'operating_period', 'region', 'approval_date', 'previous_names', 
                                  'registration_authority', 'industry', 'business_scope'])

    def parse_detail_page(self, url):
        """处理详情页：精准提取 JD、公司简介和工商信息"""
        tab = self.page.new_tab(url)
        time.sleep(random.uniform(2, 4)) 
        
        res_data = {
            'description': 'N/A',
            'company_intro': 'N/A',
            'biz': {k: 'N/A' for k in ['credit_code', 'company_name', 'legal_representative', 
                                       'establishment_date', 'company_type', 'business_status', 'registered_capital', 
                                       'registered_address', 'operating_period', 'region', 'approval_date', 
                                       'previous_names', 'registration_authority', 'industry', 'business_scope']}
        }

        try:
            # 1. 提取职位描述 (JD) - 寻找包含“职位描述”标题的块
            jd_header = tab.ele('text=职位描述')
            if jd_header:
                # 找到标题后，取其同级的文本内容块
                jd_text = tab.ele('.job-sec-text')
                res_data['description'] = jd_text.text.strip() if jd_text else "N/A"

            # 2. 提取公司介绍 - 寻找包含“公司介绍”标题的块
            intro_header = tab.ele('text=公司介绍')
            if intro_header:
                # 公司介绍通常在 jd-container 之后的另一个 job-sec-text 中
                intro_text = intro_header.parent().ele('.job-sec-text')
                res_data['company_intro'] = intro_text.text.strip() if intro_text else "N/A"

            # 3. 提取工商信息
            # 点击“查看全部”或展开箭头
            expand_btn = tab.ele('.business-detail-info-btn', timeout=2)
            if expand_btn:
                tab.actions.move_to(expand_btn).click()
                time.sleep(0.8)
            
            biz_section = tab.ele('.business-detail')
            if biz_section:
                items = biz_section.eles('tag:li')
                mapping = {
                    '统一社会信用代码': 'credit_code', '企业名称': 'company_name',
                    '法定代表人': 'legal_representative', '成立日期': 'establishment_date',
                    '企业类型': 'company_type', '经营状态': 'business_status',
                    '注册资本': 'registered_capital', '注册地址': 'registered_address',
                    '营业期限': 'operating_period', '所属地区': 'region',
                    '核准日期': 'approval_date', '曾用名': 'previous_names',
                    '登记机关': 'registration_authority', '所属行业': 'industry',
                    '经营范围': 'business_scope'
                }
                for item in items:
                    text = item.text.replace('：', ':')
                    if ':' in text:
                        k, v = text.split(':', 1)
                        k = k.strip()
                        if k in mapping:
                            res_data['biz'][mapping[k]] = v.strip()
        except Exception as e:
            print(f"详情页解析异常: {e}")
        
        tab.close()
        return res_data

    def run(self, keyword):
        search_url = f'https://www.zhipin.com/web/geek/job?query={keyword}&city=100010000'
        self.page.get(search_url)
        
        print("请在浏览器中完成登录和验证...")
        
        while True:
            packet = self.page.listen.wait(timeout=10)
            if not packet:
                print("未捕获到数据包，尝试手动滚动或检查是否已到底")
                break

            json_data = packet.response.body
            if not json_data or 'zpData' not in json_data:
                continue
                
            job_list = json_data['zpData'].get('jobList', [])
            
            for job in job_list:
                job_name = job.get('jobName')
                salary = job.get('salaryDesc')
                location = f"{job.get('cityName')}·{job.get('areaDistrict', '')}"
                exp_year = job.get('jobExperience')
                edu_level = job.get('jobDegree')
                tags = ",".join(job.get('skills', []))
                pub_ts = job.get('lastModifyTime')
                publishdate = time.strftime('%Y-%m-%d', time.localtime(pub_ts/1000)) if pub_ts else 'N/A'
                
                detail_url = f"https://www.zhipin.com/job_detail/{job['encryptJobId']}.html"
                print(f"正在抓取 [{self.job_id_counter}]: {job_name} - {job['brandName']}")
                
                # 抓取详情
                detail_res = self.parse_detail_page(detail_url)
                biz = detail_res['biz']
                
                # 信用代码：如果工商信息里没抓到，就填 N/A，不要填公司名
                credit_code = biz['credit_code']
                # 唯一标识用于去重（如果没信用代码，用公司名去重）
                corp_key = credit_code if credit_code != 'N/A' else job.get('brandName')
                
                # 写入 jobinfo
                self.job_writer.writerow([
                    self.job_id_counter, job_name, credit_code, salary, location,
                    exp_year, edu_level, detail_res['description'], tags, publishdate
                ])
                self.job_id_counter += 1
                
                # 写入 corpsinfo (去重)
                if corp_key not in self.processed_corps:
                    self.corp_writer.writerow([
                        credit_code, detail_res['company_intro'], biz['company_name'] if biz['company_name'] != 'N/A' else job.get('brandName'),
                        biz['legal_representative'], biz['establishment_date'], biz['company_type'],
                        biz['business_status'], biz['registered_capital'], biz['registered_address'],
                        biz['operating_period'], biz['region'], biz['approval_date'],
                        biz['previous_names'], biz['registration_authority'], biz['industry'],
                        biz['business_scope']
                    ])
                    self.processed_corps.add(corp_key)
                
                self.job_file.flush()
                self.corp_file.flush()
                time.sleep(random.uniform(3, 5))

            # 翻页逻辑修复
            try:
                # 寻找“下一页”按钮
                next_btn = self.page.ele('css:a.next')
                # 检查按钮是否存在且没有被禁用
                if next_btn and 'disabled' not in next_btn.attrs.get('class', []):
                    print("点击下一页...")
                    next_btn.click()
                    time.sleep(random.uniform(4, 6))
                else:
                    print("已到达最后一页或无法找到翻页按钮")
                    break
            except Exception as e:
                print(f"翻页尝试失败: {e}")
                break

    def __del__(self):
        try:
            self.job_file.close()
            self.corp_file.close()
        except:
            pass

if __name__ == '__main__':
    scraper = BossScraper()
    try:
        scraper.run('爬虫工程师')
    except KeyboardInterrupt:
        print("程序被手动停止")
    except Exception as e:
        print(f"运行崩溃: {e}")