import time
import re
import pymysql
import pandas as pd
import os
from DrissionPage import ChromiumPage

# ================= 配置区 =================
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'XZGxzg20061203',  
    'database': 'jobs',
    'charset': 'utf8mb4'
}
# ==========================================

class BossSpider:
    def __init__(self):
        self.page = ChromiumPage()
        try:
            self.conn = pymysql.connect(**MYSQL_CONFIG)
            self.cursor = self.conn.cursor()
            print("✅ 数据库连接成功")
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            exit()

    def get_job_description(self, encrypt_id, lid, security_id):
        """1. 抓取职位详情页（完全保留你原有的 URL 构造和抓取逻辑）"""
        detail_url = f"https://www.zhipin.com/job_detail/{encrypt_id}.html"
        if lid and security_id:
            detail_url += f"?lid={lid}&securityId={security_id.lower()}"
        
        tab = self.page.new_tab(detail_url)
        tab.wait.load_start()
        time.sleep(5)
        
        desc = '未提取到职位描述'
        try:
            desc_ele = tab.ele('css:.job-sec-text', timeout=10)
            if desc_ele:
                desc = desc_ele.text.strip()
        except: pass
        tab.close()
        return desc, detail_url

    def get_company_details(self, brand_id):
        """2. 抓取公司主页"""
        company_url = f"https://www.zhipin.com/gongsi/{brand_id}.html"
        tab = self.page.new_tab(company_url)
        tab.wait.load_start()
        
        # 滚动到底部并等待，确保工商信息加载
        tab.scroll.to_bottom()
        time.sleep(7) 
        
        # A. 提取发布时间 (你原来的正则逻辑)
        pub_date = "未知"
        time_match = re.search(r'职位列表第一个职位更新时间[:：]\s*(\d{4}-\d{2}-\d{2})', tab.html)
        if time_match:
            pub_date = time_match.group(1)

        # B. 初始化工商信息字段 (按作业要求字段名)
        info = {
            'credit_code': '未显示', 'companydescription': '未找到', 'company_name': '未找到',
            'legal_representative': '未显示', 'establishment_date': '未显示', 'company_type': '未显示',
            'business_status': '未显示', 'registered_capital': '未显示', 'registered_address': '未显示',
            'operating_period': '未显示', 'region': '未显示', 'approval_date': '未显示',
            'previous_names': '未显示', 'registration_authority': '未显示', 'industry': '未显示', 'business_scope': '未显示'
        }

        try:
            # 提取企业介绍
            intro_ele = tab.ele('css:.fold-text', timeout=8)
            if intro_ele:
                info['companydescription'] = intro_ele.text.strip()

            # --- 关键：改回你原来的 business-detail 定位逻辑 ---
            gsb_map = {
                'business-detail-name': 'company_name',      # 企业名称
                'business-detail-user': 'legal_representative',
                'business-detail-time': 'establishment_date',
                'business-detail-type': 'company_type',
                'business-detail-status': 'business_status',
                'business-detail-money': 'registered_capital',
                'business-detail-location': 'registered_address',
                'business-detail-business-time': 'operating_period',
                'business-detail-belone-location': 'region',
                'business-detail-id': 'credit_code',         # 统一社会信用代码
                'business-detail-check-time': 'approval_date',
                'business-detail-old-name': 'previous_names',
                'business-detail-orang': 'registration_authority'
            }
            
            for cls, key in gsb_map.items():
                ele = tab.ele(f'css:.{cls}', timeout=3)
                if ele:
                    # 剥离“企业名称：”等前缀
                    info[key] = ele.text.split('：')[-1].strip() if '：' in ele.text else ele.text

            # 经营范围和所属行业 (保留你原有的 li.col-auto 定位)
            range_ele = tab.ele('css:li.col-auto:last-child', timeout=3)
            if range_ele and '经营范围' in range_ele.text:
                info['business_scope'] = range_ele.text.split('：')[-1].strip()
            
            ind_ele = tab.ele('css:li.col-auto span.t:contains(所属行业)', timeout=3)
            if ind_ele:
                info['industry'] = ind_ele.parent().text.split('：')[-1].strip()
        except:
            pass

        tab.close()
        return info, pub_date

    def save_to_db(self, company_data, job_data):
        """3. 数据保存（ON DUPLICATE KEY UPDATE 确保不报外键错）"""
        try:
            # 存 CorpsInfo
            c_cols = ', '.join(company_data.keys())
            c_place = ', '.join(['%s'] * len(company_data))
            c_upd = ', '.join([f"{k}=VALUES({k})" for k in company_data.keys() if k != 'credit_code'])
            c_sql = f"INSERT INTO CorpsInfo ({c_cols}) VALUES ({c_place}) ON DUPLICATE KEY UPDATE {c_upd}"
            self.cursor.execute(c_sql, list(company_data.values()))
            
            # 存 jobinfo
            j_cols = ', '.join(job_data.keys())
            j_place = ', '.join(['%s'] * len(job_data))
            j_sql = f"INSERT INTO jobinfo ({j_cols}) VALUES ({j_place})"
            self.cursor.execute(j_sql, list(job_data.values()))
            
            self.conn.commit()
            print(f"✓ 已入库: {job_data['jobname']} @ {company_data['company_name']}")
        except Exception as e:
            self.conn.rollback()
            print(f"❌ 入库失败: {e}")

    def run(self, max_pages=19):
        self.page.listen.start('wapi/zpgeek/search/joblist.json')
        self.page.get('https://www.zhipin.com/web/geek/jobs?query=爬虫工程师&city=101030100')
        
        for p in range(1, max_pages + 1):
            print(f"\n=== 正在抓取第 {p} 页 ===")
            resp = self.page.listen.wait(timeout=10)
            if not resp: break
            
            job_list = resp.response.body.get('zpData', {}).get('jobList', [])
            for j in job_list:
                # 抓取详情描述
                description, _ = self.get_job_description(j.get('encryptJobId'), j.get('lid'), j.get('securityId'))
                
                # 抓取公司工商及日期 (你的核心逻辑)
                corp_info, publish_date = self.get_company_details(j.get('encryptBrandId'))
                
                # 组装作业要求的 jobinfo 数据
                job_data = {
                    'jobname': j.get('jobName'),
                    'credit_code': corp_info['credit_code'],
                    'salary': j.get('salaryDesc'),
                    'location': f"{j.get('cityName')}-{j.get('areaDistrict', '')}",
                    'exp_year': j.get('jobExperience'),
                    'edu_level': j.get('jobDegree'),
                    'Description': description,
                    'tags': ",".join(j.get('skills', [])),
                    'publishdate': publish_date
                }
                
                self.save_to_db(corp_info, job_data)
                time.sleep(3)

        self.export_csv()
        self.conn.close()
        self.page.quit()

    def export_csv(self):
        """从数据库导出符合作业要求的两个 CSV"""
        print("\n📊 正在导出 CSV 文件...")
        df_job = pd.read_sql("SELECT * FROM jobinfo", self.conn)
        df_corp = pd.read_sql("SELECT * FROM CorpsInfo", self.conn)
        df_job.to_csv("jobinfo.csv", index=False, encoding='utf-8-sig')
        df_corp.to_csv("CorpsInfo.csv", index=False, encoding='utf-8-sig')
        print("✅ 导出成功: jobinfo.csv, CorpsInfo.csv")

if __name__ == "__main__":
    spider = BossSpider()
    spider.run(max_pages=19) # 你可以根据需要修改页数










import time
import re
import pymysql
import pandas as pd
from DrissionPage import ChromiumPage

# ================= 配置区 =================
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'XZGxzg20061203',  
    'database': 'jobs',
    'charset': 'utf8mb4'
}
# ==========================================

class BossSpider:
    def __init__(self):
        self.page = ChromiumPage()
        try:
            self.conn = pymysql.connect(**MYSQL_CONFIG)
            self.cursor = self.conn.cursor()
            print("✅ 数据库连接成功")
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            exit()

    def get_company_details(self, brand_id):
        """增强版抓取：深度等待15秒"""
        company_url = f"https://www.zhipin.com/gongsi/{brand_id}.html"
        tab = self.page.new_tab(company_url)
        tab.wait.load_start()
        
        tab.scroll.to_bottom()
        print(f"⏳ 正在为 ID {brand_id} 深度加载工商信息（15秒）...")
        time.sleep(15) 
        
        info = {
            'credit_code': brand_id, 'companydescription': '未找到', 'company_name': '未找到',
            'legal_representative': '未显示', 'establishment_date': '未显示', 'company_type': '未显示',
            'business_status': '未显示', 'registered_capital': '未显示', 'registered_address': '未显示',
            'operating_period': '未显示', 'region': '未显示', 'approval_date': '未显示',
            'previous_names': '未显示', 'registration_authority': '未显示', 'industry': '未显示', 'business_scope': '未显示'
        }

        try:
            intro_ele = tab.ele('css:.fold-text', timeout=5)
            if intro_ele:
                info['companydescription'] = intro_ele.text.strip()

            gsb_map = {
                'business-detail-name': 'company_name',
                'business-detail-user': 'legal_representative',
                'business-detail-time': 'establishment_date',
                'business-detail-type': 'company_type',
                'business-detail-status': 'business_status',
                'business-detail-money': 'registered_capital',
                'business-detail-location': 'registered_address',
                'business-detail-business-time': 'operating_period',
                'business-detail-belone-location': 'region',
                'business-detail-id': 'credit_code',
                'business-detail-check-time': 'approval_date',
                'business-detail-old-name': 'previous_names',
                'business-detail-orang': 'registration_authority'
            }
            
            for cls, key in gsb_map.items():
                ele = tab.ele(f'css:.{cls}', timeout=2)
                if ele:
                    info[key] = ele.text.split('：')[-1].strip() if '：' in ele.text else ele.text

            range_ele = tab.ele('css:li.col-auto:last-child', timeout=2)
            if range_ele and '经营范围' in range_ele.text:
                info['business_scope'] = range_ele.text.split('：')[-1].strip()
            
            ind_ele = tab.ele('css:li.col-auto span.t:contains(所属行业)', timeout=2)
            if ind_ele:
                info['industry'] = ind_ele.parent().text.split('：')[-1].strip()
        except: pass
        tab.close()
        return info

    def save_to_db(self, company_data):
        """仅更新或插入 CorpsInfo 避免 jobinfo 重复"""
        try:
            c_cols = ', '.join(company_data.keys())
            c_place = ', '.join(['%s'] * len(company_data))
            c_upd = ', '.join([f"{k}=VALUES({k})" for k in company_data.keys() if k != 'credit_code'])
            c_sql = f"INSERT INTO CorpsInfo ({c_cols}) VALUES ({c_place}) ON DUPLICATE KEY UPDATE {c_upd}"
            self.cursor.execute(c_sql, list(company_data.values()))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"❌ 入库失败: {e}")

    def run(self):
        self.page.listen.start('wapi/zpgeek/search/joblist.json')
        self.page.get('https://www.zhipin.com/web/geek/jobs?query=爬虫工程师&city=101030100')
        
        for p in range(1, 20):
            print(f"\n=== 正在处理第 {p} 页 ===")
            resp = self.page.listen.wait(timeout=10)
            if not resp: break
            
            job_list = resp.response.body.get('zpData', {}).get('jobList', [])
            for j in job_list:
                brand_id = j.get('encryptBrandId')
                brand_name = j.get('brandName')

                # 检查是否已存在（名字准确且不是“未找到”）
                check_sql = "SELECT count(*) FROM CorpsInfo WHERE (company_name = %s OR credit_code = %s) AND company_name != '未找到'"
                self.cursor.execute(check_sql, (brand_name, brand_id))
                if self.cursor.fetchone()[0] > 0:
                    print(f"⏩ 跳过已存在: {brand_name}")
                    continue

                print(f"🔍 补爬缺失: {brand_name}")
                corp_info = self.get_company_details(brand_id)
                self.save_to_db(corp_info)
                print(f"✅ 成功补全: {corp_info['company_name']}")
                time.sleep(2)

        self.export_csv()
        self.conn.close()
        self.page.quit()

    def export_csv(self):
        """导出函数"""
        print("\n📊 正在导出 CSV 文件...")
        # 导出时用 pd.read_sql，解决你之前遇到的 UserWarning，建议忽略或用 SQLAlchemy，但这里保持简单
        df_job = pd.read_sql("SELECT * FROM jobinfo", self.conn)
        df_corp = pd.read_sql("SELECT * FROM CorpsInfo", self.conn)
        df_job.to_csv("jobinfo_new.csv", index=False, encoding='utf-8-sig')
        df_corp.to_csv("CorpsInfo_new.csv", index=False, encoding='utf-8-sig')
        print("✅ 导出成功: jobinfo_new.csv, CorpsInfo_new.csv")

if __name__ == "__main__":
    spider = BossSpider()
    spider.run()
















