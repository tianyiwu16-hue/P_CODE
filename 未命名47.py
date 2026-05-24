import requests
from bs4 import BeautifulSoup

url = "http://goootech.com/signup_personal.aspx"

# 你的注册信息
info = {
    'UEmail': '123@qq.com',
    'URealName': '张三',
    'UNickName': '阿三',
    'UTel': '13888888888',
    'UPwd1': 'nicaibudao666',
    'UPwd2': 'nicaibudao666',
    'UCompany': 'XXX有限公司',
    'ChinaProvince': '河北',
    'ChinaCity': '唐山',
    'UCity$ChinaCityCode': '',
    'UTrade': '畜牧养殖',
}

session = requests.Session()

# ---------------------------
# Step 1: GET 页面，提取隐藏字段
# ---------------------------
r = session.get(url)
soup = BeautifulSoup(r.text, "html.parser")

def get_val(name):
    tag = soup.find("input", {"name": name})
    return tag["value"] if tag and "value" in tag.attrs else ""

viewstate = get_val("__VIEWSTATE")
eventvalidation = get_val("__EVENTVALIDATION")
viewstategenerator = get_val("__VIEWSTATEGENERATOR")

print("VIEWSTATE:", len(viewstate))
print("EVENTVALIDATION:", len(eventvalidation))

# ---------------------------
# Step 2: 人工输入验证码
# ---------------------------
captcha = input("请输入验证码（UValicode）：")

# ---------------------------
# Step 3: 构造 POST 提交表单
# ---------------------------
post_data = {
    "__EVENTTARGET": "",
    "__EVENTARGUMENT": "",
    "__VIEWSTATE": viewstate,
    "__VIEWSTATEGENERATOR": viewstategenerator,
    "__EVENTVALIDATION": eventvalidation,

    "UEmail": info["UEmail"],
    "URealName": info["URealName"],
    "UNickName": info["UNickName"],
    "UTel": info["UTel"],
    "UPwd1": info["UPwd1"],
    "UPwd2": info["UPwd2"],
    "UCompany": info["UCompany"],
    "ChinaProvince": info["ChinaProvince"],
    "ChinaCity": info["ChinaCity"],
    "UCity$ChinaCityCode": info["UCity$ChinaCityCode"],
    "UTrade": info["UTrade"],

    "UValicode": captcha,       # 验证码
    "btnOk": "确 定"            # 按钮字段名（ASP.NET 必须）
}

# ---------------------------
# Step 4: POST 提交注册
# ---------------------------
headers = {
    "User-Agent": "Mozilla/5.0"
}

resp = session.post(url, data=post_data, headers=headers)

print("\n服务器返回状态码:", resp.status_code)
print("是否注册成功（含跳转/提示信息）:")
print(resp.text[:500])










import os
import sys
import time
import winreg
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementNotInteractableException
from webdriver_manager.chrome import ChromeDriverManager

# ================= 配置区域 =================
# 替换为包含该表单的实际页面 URL
TARGET_URL = "http://goootech.com/signup_personal.aspx"  # 示例地址，请改为实际地址
DRIVER_PATH = r"D:\tools\chromedriver"
HEADLESS_MODE = False
# ===========================================

def get_chrome_version():
    """从注册表获取 Chrome 主版本号"""
    try:
        key_path = r"Software\Google\Chrome\BLBeacon"
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path)
        except FileNotFoundError:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
        version, _ = winreg.QueryValueEx(key, "version")
        print(f"[*] 检测到系统 Chrome 版本: {version}")
        return version.split('.')[0]
    except Exception as e:
        print(f"[!] 无法检测 Chrome 版本: {e}")
        return None

def setup_driver():
    """初始化 WebDriver (适配 webdriver_manager 4.x+)"""
    print(f"[*] 正在初始化 ChromeDriver...") # 路径参数已移除，使用默认缓存
    
    chrome_options = Options()
    if HEADLESS_MODE:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--ignore-certificate-errors")

    try:
        # --- 修复点：移除了 path 参数，直接调用 ---
        service = Service(ChromeDriverManager().install())
        # ----------------------------------------
        
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        print(f"[!] 驱动初始化失败: {e}")
        # 打印详细错误方便调试
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    get_chrome_version()
    driver = setup_driver()
    wait = WebDriverWait(driver, 20)

    try:
        print(f"[*] 打开网页: {TARGET_URL}")
        driver.get(TARGET_URL)

        # 1. 等待核心表单加载 (使用 ID: UEmail)
        try:
            wait.until(EC.presence_of_element_located((By.ID, "UEmail")))
            print("[*] 页面加载完成")
        except TimeoutException:
            print("[!] 页面加载超时，未找到表单元素。")
            return

        print("[*] 开始填写注册信息...")

        # 2. 填写基础文本框 (直接使用 HTML 中的 ID)
        # 字典映射: {Element_ID: Value}
        text_fields = {
            "UEmail": "test_user@example.com",
            "URealName": "张三",
            "UNickName": "SanZhang",
            "UTel": "13800138000",
            "UPwd1": "Test@1234",
            "UPwd2": "Test@1234",
            "UCompany": "测试科技发展有限公司"
        }

        for elem_id, value in text_fields.items():
            try:
                element = driver.find_element(By.ID, elem_id)
                element.clear()
                element.send_keys(value)
                print(f"    - 已填写 {elem_id}")
            except NoSuchElementException:
                print(f"[!] 错误: 未找到元素 ID {elem_id}")

        # 3. 处理省市联动 (Select Dropdown)
        print("[*] 处理省市选择...")
        try:
            # 选择省份
            prov_select = Select(driver.find_element(By.ID, "ChinaProvince"))
            prov_select.select_by_visible_text("北京") # 选择 Text 为 "北京" 的选项
            print("    - 省份选择完成: 北京")
            
            # 等待 JS 触发并刷新城市列表 (简单等待一点时间让 DOM 更新)
            time.sleep(1) 
            
            # 选择城市
            city_select = Select(driver.find_element(By.ID, "ChinaCity"))
            city_select.select_by_visible_text("北京市") # 选择 Text 为 "北京市"
            print("    - 城市选择完成: 北京市")

            # 验证 Hidden Field (UCity$ChinaCityCode) 是否自动填充
            # HTML name="UCity$ChinaCityCode" id="UCity_ChinaCityCode"
            hidden_code = driver.find_element(By.ID, "UCity_ChinaCityCode").get_attribute("value")
            if not hidden_code:
                # 如果 JS 没生效，手动注入值
                print("    - [!] 检测到隐藏域未自动填充，尝试手动注入...")
                driver.execute_script("document.getElementById('UCity_ChinaCityCode').value = '110100';")
            else:
                print(f"    - 城市编码自动获取成功: {hidden_code}")

        except Exception as e:
            print(f"[!] 省市选择出错: {e}")

        # 4. 处理行业单选框 (UTrade)
        # HTML 中是 input type="radio" name="UTrade"
        try:
            # 根据提供的 HTML，'环保' 的 value='22'
            target_trade_value = "22" 
            trade_radio = driver.find_element(By.XPATH, f"//input[@name='UTrade'][@value='{target_trade_value}']")
            if not trade_radio.is_selected():
                trade_radio.click()
            print("    - 行业选择完成 (环保)")
        except NoSuchElementException:
            print("[!] 警告: 未找到指定的行业单选框")

        # 5. 精确截取验证码 (核心需求)
        print("[*] 获取验证码...")
        captcha_input = driver.find_element(By.ID, "UValicode")
        
        try:
            # 直接定位图片元素 ID="chkimg"
            img_element = driver.find_element(By.ID, "chkimg")
            
            # 确保图片已加载
            wait.until(EC.visibility_of(img_element))
            
            # Selenium 4+ 支持直接对元素截图
            img_element.screenshot("captcha.png")
            print("    [√] 验证码图片已单独保存为 'captcha.png'")
            
            # 交互输入
            user_code = input("\n>>> 请查看目录下的 captcha.png 并输入验证码: ")
            captcha_input.send_keys(user_code)
            
        except Exception as e:
            print(f"[!] 验证码截图失败: {e}")
            print("    尝试保存全屏截图辅助...")
            driver.save_screenshot("full_page_debug.png")

        # 6. 提交注册
        # 按钮是 <input type="image" id="btnOk" ...>
        print("[*] 准备提交...")
        try:
            submit_btn = driver.find_element(By.ID, "btnOk")
            # 滚动到按钮位置确保可见
            driver.execute_script("arguments[0].scrollIntoView();", submit_btn)
            time.sleep(0.5)
            
            submit_btn.click()
            print("[*] 已点击提交按钮")
            
            # 7. 结果检测
            # 假设提交后 URL 变化或出现特定提示
            time.sleep(3) # 等待服务器响应
            print(f"[*] 当前页面标题: {driver.title}")
            print("[*] 注册流程操作结束，请检查浏览器实际结果。")
            
        except Exception as e:
            print(f"[!] 提交操作异常: {e}")

    except Exception as e:
        print(f"\n[!!!] 脚本运行发生致命错误: {e}")
        driver.save_screenshot("fatal_error.png")
    
    finally:
        if not HEADLESS_MODE:
            input("\n按回车键关闭浏览器...")
        driver.quit()

if __name__ == "__main__":
    main()









import os
import sys
import time
import winreg
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# --- 极简配置 ---
TARGET_URL = "http://goootech.com/signup_personal.aspx" 
HEADLESS_MODE = False
WAIT_TIME = 10 
# ----------------

def get_chrome_version():
    """获取 Chrome 主版本号 (功能1)"""
    try:
        key_path = r"Software\Google\Chrome\BLBeacon"
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path)
        version, _ = winreg.QueryValueEx(key, "version")
        print(f"[*] Chrome版本: {version.split('.')[0]}")
    except:
        pass

def setup_driver():
    """初始化 WebDriver (功能2, 3)"""
    chrome_options = Options()
    if HEADLESS_MODE:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--ignore-certificate-errors")

    try:
        service = Service(ChromeDriverManager().install()) # 使用默认路径
        return webdriver.Chrome(service=service, options=chrome_options)
    except Exception as e:
        print(f"[!] 驱动初始化失败: {e}")
        sys.exit(1)

def main():
    get_chrome_version()
    driver = setup_driver()
    wait = WebDriverWait(driver, WAIT_TIME)

    try:
        # 4. 打开目标网页
        driver.get(TARGET_URL)
        wait.until(EC.presence_of_element_located((By.ID, "UEmail")))

        # 5. 定位并填写表单
        fields = {
            "UEmail": "min@minimal.com", "URealName": "极简用户", "UNickName": "min_zs",
            "UTel": "13911110000", "UPwd1": "Pass1234", "UPwd2": "Pass1234",
            "UCompany": "极简科技公司"
        }
        for elem_id, value in fields.items():
            driver.find_element(By.ID, elem_id).send_keys(value)

        # 省市选择 (修正后的稳定逻辑)
        Select(driver.find_element(By.ID, "ChinaProvince")).select_by_visible_text("北京")
        
        target_city = "北京"
        city_xpath = f"//select[@id='ChinaCity']/option[text()='{target_city}']"
        wait.until(EC.presence_of_element_located((By.XPATH, city_xpath)))
        Select(driver.find_element(By.ID, "ChinaCity")).select_by_visible_text(target_city) 

        # 行业选择 (value='22' for 环保)
        driver.find_element(By.XPATH, "//input[@name='UTrade'][@value='22']").click()
        
        # 6. 截图验证码
        img_element = driver.find_element(By.ID, "chkimg")
        img_element.screenshot("captcha.png")
        user_code = input(">>> 请输入验证码 (captcha.png): ")
        driver.find_element(By.ID, "UValicode").send_keys(user_code)
        
        # 7. 点击提交按钮
        driver.find_element(By.ID, "btnOk").click()
        
        # 8. 检查结果 (简单等待)
        time.sleep(3)
        if driver.title != "个人用户注册--填写注册信息":
            print("[*] 注册成功或页面已跳转")
        else:
            print("[*] 注册失败或停留在原页面")

    except (TimeoutException, NoSuchElementException, Exception) as e:
        print(f"[!] 发生错误: {type(e).__name__}")
        driver.save_screenshot("error_minimal.png")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()




















