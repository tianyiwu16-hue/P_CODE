import json

# 示例数据
data = {
    "name": "张三",
    "age": 28,
    "city": "北京",
    "is_student": False,
    "courses": ["数学", "物理", "计算机科学"],
    "details": {
        "email": "zhangsan@example.com",
        "phone": "1234567890"
    }
}

# 将数据转换为JSON格式并写入文件
with open('D:/学习工作文件夹/原神表格文件.json', 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print("JSON文件已成功创建")