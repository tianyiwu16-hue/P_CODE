import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# 输出目录
os.makedirs("synthetic_dataset", exist_ok=True)

# 基础参数
n_students = 2000
start_date = datetime(2022, 9, 1)
end_date = datetime(2023, 6, 30)
days = (end_date - start_date).days + 1

user_ids = [f"U{str(i).zfill(4)}" for i in range(1, n_students+1)]

# 1. DemoGraphic 表
faculties = ["Engineering", "Science", "Arts", "Business", "Medicine"]
residencies = ["On-campus", "Off-campus"]

demo = pd.DataFrame({
    "user_id": user_ids,
    "gender": np.random.choice(["M", "F"], n_students),
    "birth_year": np.random.choice(range(1999, 2005), n_students),
    "faculty": np.random.choice(faculties, n_students),
    "residency": np.random.choice(residencies, n_students)
})
demo.to_csv("synthetic_dataset/DemoGraphic.csv", index=False)

# 时间生成工具
def random_dates(start, end, n):
    return [start + timedelta(seconds=random.randint(0, int((end-start).total_seconds()))) for _ in range(n)]

# 2. LocationLog
buildings = ["Canteen", "Dorm", "Library", "LectureHall", "Gym", "Shop"]
loc_rows = 1_000_000
loc = pd.DataFrame({
    "user_id": np.random.choice(user_ids, loc_rows),
    "datetime": random_dates(start_date, end_date, loc_rows),
    "building": np.random.choice(buildings, loc_rows),
    "floor": np.random.randint(1, 6, loc_rows),
    "latitude": np.random.uniform(39.9, 40.1, loc_rows),
    "longitude": np.random.uniform(32.7, 32.9, loc_rows)
})
loc.to_csv("synthetic_dataset/LocationLog.csv", index=False)

# 3. ConsumptionLog
stores = ["Cafe", "Canteen", "Stationery", "Supermarket", "GymStore"]
categories = ["Food", "Drink", "Books", "Clothes", "Sports", "Services", "Entertainment", "Transport", "Other"]
cons_rows = 400_000
cons = pd.DataFrame({
    "user_id": np.random.choice(user_ids, cons_rows),
    "datetime": random_dates(start_date, end_date, cons_rows),
    "store": np.random.choice(stores, cons_rows),
    "amount": np.random.uniform(0, 100, cons_rows).round(2),
    "category": np.random.choice(categories, cons_rows)
})
cons.to_csv("synthetic_dataset/ConsumptionLog.csv", index=False)

# 4. LibrarySeat
lib_rows = 90_000
lib = pd.DataFrame({
    "user_id": np.random.choice(user_ids, lib_rows),
    "date": np.random.choice(pd.date_range(start_date, end_date), lib_rows),
    "enter_time": np.random.randint(8, 20, lib_rows),
    "exit_time": np.random.randint(9, 22, lib_rows),
    "seat_id": np.random.randint(1, 500, lib_rows),
    "floor": np.random.randint(1, 5, lib_rows)
})
lib.to_csv("synthetic_dataset/LibrarySeat.csv", index=False)

# 5. DormEntry
dorm_rows = 150_000
dorm = pd.DataFrame({
    "user_id": np.random.choice(user_ids, dorm_rows),
    "datetime": random_dates(start_date, end_date, dorm_rows),
    "direction": np.random.choice(["in", "out"], dorm_rows)
})
dorm.to_csv("synthetic_dataset/DormEntry.csv", index=False)

# 6. GradeItem
grade_rows = 18_000
grade = pd.DataFrame({
    "user_id": np.random.choice(user_ids, grade_rows),
    "course_id": [f"C{random.randint(100,999)}" for _ in range(grade_rows)],
    "credit": np.random.choice([2, 3, 4], grade_rows),
    "grade_point": np.random.uniform(0, 4, grade_rows).round(2)
})
grade.to_csv("synthetic_dataset/GradeItem.csv", index=False)

print("✅ 六张模拟数据表已生成，保存在 synthetic_dataset 文件夹中！")




import os
print("当前工作目录是：")
print(os.getcwd())