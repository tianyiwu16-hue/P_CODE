"""
GPA Predictor Web App (Flask)

How this file is organized:
1. Train a RandomForestRegressor on your uploaded dataset `student_lifestyle_dataset.xlsx` and save the model to `rf_model.joblib`.
2. Start a lightweight Flask web app with:
   - GET / : a simple HTML form to input the five features
   - POST /predict : returns an HTML page with predicted GPA and actionable suggestions
   - POST /api/predict : JSON API that returns prediction and suggestions

Requirements (install with pip):
    pip install flask pandas scikit-learn joblib

Run:
    python gpa_predictor_app.py

Then open http://127.0.0.1:5000 in your browser.

Notes:
- This is a single-file demo useful for local testing and demos. For production you should:
  * put the model file in a managed storage, use environment configs
  * add input validation & authentication for any hosted endpoint
  * run under gunicorn/nginx
"""

import os
import math
from flask import Flask, request, render_template_string, jsonify, send_from_directory
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# ---------------------- Configuration ----------------------
DATA_PATH = r"D:\数据要素大赛作品\Desktopgpa_project\student_lifestyle_dataset.xlsx"  # change if needed
MODEL_PATH = "./rf_model.joblib"
RANDOM_STATE = 42
FEATURES = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
]
TARGET = "GPA"

# ---------------------- Train (if model not exists) ----------------------
def train_and_save_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please place your Excel dataset there.")

    df = pd.read_excel(data_path)
    # Basic sanity: drop rows with NA in features/target
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    X = df[FEATURES]
    y = df[TARGET]

    # train-test split just for demonstration
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    # save model
    joblib.dump(rf, model_path)
    print(f"Model trained and saved to {model_path}")
    return rf

# ---------------------- Utility: advice generation ----------------------
def generate_advice(input_features: dict, prediction: float):
    """
    Returns a list of short, actionable suggestions based on the input features and the predicted GPA.
    The rules below are simple, interpretable heuristics derived from the SHAP/PDP analysis done earlier.
    """
    adv = []

    study = input_features.get("Study_Hours_Per_Day")
    sleep = input_features.get("Sleep_Hours_Per_Day")
    physical = input_features.get("Physical_Activity_Hours_Per_Day")
    social = input_features.get("Social_Hours_Per_Day")
    extra = input_features.get("Extracurricular_Hours_Per_Day")

    # Study: positive returns up to a point (approx 6-8h). If below 4, encourage more focused study
    if study is not None:
        if study < 4:
            adv.append(f"学习时间偏少（{study:.1f}h/天）：建议每天安排稳定的2-4段学习专注时段，总计至少4–6小时，使用番茄钟提升效率。")
        elif study < 6:
            adv.append(f"学习时间合理（{study:.1f}h/天）：保持当前学习习惯，进一步提高学习质量（有计划、有目标）。")
        elif study <= 8:
            adv.append(f"学习时间充足（{study:.1f}h/天）：你已处于高产出区间，注意保持良好的学习-休息节奏，避免疲劳。")
        else:
            adv.append(f"学习时间过长（{study:.1f}h/天）：增加学习时间回报递减，建议优化学习方法并确保充足睡眠与运动。")

    # Sleep: aim for 6-8
    if sleep is not None:
        if sleep < 6:
            adv.append(f"睡眠偏少（{sleep:.1f}h/天）：建议目标睡眠 6–8 小时，睡眠不足会影响记忆与学习效率。")
        elif 6 <= sleep <= 8:
            adv.append(f"睡眠充足（{sleep:.1f}h/天）：保持当前睡眠习惯，有利于巩固学习效果。")
        else:
            adv.append(f"睡眠偏多（{sleep:.1f}h/天）：若感到精力不足，检查昼间活动与睡眠质量，适度活动可提高效率。")

    # Physical activity: moderate (0.5-2h)
    if physical is not None:
        if physical < 0.5:
            adv.append(f"运动较少（{physical:.1f}h/天）：建议每日安排至少30分钟中等强度运动，有助于注意力与心情。")
        elif physical <= 2:
            adv.append(f"运动适中（{physical:.1f}h/天）：保持，适度运动对学习有辅助作用。")
        else:
            adv.append(f"运动较多（{physical:.1f}h/天）：若占用学习时间过多，建议适当平衡运动与学习时间。")

    # Social: moderate
    if social is not None:
        if social > 3:
            adv.append(f"社交时间较长（{social:.1f}h/天）：若影响学习，尝试把社交集中到休息时段或周末。")
        elif social < 0.5:
            adv.append(f"社交较少（{social:.1f}h/天）：适度社交有助于心理健康，建议每周安排社交活动。")
        else:
            adv.append(f"社交适中（{social:.1f}h/天）：保持平衡，避免影响深度学习时间。")

    # Extracurricular: moderate
    if extra is not None:
        if extra > 4:
            adv.append(f"课外活动占比较大（{extra:.1f}h/天）：若影响学业，可考虑优先级排序或减少每周频次。")
        elif extra > 0:
            adv.append(f"适度课外活动（{extra:.1f}h/天）：有助于综合素质发展，建议保证不影响主要课程学习。 ")
        else:
            adv.append(f"无课外活动：可以挑选一项兴趣活动帮助减压与提升综合能力（适度为宜）。")

    # General recommendation based on predicted GPA
    if prediction is not None:
        if prediction >= 3.5:
            adv.insert(0, f"预测 GPA：{prediction:.2f}（优秀）— 继续保持，你的时间分配比较有效，可精细化提升弱项。）")
        elif prediction >= 3.0:
            adv.insert(0, f"预测 GPA：{prediction:.2f}（良好）— 有提升空间，优化学习效率与保证睡眠通常能带来提升。")
        elif prediction >= 2.5:
            adv.insert(0, f"预测 GPA：{prediction:.2f}（中等）— 建议增加高质量学习时间并优化学习方法，同时保证睡眠与运动。")
        else:
            adv.insert(0, f"预测 GPA：{prediction:.2f}（较低）— 建议调整学习计划、寻求学业辅导并改善生活习惯（睡眠/运动/饮食）。")

    # remove duplicates and limit to 8 adv lines
    seen = set()
    adv_unique = []
    for s in adv:
        if s not in seen:
            adv_unique.append(s)
            seen.add(s)
        if len(adv_unique) >= 8:
            break
    return adv_unique

# ---------------------- Flask App ----------------------
app = Flask(__name__)

# load or train model
if os.path.exists(MODEL_PATH):
    rf_model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
else:
    print("Training model because no saved model found...")
    rf_model = train_and_save_model()

# Simple HTML template (Bootstrap-free, lightweight)
HOME_HTML = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <title>GPA 预测器</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body{font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; padding:20px}
      .card{max-width:720px;margin:0 auto;padding:20px;border:1px solid #eee;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,0.03)}
      label{display:block;margin-top:12px}
      input[type=number]{width:100%;padding:8px;margin-top:6px;box-sizing:border-box}
      button{margin-top:16px;padding:10px 16px;background:#2563eb;color:white;border:none;border-radius:6px}
      .result{margin-top:18px;padding:12px;background:#f8fafc;border-radius:8px}
      ul.suggestions{padding-left:18px}
      #total-warning {
        font-size: 0.9em;
        margin-top: 8px;
        color: #666;
        font-weight: 500;
      }
      #total-warning.error {
        color: red;
      }
    </style>
  </head>
  <body>
    <div class="card">
      <h2>GPA 预测器（基于随机森林）</h2>
      <p>输入五项每日平均小时数，点击预测，系统会返回预测 GPA 并给出改进建议。</p>
      <form method="post" action="/predict">
        {% for f in FEATURES %}
        <label for="{{f}}">{{f.replace('_',' ').replace('Per Day', '（小时/天）')}}</label>
        <input step="0.1" required type="number" name="{{f}}" id="{{f}}" value="{{defaults[f]}}" oninput="updateTotal()">
        {% endfor %}
        <p id="total-warning">总计：0.0 小时</p>
        <button type="submit">预测 GPA</button>
      </form>
      {% if suggestions is defined %}
      <div class="result">
        {% if prediction is defined %}
        <h3>预测结果：{{ "%.2f" % prediction }}</h3>
        {% endif %}
        <h4>建议</h4>
        <ul class="suggestions">
        {% for s in suggestions %}
          <li>{{s}}</li>
        {% endfor %}
        </ul>
      </div>
      {% endif %}
    </div>

    <script>
      function updateTotal() {
        const fields = {{ FEATURES|tojson }};  <!-- 修复：现在 FEATURES 已从 Python 传入 -->
        let total = 0;
        fields.forEach(f => {
          const val = parseFloat(document.getElementById(f)?.value || 0);
          total += isNaN(val) ? 0 : val;
        });
        const warning = document.getElementById('total-warning');
        warning.textContent = `总计：${total.toFixed(1)} 小时`;
        if (total > 24) {
          warning.className = 'error';
        } else {
          warning.className = '';
        }
      }
      // 初始化显示
      document.addEventListener('DOMContentLoaded', updateTotal);
    </script>
  </body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    # default values are median of dataset for convenience
    try:
        df = pd.read_excel(DATA_PATH)
        defaults = {f: float(df[f].median()) for f in FEATURES}
    except Exception:
        defaults = {f: 1.0 for f in FEATURES}
    return render_template_string(
        HOME_HTML,
        FEATURES=FEATURES,      # ✅ 关键修复：传入 FEATURES 给模板
        defaults=defaults
    )

@app.route('/predict', methods=['POST'])
def predict_form():
    try:
        input_vals = {}
        for f in FEATURES:
            v = request.form.get(f, type=float)
            if v is None:
                return "缺少输入: %s" % f, 400
            input_vals[f] = float(v)

        # ✅ 校验总时间是否超过 24 小时
        total_hours = sum(input_vals.values())
        if total_hours > 24:
            suggestions = [f"⚠️ 输入无效：总时间为 {total_hours:.1f} 小时，超过了一天的 24 小时，请合理分配时间。"]
            return render_template_string(
                HOME_HTML,
                FEATURES=FEATURES,
                defaults=input_vals,
                suggestions=suggestions
            )

        X_input = [[input_vals[f] for f in FEATURES]]
        pred = float(rf_model.predict(X_input)[0])
        suggestions = generate_advice(input_vals, pred)

        return render_template_string(
            HOME_HTML,
            FEATURES=FEATURES,
            defaults=input_vals,
            prediction=pred,
            suggestions=suggestions
        )
    except Exception as e:
        return f"预测时出错: {e}", 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    JSON API endpoint. Expects JSON like:
      { "Study_Hours_Per_Day": 5.0, "Extracurricular_Hours_Per_Day": 1.0, ... }
    Returns JSON:
      { "prediction": 3.12, "suggestions": [ ... ] }
    """
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "invalid json"}), 400

    try:
        input_vals = {f: float(payload.get(f, 0.0)) for f in FEATURES}
        total_hours = sum(input_vals.values())
        if total_hours > 24:
            return jsonify({
                "error": f"总时间 {total_hours:.1f} 小时超过 24 小时",
                "suggestions": ["每日活动总时间不能超过 24 小时，请检查输入。"]
            }), 400

        X_input = [[input_vals[f] for f in FEATURES]]
        pred = float(rf_model.predict(X_input)[0])
        suggestions = generate_advice(input_vals, pred)
        return jsonify({"prediction": pred, "suggestions": suggestions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# static route to download model or outputs if needed
@app.route('/downloads/<path:filename>')
def downloads(filename):
    base = os.path.abspath('.')
    return send_from_directory(base, filename, as_attachment=True)

if __name__ == '__main__':
    print('Starting Flask app...')
    app.run(debug=True)