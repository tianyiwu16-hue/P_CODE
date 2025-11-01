import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings as w
w.filterwarnings('ignore')
import math


df=pd.read_csv("D:\大学生生活方式数据\student_depression_dataset.csv")
df.head()




df.isnull().sum()


sns.set(style='whitegrid')


num_cols=df.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols=df.select_dtypes(include=['object']).columns.tolist()



n=len(num_cols)
cols=2
rows=math.ceil(n/cols)
fig,axes=plt.subplots(rows,cols,figsize=(cols*4,rows*4))
axes=axes.flatten()

for i , col in enumerate(num_cols):
    sns.histplot(df[col],kde=True,ax=axes[i],bins=20)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()


n=len(cat_cols)
cols=2
rows=math.ceil(n/cols)
fig,axes=plt.subplots(rows,cols,figsize=(cols*6,rows*4))
axes=axes.flatten()

for i , col in enumerate(cat_cols):
    sns.histplot(df[col],kde=True,ax=axes[i],bins=20)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].set_xticklabels(axes[i].get_xticklabels(),rotation=90)
plt.tight_layout()
plt.show()



from sklearn.model_selection import train_test_split
x=df.drop('Depression',axis=1)
y=df['Depression']
num_cols=x.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols=x.select_dtypes(include=['object']).columns.tolist()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
preprocessing = ColumnTransformer([('scaler',StandardScaler(),num_cols),
                                  ('encoder',OneHotEncoder(drop='first',handle_unknown='ignore'),cat_cols)])
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
model=Pipeline([('pre',preprocessing),
               ('model',DecisionTreeClassifier())])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(f'Accuracy Score: {accuracy_score(y_test,y_pred) *100 :.2f}')




from sklearn.model_selection import train_test_split
x=df.drop('Depression',axis=1)
y=df['Depression']
num_cols=x.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols=x.select_dtypes(include=['object']).columns.tolist()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
preprocessing = ColumnTransformer([('scaler',StandardScaler(),num_cols),
                                  ('encoder',OneHotEncoder(drop='first',handle_unknown='ignore'),cat_cols)])

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Classifier': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

from sklearn.pipeline import Pipeline
results={}
for name,model in models.items():
    pipeline=Pipeline([('pre',preprocessing),
                      ('model',model)])
    pipeline.fit(x_train,y_train)
    y_pred=pipeline.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    results[name]=acc
    print(f'Accuracy score for {name} is {acc *100 :.2f}')
    
    
    
    






















































