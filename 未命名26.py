###3######1.
# import libraries and suppress warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Ensuring backend for matplotlib is Agg
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Ensuring plt backend is switched as required

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression

# Enable inline plotting
%matplotlib inline

# Set plot aesthetics
sns.set(style='whitegrid', palette='muted', color_codes=True)





# Load the dataset
data_path = "D:\大学生生活方式数据\Student Mental Health Analysis During Online Learning.csv"
df = pd.read_csv(data_path, encoding='ascii', delimiter=',')

# Quick overview of the dataset
print('Dataset Shape:', df.shape)
print('\nData Types:\n', df.dtypes)

# Display first few rows (this output will be generated when the notebook is run)
df.head()



# Check for missing values
missing_values = df.isnull().sum()
print('Missing Values per Column:\n', missing_values)

# Basic cleaning: drop rows with missing essential values (if any)
df.dropna(inplace=True)

# Convert categorical columns to numeric if needed for modeling
le = LabelEncoder()

# Encode 'Gender'
df['Gender_encoded'] = le.fit_transform(df['Gender'])

# Encode 'Education Level'
df['EducationLevel_encoded'] = le.fit_transform(df['Education Level'])

# Encode 'Stress Level' (our predictive target)
df['StressLevel_encoded'] = le.fit_transform(df['Stress Level'])

# Optionally encode other categorical fields if used later
df['AnxiousBeforeExams_encoded'] = le.fit_transform(df['Anxious Before Exams'])
df['AcademicPerformanceChange_encoded'] = le.fit_transform(df['Academic Performance Change'])

print('Data cleaning completed. Dataframe now has the following columns:')
print(df.columns.tolist())






# Plot Histogram for Screen Time, Sleep Duration, and Physical Activity
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.histplot(df['Screen Time (hrs/day)'], kde=True, color='skyblue')
plt.title('Distribution of Screen Time (hrs/day)')

plt.subplot(1, 3, 2)
sns.histplot(df['Sleep Duration (hrs)'], kde=True, color='salmon')
plt.title('Distribution of Sleep Duration (hrs)')

plt.subplot(1, 3, 3)
sns.histplot(df['Physical Activity (hrs/week)'], kde=True, color='lime')
plt.title('Distribution of Physical Activity (hrs/week)')

plt.tight_layout()
plt.show()

# Count plot for Stress Level
plt.figure(figsize=(6, 4))
sns.countplot(x='Stress Level', data=df, palette='Set2')
plt.title('Count of Stress Level Categories')
plt.show()


# Prepare data for correlation heatmap: use only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Check that we have at least four numeric variables
if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(10, 8))
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.show()
else:
    print('Not enough numeric columns for a correlation heatmap.')

# Pair Plot for select numerical features
sns.pairplot(numeric_df[['Age', 'Screen Time (hrs/day)', 'Sleep Duration (hrs)', 'Physical Activity (hrs/week)', 'StressLevel_encoded']])
plt.suptitle('Pair Plot of Key Numerical Features', y=1.02)
plt.show()











###33######2.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from tabulate import tabulate

# إعدادات العرض
sns.set(style='whitegrid', palette='muted', color_codes=True)

# النمذجة التنبؤية
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# تثبيت العشوائية
import random
random.seed(42)
np.random.seed(42)


# مكتبة الإحصاء والتحذيرات
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')





# Load the Dataset
# Read the dataset into a pandas DataFrame 
df = pd.read_csv("D:\大学生生活方式数据\Student Mental Health Analysis During Online Learning.csv")

# Preview the Top Rows
# Display the first three rows of the dataset to get an initial look at the data
df.head()




df.tail()





df.shape





df.describe().T.plot(kind='bar')







numeric_cols = df.select_dtypes(include=['number']).columns

# رسم التوزيع لكل عمود رقمي
for col in numeric_cols:
    sns.histplot(x=col, data=df, kde=True)
    plt.show()






# Check for missing values
missing_values = df.isnull().sum()
print('Missing Values per Column:\n', missing_values)

# Basic cleaning: drop rows with missing essential values (if any)
df.dropna(inplace=True)

# Convert categorical columns to numeric if needed for modeling
le = LabelEncoder()

# Encode 'Gender'
df['Gender_encoded'] = le.fit_transform(df['Gender'])

# Encode 'Education Level'
df['EducationLevel_encoded'] = le.fit_transform(df['Education Level'])

# Encode 'Stress Level' (our predictive target)
df['StressLevel_encoded'] = le.fit_transform(df['Stress Level'])

# Optionally encode other categorical fields if used later
df['AnxiousBeforeExams_encoded'] = le.fit_transform(df['Anxious Before Exams'])
df['AcademicPerformanceChange_encoded'] = le.fit_transform(df['Academic Performance Change'])

print('Data cleaning completed. Dataframe now has the following columns:')
print(df.columns.tolist())






# Plot Histogram for Screen Time, Sleep Duration, and Physical Activity
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.histplot(df['Screen Time (hrs/day)'], kde=True, color='skyblue')
plt.title('Distribution of Screen Time (hrs/day)')

plt.subplot(1, 3, 2)
sns.histplot(df['Sleep Duration (hrs)'], kde=True, color='salmon')
plt.title('Distribution of Sleep Duration (hrs)')

plt.subplot(1, 3, 3)
sns.histplot(df['Physical Activity (hrs/week)'], kde=True, color='lime')
plt.title('Distribution of Physical Activity (hrs/week)')

plt.tight_layout()
plt.show()

# Count plot for Stress Level
plt.figure(figsize=(6, 4))
sns.countplot(x='Stress Level', data=df, palette='Set2')
plt.title('Count of Stress Level Categories')
plt.show()


# Prepare data for correlation heatmap: use only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Check that we have at least four numeric variables
if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(10, 8))
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.show()
else:
    print('Not enough numeric columns for a correlation heatmap.')

# Pair Plot for select numerical features
sns.pairplot(numeric_df[['Age', 'Screen Time (hrs/day)', 'Sleep Duration (hrs)', 'Physical Activity (hrs/week)', 'StressLevel_encoded']])
plt.suptitle('Pair Plot of Key Numerical Features', y=1.02)
plt.show()






# Define features and target for prediction
features = ['Age', 'Screen Time (hrs/day)', 'Sleep Duration (hrs)', 'Physical Activity (hrs/week)', 'Gender_encoded', 'EducationLevel_encoded']
target = 'StressLevel_encoded'

# Split into training and testing sets
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression classifier
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate prediction accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Prediction Accuracy:', accuracy)

# Generate and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()











#############3.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("D:\大学生生活方式数据\Student Mental Health Analysis During Online Learning.csv")
df.head()



df.info()




df.describe()





df.columns





df.isnull().sum()




df.duplicated().sum()





df.columns= [col.replace(" ","_").lower() for col in df.columns]






edu_levels=df["education_level"].value_counts()
edu_levels.head()





sns.countplot(data=df, x="education_level",color="orange")
plt.xticks(rotation=45)
plt.show()




sns.countplot(data=df,x="education_level",palette="dark:y",hue="stress_level")
plt.xticks(rotation=45)
plt.show()





sns.countplot(data=df,x="stress_level",hue="anxious_before_exams",palette="dark:r")
plt.xticks(rotation=45)
plt.show()




def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    filtered_out = df[(df[col] > low) & (df[col] < high)]
    return filtered_out

numeric_col = df.select_dtypes(include='number').columns
for col in numeric_col:
    removed_out = remove_outliers(df, col)
    
    
    
    
colors=['g', 'b', 'r', 'c', 'm', 'y', 'k']
for i,col in enumerate(numeric_col):
    sns.boxplot(x=df[col],color=colors[i%len(colors)])
    plt.title("Boxplot of {}".format(col))
    plt.show()



# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[["age", "screen_time_(hrs/day)", "sleep_duration_(hrs)", "physical_activity_(hrs/week)"]].corr(), annot=True, cmap="viridis")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()



# Histograms for Numerical Features
num_cols = ["age", "screen_time_(hrs/day)", "sleep_duration_(hrs)", "physical_activity_(hrs/week)"]
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=20, color="skyblue")
    plt.title(f"Distribution of {col.replace('_',' ').title()}")
    plt.xlabel(col.replace('_',' ').title())
    plt.ylabel("Count")
    plt.show()
    
    
    


def z_score(df,columns):
    df_normalized=df.copy()
    for col in columns:
        mean=df[col].mean()
        std=df[col].std()
        df_normalized[col]=(df[col]-mean)/std
    return df_normalized
df_normalized=z_score(df,numeric_col)


df_normalized.head()



cat_col=df.select_dtypes(include=['object']).columns
cat_col



df_normalized['gender'] = df_normalized['gender'].apply(lambda x: 1 if x.lower().strip() == 'male' else 0)
df_normalized.head()



df_normalized['education_level'] = df_normalized['education_level'].astype(str).str.lower().str.strip()
edu_uq = df_normalized['education_level'].unique()
edu_map = {value: idx for idx, value in enumerate(edu_uq)}
df_normalized['education_level'] = df_normalized['education_level'].map(edu_map)




stress_uq = df['stress_level'].str.lower().unique()
stress_map = {value: idx for idx, value in enumerate(stress_uq)}
df_normalized['stress_level'] = df['stress_level'].str.lower().map(stress_map)
df_normalized.head()




df_normalized['anxious_before_exams']=df_normalized['anxious_before_exams'].str.lower().map({'yes':1,'no':0})
df_normalized.head()




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np

features = df_normalized.drop(['name', 'academic_performance_change'], axis=1)
pca = PCA()
pca.fit(features)
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
pca = PCA(n_components=n_components)
x_pca = pca.fit_transform(features)
df_normalized['academic_performance_change'] = df_normalized['academic_performance_change'].apply(lambda x: 1 if x.lower() == 'same' else 0)
y = df_normalized['academic_performance_change']

X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prb = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
print(f"accuracy_score:{acc:2f}")







































