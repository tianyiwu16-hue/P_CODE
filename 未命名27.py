# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




import pandas as pd

df = pd.read_csv(r"D:\大学生生活方式数据\student_lifestyle_dataset.csv")
df


df.shape


df.columns



df.head(10)



df.describe(include='all')




df.info()




df.isnull().sum()




df.duplicated().sum()




df.nunique()




df.dtypes



from summarytools import dfSummary



df = pd.read_csv(r"D:\大学生生活方式数据\student_lifestyle_dataset.csv")
df.tail()



dfSummary(df)




import matplotlib.pyplot as plt
import seaborn as sns



import warnings
warnings.filterwarnings('ignore')



df['Study_Hours_Per_Day'].value_counts()



plt.figure(figsize=(10,6))

plt.xlabel("Study_Hours_Per_Day", fontsize=20)
plt.ylabel("Count", fontsize=20)

plt.title("Histplot of Study_Hours_Per_Day", fontsize=23)

sns.histplot(data=df, x="Study_Hours_Per_Day", kde=True, color='teal')

plt.show()







high_stress = df[df['Stress_Level'] == 'High']['Stress_Level'].count()
moderate_stress = df[df['Stress_Level'] == 'Moderate']['Stress_Level'].count()
low_stress = df[df['Stress_Level'] == 'Low']['Stress_Level'].count()

index_values = [high_stress, moderate_stress, low_stress]
index_labels = ['High', 'Moderate', 'Low']

plt.pie(index_values, labels = index_labels, autopct='%2.2f%%')

plt.title('Overall Distribution of Stress Level in Students Dataframe', fontsize=17)

plt.show()





sns.jointplot(x='Extracurricular_Hours_Per_Day', y='Physical_Activity_Hours_Per_Day', data=df, kind='hex', color='teal')
plt.suptitle('Joint Plot of Physical_Activity_Hours_Per_Day and Extracurricular_Hours_Per_Day', y=1.02, fontsize=12)
plt.show()




plt.figure(figsize=(10,6))
sns.regplot(x='Physical_Activity_Hours_Per_Day', y='Social_Hours_Per_Day', data=df, scatter_kws={'alpha':0.5}, line_kws = {'color':'red'}, color='green')
plt.title('Social_Hours_Per_Day vs. Physical_Activity_Hours_Per_Day', fontsize=19)
plt.xlabel('Physical_Activity_Hours_Per_Day', fontsize=17)
plt.ylabel('Social_Hours_Per_Day', fontsize=17)
plt.show()










plt.figure(figsize=(10,6))

plt.xlabel("Physical_Activity_Hours_Per_Day", fontsize=20)
plt.ylabel("Count", fontsize=20)

plt.title("Histplot of Physical_Activity_Hours_Per_Day", fontsize=23)

sns.histplot(data=df, x="Physical_Activity_Hours_Per_Day", kde=True, palette='magma', hue='Stress_Level', multiple='stack')

plt.show()




plt.figure(figsize=(10,6))
sns.violinplot(x='Physical_Activity_Hours_Per_Day', data=df, color='gold')

plt.title('Violin Plot of Physical_Activity_Hours_Per_Day', fontsize=20)

plt.xlabel('Physical_Activity_Hours_Per_Day', fontsize=18)

plt.show()



from sklearn.preprocessing import LabelEncoder


df['Stress_Level_Encoded'] = LabelEncoder().fit_transform(df['Stress_Level'])
df['Stress_Level_Encoded']


df['Stress_Level']



sns.jointplot(x='Extracurricular_Hours_Per_Day', y='Physical_Activity_Hours_Per_Day', data=df, kind='hex', color='teal')
plt.suptitle('Joint Plot of Physical_Activity_Hours_Per_Day and Extracurricular_Hours_Per_Day', y=1.02, fontsize=12)
plt.show()



plt.figure(figsize=(10,6))
sns.regplot(x='Study_Hours_Per_Day', y='GPA', data=df, scatter_kws={'alpha':0.5}, line_kws = {'color':'red'})
plt.title('GPA vs. Study_Hours_Per_Day', fontsize=19)
plt.xlabel('Study_Hours_Per_Day', fontsize=17)
plt.ylabel('GPA', fontsize=17)
plt.show()




plt.figure(figsize=(10,6))
sns.regplot(x='Physical_Activity_Hours_Per_Day', y='GPA', data=df, scatter_kws={'alpha':0.5}, line_kws = {'color':'red'})
plt.title('GPA vs. Physical_Activity_Hours_Per_Day', fontsize=19)
plt.xlabel('Physical_Activity_Hours_Per_Day', fontsize=17)
plt.ylabel('GPA', fontsize=17)
plt.show()





sns.jointplot(y='Social_Hours_Per_Day', x='Physical_Activity_Hours_Per_Day', data=df, kind='hex', color='orange')
plt.suptitle('Joint Plot of Social_Hours_Per_Day and Physical_Activity_Hours_Per_Day', y=1.02, fontsize=12)
plt.show()



plt.figure(figsize=(10,6))
sns.regplot(x='Physical_Activity_Hours_Per_Day', y='Social_Hours_Per_Day', data=df, scatter_kws={'alpha':0.5}, line_kws = {'color':'red'}, color='green')
plt.title('Social_Hours_Per_Day vs. Physical_Activity_Hours_Per_Day', fontsize=19)
plt.xlabel('Physical_Activity_Hours_Per_Day', fontsize=17)
plt.ylabel('Social_Hours_Per_Day', fontsize=17)
plt.show()




plt.figure(figsize=(10,6))

df.plot(kind='hexbin', x='Physical_Activity_Hours_Per_Day', y='Study_Hours_Per_Day', gridsize=20)

plt.title('Hexagonal Binning Plot', fontsize=19)

plt.xlabel('Physical_Activity_Hours_Per_Day', fontsize=17)
plt.ylabel('Study_Hours_Per_Day', fontsize=17)

plt.show()




plt.figure(figsize=(10,6))

sns.scatterplot(x='Study_Hours_Per_Day', y='GPA', data=df, hue='Stress_Level', palette='RdBu')



plt.figure(figsize=(10,6))

sns.scatterplot(x='Sleep_Hours_Per_Day', y='GPA', data=df, hue='Stress_Level', palette='magma')


plt.figure(figsize=(10,6))

sns.scatterplot(x='Extracurricular_Hours_Per_Day', y='GPA', data=df, hue='Stress_Level', palette='viridis')




plt.figure(figsize=(10,6))

sns.scatterplot(x='Social_Hours_Per_Day', y='GPA', data=df, hue='Stress_Level', palette='muted')




plt.figure(figsize=(10,6))

sns.scatterplot(x='Physical_Activity_Hours_Per_Day', y='GPA', data=df, hue='Stress_Level', palette='plasma')





plt.figure(figsize=(10,6))

sns.scatterplot(x='Stress_Level_Encoded', y='GPA', data=df, hue='Stress_Level', palette='mako')




sns.pairplot(df)




sns.pairplot(df, kind='kde')





sns.pairplot(df, hue='Stress_Level', palette='viridis')




plt.figure(figsize=(10,6))

sns.scatterplot(x='Stress_Level_Encoded', y='GPA', data=df, hue='Stress_Level', palette='mako')




import numpy as np

plt.figure(figsize=(10,6))

corr_matrix = df.drop(columns=['Student_ID', 'Stress_Level', 'Stress_Level_Encoded']).corr()

sns.heatmap(corr_matrix, annot=True, cmap='RdBu', fmt='.2f')

plt.title('Correlation Heatmap', fontsize=19)
plt.show()






############2.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




import pandas as pd

# Load the dataset from the file path
data = pd.read_csv("D:\大学生生活方式数据\student_lifestyle_dataset.csv")

# Display the first few rows to check the data
data.head()



# Get general information about the dataset
print(data.info())

# Get a statistical summary of the dataset
print(data.describe())

# Check for any missing values
print(data.isnull().sum())




# Display the column names to see if there are any discrepancies
print(data.columns)




import matplotlib.pyplot as plt

# Example: Study Hours vs GPA
plt.scatter(data['Study_Hours_Per_Day'], data['GPA'], color='blue')
plt.title('Study Hours vs GPA')
plt.xlabel('Study Hours')
plt.ylabel('GPA')
plt.show()






#######33.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, accuracy_score, classification_report

import warnings
warnings.filterwarnings("ignore")




# Load the dataset
df = pd.read_csv("D:\大学生生活方式数据\student_lifestyle_dataset.csv")




column_name = 'Stress_Level'
plt.figure(figsize=(10, 4))

# First subplot: Count plot
plt.subplot(1, 2, 1)
sns.countplot(y=column_name, data=df, palette='muted')  
plt.title(f'Distribution of {column_name}')

ax = plt.gca()
for p in ax.patches:
    ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2), 
                ha='center', va='center', xytext=(10, 0), textcoords='offset points')

sns.despine(left=True, bottom=True)

# Second subplot: Pie chart
plt.subplot(1, 2, 2)
df[column_name].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('muted'), startangle=90, explode=[0.05]*df[column_name].nunique())
plt.title(f'Percentage Distribution of {column_name}')
plt.ylabel('')  

plt.tight_layout()
plt.show()







# Define muted colors from the palette
muted_colors = sns.color_palette("muted")

# Scatter plots with linear regression lines 
plt.figure(figsize=(14, 14))

# Study Hours vs. GPA
plt.subplot(3, 2, 1)
sns.regplot(x='Study_Hours_Per_Day', y='GPA', data=df,
            color=muted_colors[0], scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
plt.title('Study Hours Per Day vs. GPA')
plt.xlabel('Study Hours Per Day')
plt.ylabel('GPA')

# Sleep Hours vs. GPA
plt.subplot(3, 2, 2)
sns.regplot(x='Sleep_Hours_Per_Day', y='GPA', data=df,
            color=muted_colors[1], scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
plt.title('Sleep Hours Per Day vs. GPA')
plt.xlabel('Sleep Hours Per Day')
plt.ylabel('GPA')

# Physical Activity Hours vs. GPA
plt.subplot(3, 2, 3)
sns.regplot(x='Physical_Activity_Hours_Per_Day', y='GPA', data=df,
            color=muted_colors[2], scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
plt.title('Physical Activity Hours Per Day vs. GPA')
plt.xlabel('Physical Activity Hours Per Day')
plt.ylabel('GPA')

# Social Hours vs. GPA
plt.subplot(3, 2, 4)
sns.regplot(x='Social_Hours_Per_Day', y='GPA', data=df,
            color=muted_colors[3], scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
plt.title('Social Hours Per Day vs. GPA')
plt.xlabel('Social Hours Per Day')
plt.ylabel('GPA')

# Extracurricular Hours vs. GPA
plt.subplot(3, 2, 5)
sns.regplot(x='Extracurricular_Hours_Per_Day', y='GPA', data=df,
            color=muted_colors[4], scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
plt.title('Extracurricular Hours Per Day vs. GPA')
plt.xlabel('Extracurricular Hours Per Day')
plt.ylabel('GPA')

plt.tight_layout()
plt.show()




numerical_features = [
    'Study_Hours_Per_Day', 
    'Sleep_Hours_Per_Day', 
    'Physical_Activity_Hours_Per_Day', 
    'Social_Hours_Per_Day', 
    'Extracurricular_Hours_Per_Day'
]

# Create boxplots for each numerical feature grouped by Stress_Level
plt.figure(figsize=(14, 12))

for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x='Stress_Level', y=feature, data=df, palette='muted')
    plt.title(f'{feature.replace("_", " ")} Across Stress Levels')
    plt.xlabel('Stress Level')
    plt.ylabel(feature.replace("_", " "))

plt.tight_layout()
plt.show()




# Numerical features against GPA, grouped by Stress_Level
plt.figure(figsize=(14, 12))

numerical_features = [
    'Study_Hours_Per_Day', 
    'Sleep_Hours_Per_Day', 
    'Physical_Activity_Hours_Per_Day', 
    'Social_Hours_Per_Day', 
    'Extracurricular_Hours_Per_Day'
]

# Plot each feature
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 2, i)
    sns.scatterplot(
        x=feature, 
        y='GPA', 
        hue='Stress_Level', 
        data=df, 
        palette='muted', 
        alpha=0.7
    )
    plt.title(f'{feature.replace("_", " ")} vs GPA by Stress Level')
    plt.xlabel(feature.replace('_', ' '))
    plt.ylabel('GPA')
    plt.legend(title='Stress Level')

plt.tight_layout()
plt.show()



plt.figure(figsize=(14, 12))

numerical_features = [
    'Study_Hours_Per_Day', 
    'Sleep_Hours_Per_Day', 
    'Physical_Activity_Hours_Per_Day', 
    'Social_Hours_Per_Day', 
    'Extracurricular_Hours_Per_Day'
]

# Define colors for different stress levels
stress_colors = sns.color_palette('muted')

# Plot each feature
for i, feature in enumerate(numerical_features):
    plt.subplot(3, 2, i + 1)
    
    # Scatter plot with matching colors for stress levels
    sns.scatterplot(
        x=feature,
        y='GPA',
        hue='Stress_Level',
        data=df,
        palette=stress_colors,
        alpha=0.7
    )
    
    # Add linear regression lines for each stress level with matching colors
    for j, stress_level in enumerate(df['Stress_Level'].unique()):
        sns.regplot(
            x=feature,
            y='GPA',
            data=df[df['Stress_Level'] == stress_level],
            scatter=False,
            color=stress_colors[j], # Use the same color for the line as in the scatter plot
            line_kws={'label': stress_level}
        )
    
    plt.title(f'{feature.replace("_", " ")} vs GPA by Stress Level')
    plt.xlabel(feature.replace('_', ' '))
    plt.ylabel('GPA')
    plt.legend(title='Stress Level')

plt.tight_layout()
plt.show()



# Define thresholds for low and high GPA
low_gpa_threshold = 2.5
high_gpa_threshold = 3.5  

# Filter students with low GPA (less than or equal to the threshold)
low_gpa_students = df[df['GPA'] <= low_gpa_threshold][
    ['GPA', 'Study_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'Stress_Level']
]

# Display results in a clear format
print("----- Students with Low GPA (<= 2.5) -----")
display(low_gpa_students)

# Filter students with high GPA (equal to the threshold)
high_gpa_students = df[df['GPA'] >= high_gpa_threshold][
    ['GPA', 'Study_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'Stress_Level']
]

# Display results in a clear format
print("\n----- Students with High GPA (>= 3.5) -----")
display(high_gpa_students)






# Combine low and high GPA students for comparison
low_high_gpa_comparison = pd.concat([low_gpa_students, high_gpa_students])

# Create scatter plots to compare study hours and physical activity hours with stress level and GPA
plt.figure(figsize=(14, 12))

# Study Hours vs GPA
plt.subplot(2, 2, 1)
sns.scatterplot(
    x='Study_Hours_Per_Day', 
    y='GPA', 
    hue='Stress_Level', 
    data=low_high_gpa_comparison, 
    palette='muted', 
    alpha=0.7
)
plt.title('Study Hours vs GPA by Stress Level')
plt.xlabel('Study Hours Per Day')
plt.ylabel('GPA')
plt.legend(title='Stress Level')

# Physical Activity Hours vs GPA
plt.subplot(2, 2, 2)
sns.scatterplot(
    x='Physical_Activity_Hours_Per_Day', 
    y='GPA', 
    hue='Stress_Level', 
    data=low_high_gpa_comparison, 
    palette='muted', 
    alpha=0.7
)
plt.title('Physical Activity Hours vs GPA by Stress Level')
plt.xlabel('Physical Activity Hours Per Day')
plt.ylabel('GPA')
plt.legend(title='Stress Level')

# Study Hours vs Stress Level
plt.subplot(2, 2, 3)
sns.boxplot(
    x='Stress_Level', 
    y='Study_Hours_Per_Day', 
    data=low_high_gpa_comparison, 
    palette='muted'
)
plt.title('Study Hours by Stress Level')
plt.xlabel('Stress Level')
plt.ylabel('Study Hours Per Day')

# Physical Activity Hours vs Stress Level
plt.subplot(2, 2, 4)
sns.boxplot(
    x='Stress_Level', 
    y='Physical_Activity_Hours_Per_Day', 
    data=low_high_gpa_comparison, 
    palette='muted'
)
plt.title('Physical Activity Hours by Stress Level')
plt.xlabel('Stress Level')
plt.ylabel('Physical Activity Hours Per Day')

plt.tight_layout()
plt.show()



# Identify the maximum and minimum GPA in the dataset
max_gpa = df['GPA'].max()
min_gpa = df['GPA'].min()

# Count the number of students with maximum and minimum GPA
num_students_max_gpa = df[df['GPA'] == max_gpa].shape[0]
num_students_min_gpa = df[df['GPA'] == min_gpa].shape[0]

max_gpa_info = {
    "Max GPA": max_gpa,
    "Number of Students with Max GPA": num_students_max_gpa
}

min_gpa_info = {
    "Min GPA": min_gpa,
    "Number of Students with Min GPA": num_students_min_gpa
}

print("--- GPA Summary Statistics ---\n")
print("Maximum GPA Information:")
for key, value in max_gpa_info.items():
    print(f"- {key}: {value}")

print("\nMinimum GPA Information:")
for key, value in min_gpa_info.items():
    print(f"- {key}: {value}")



sns.pairplot(df, hue='Stress_Level', diag_kind='hist', palette='muted')








































