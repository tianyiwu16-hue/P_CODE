import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")


# Load the dataset to examine its structure and get an overview of the data
file_path = "D:\大学生生活方式数据\MentalHealthSurvey.csv"
mental_health_data = pd.read_csv(file_path)

# Display the first few rows and summary information
mental_health_data.head()



def horizontal_countplot_with_percentage(column_name):
    plt.figure(figsize=(12, 8))
    
    ax = sns.countplot(y=column_name, data=mental_health_data, palette='Set2')
    
    total = len(mental_health_data[column_name])
    
    for p in ax.patches:
        percentage = 100 * p.get_width() / total
        count = int(p.get_width())
        ax.annotate(f'{count}({percentage:.1f}%)', 
                    (p.get_width(), p.get_y() + p.get_height() / 2), 
                    ha='left', va='center', fontsize=10, color='black', 
                    xytext=(5, 0), textcoords='offset points')

    plt.title(f'Distribution of {column_name} (Count & Percentage)', fontsize=14)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(column_name, fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    
    
horizontal_countplot_with_percentage('university')



horizontal_countplot_with_percentage('stress_relief_activities')





# Bivariate analysis for 'university' vs mental health and well-being variables
# Using count plots to show the distribution of depression, anxiety, isolation, and future insecurity across university

mental_health_vars = ['depression', 'anxiety', 'isolation', 'future_insecurity']

plt.figure(figsize=(16, 12))

for i, var in enumerate(mental_health_vars, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x='university', hue=var, data=mental_health_data, palette='Set1')
    plt.title(f'University vs {var.replace("_", " ").title()}')
    plt.xlabel('University')
    plt.ylabel('Count')

plt.tight_layout()
plt.show()




# Multivariate analysis: Age vs Mental Health (depression, anxiety, isolation, future_insecurity) by degree_major using FacetGrid

plt.figure(figsize=(18, 12))

for i, var in enumerate(mental_health_vars, 1):
    g = sns.FacetGrid(mental_health_data, col='degree_major', height=4, aspect=1.2)
    g.map(sns.scatterplot, 'age', var)
    g.add_legend()
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle(f'Age vs {var.replace("_", " ").title()} by Degree Major')
    
plt.show()













































