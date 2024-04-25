#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the dataset
file_path = '/content/The_World_Bank_Population_growth_(annual_).csv'
data = pd.read_csv(file_path)

# Calculate the average population growth rate for each year
average_growth_per_year = data.iloc[:, 2:].mean()

# Plotting the average world population growth over the years with a 10-year interval
average_growth_per_decade = average_growth_per_year[::10]
plt.figure(figsize=(14, 8))
plt.plot(average_growth_per_decade.index, average_growth_per_decade.values, marker='o', linestyle='-',
         color='b', markersize=7, linewidth=2, label='Average Growth per Decade')
plt.title('Decadal Average World Population Growth Trend (1961-2022)')
plt.xlabel('Year')
plt.ylabel('Average Population Growth (%)')
plt.xticks(rotation=45)
plt.grid(True)
z_decade = np.polyfit(average_growth_per_decade.index.astype(float), average_growth_per_decade.values, 1)
p_decade = np.poly1d(z_decade)
plt.plot(average_growth_per_decade.index, p_decade(average_growth_per_decade.index.astype(float)), "r--",
         linewidth=2, label='Trendline')
plt.legend()
plt.tight_layout()
plt.show()


# 2. Code for the Heatmap Visualization:
# 

# In[8]:


import seaborn as sns

# Sample a subset of countries for a clearer heatmap
data_sampled = data.sample(30, random_state=1).set_index('country_name').iloc[:, -12:-1]

# Creating the heatmap with the sampled data
plt.figure(figsize=(18, 15))
sns.heatmap(data_sampled, cmap='coolwarm', annot=False, linewidths=.5)
plt.title('Population Growth Rate Heatmap (2012-2021) - Sampled', fontweight='bold')
plt.xlabel('Year', fontweight='bold')
plt.ylabel('Country', fontweight='bold')
plt.show()


# 3. Code for the Bar Chart Visualization:
# 

# In[3]:


# Get the last year's data
last_year = data.columns[-1]
top_bottom_countries = data[['country_name', last_year]].set_index('country_name')
top_countries = top_bottom_countries.sort_values(by=last_year, ascending=False).head(5)
bottom_countries = top_bottom_countries.sort_values(by=last_year).head(5)

# Combining top and bottom countries for the bar chart
combined_countries = pd.concat([top_countries, bottom_countries])

# Creating the bar chart
plt.figure(figsize=(12, 7))
combined_countries[last_year].plot(kind='bar', color=['green' if x in top_countries.index else 'red' for x in combined_countries.index])
plt.title(f'Population Growth Rates in {last_year} for Top and Bottom 5 Countries')
plt.xlabel('Country')
plt.ylabel('Population Growth Rate (%)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# 4. Code for Displaying Statistical Analysis Table:
# 

# In[5]:


import matplotlib.pyplot as plt
import pandas as pd

# Load the summary statistics into a DataFrame for visualization
summary_stats_data = {
    'Statistic': ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
    'Value': [264.000000, 0.912771, 1.559630, -14.257037, 0.257862, 0.898242, 1.894213, 3.712988]
}
summary_stats_df = pd.DataFrame(summary_stats_data)

# Create a figure and a single subplot
fig, ax = plt.subplots(figsize=(8, 3))
# Hide axes
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)
# Tabulate data
tabulate_table = ax.table(cellText=summary_stats_df.values, colLabels=summary_stats_df.columns, loc='center', cellLoc = 'center')
tabulate_table.auto_set_font_size(False)
tabulate_table.set_fontsize(12)
tabulate_table.scale(1.2, 1.2)
# Save the table as an image
plt.savefig('Statistical_Analysis_Table.png')

# Return the path to the saved image
'Statistical_Analysis_Table.png'

