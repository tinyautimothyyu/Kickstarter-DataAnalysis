import numpy as np
import pandas as pd 
import seaborn as sns
import mysql.connector as connection
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn import metrics

# Create connection to the database
try:
    mydb = connection.connect(host='localhost',
                                         database='sys',
                                         user='root',
                                         password='menor0312')
    if mydb.is_connected():
        db_Info = mydb.get_server_info()
        print("Connected to MySQL Server version ", db_Info)
        cursor = mydb.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)

except Exception as e:
    print("Error while connecting to MySQL", e)

query = "SELECT *, DATEDIFF(deadline,launched) AS duration_days \
        FROM campaign \
        WHERE (outcome = 'successful' OR outcome = 'failed') AND (launched >= '2010-01-01' AND launched < '2018-01-01') AND (deadline >= '2010-01-01' AND deadline < '2018-01-01')"
df = pd.read_sql(query,mydb)

# set a time lable to group different campaing duration
df['time_id'] = df['duration_days']
for i in range(len(df['duration_days'])):
    days = df['duration_days'][i]
    if days > 0 and days <= 30:
        df['time_id'][i] = 0
    if days > 30 and days <= 60:
        df['time_id'][i] = 1
    if days > 60 and  days <= 90:
        df['time_id'][i] = 2
    if days > 90:
        df['time_id'][i] = 3

# make all the pledged with $0 to $1 for easier log_calculation
df['log_goal'] = np.log10(df['goal']+1)  # The +1 is to normalize the zero or negative values
df['log_pledged'] = np.log10(df['pledged']+1)  # The +1 is to normalize the zero or negative values
df['log_backers'] = np.log10(df['backers']+1)  # The +1 is to normalize the zero or negative values
df['launched_year'] = df['launched'].dt.year

# dataframe from Tabletop category
df_boardgame = df[df['sub_category_id'] == 14]


# PLOTS

# Figure 1
sns.set_style("whitegrid")
ax = sns.boxplot(x='outcome', y='log_goal', data=df)
medians = np.log10(df.groupby('outcome')['goal'].median().values)
nobs = df.groupby('outcome')['goal'].median().values
nobs = ['{:.2f}'.format(x) for x in nobs.tolist()]
nobs = ['median: $' + i for i in nobs]
 
# Add it to the plot
pos = range(len(nobs))
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], \
            medians[tick] + 0.05, \
            nobs[tick], \
            horizontalalignment='center', \
            size='small', \
            color='w', \
            weight='semibold')

plt.title('Log10(Goal) by Project Outcome')
plt.ylabel('Log10(Goal) $')
plt.show()

# Figure 2
df['log_backers_succ'] = df['log_backers'][df['outcome'] == 'successful']
df['log_goal_succ'] = df['log_goal'][df['outcome'] == 'successful']

p = sns.jointplot(x='log_goal_succ', y='log_backers_succ', data=df, kind='reg', color='black')
plt.plot([np.log10(15000), np.log10(15000)], [0, 2.29444], 'b', label='Goal: $15000, Backers: 197')
plt.plot([0, np.log10(15000)], [2.29444, 2.29444],'b')
p.fig.suptitle('Backers by Goals for Successful Campaigns')
plt.xlabel('Log10(Goal) $')
plt.ylabel('Log10(Backers)')
plt.legend()
plt.show()

# Figure 3
sns.distplot(df_boardgame['log_backers'])
plt.plot([2.23413,2.23413], [0,0.519652], label='Backers: 171.45')
plt.ylabel('Density')
plt.xlabel('Log10(Backers)')
plt.title('Backers Distribution for Tabletop Games')
plt.legend()
plt.show()

# Figure 4
sns.set_style("whitegrid")
ax = sns.boxplot(x='time_id', y='log_pledged', data=df)
medians = np.log10(df.groupby('time_id')['pledged'].median().values)
nobs = df.groupby('time_id')['pledged'].median().values
nobs = ['{:.2f}'.format(x) for x in nobs.tolist()]
nobs = ['median: $' + i for i in nobs]
pos = range(len(nobs))
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], \
            medians[tick] + 0.05, \
            nobs[tick], \
            horizontalalignment='center', \
            size='small', \
            color='w', \
            weight='semibold')

plt.title('Log10(Pledged) by Campaign Duration')
plt.xticks([0,1,2,3], ['0-1','1-2','2-3','>3'])
plt.xlabel('duration (months)')
plt.ylabel('Log10(Pledged) $')
plt.show()

sns.relplot(data=df, x='duration_days', y='log_pledged')
plt.show()

# Time-series analysis for amount pledged by years in Tabletop Games category
sns.set_style("whitegrid")
sns.boxplot(x='launched_year', y='log_pledged', data=df_boardgame, color='coral')
plt.title('Amount Pledged by Year')
plt.xlabel('Launched Year')
plt.ylabel('Log10(Pledged)')
plt.show()