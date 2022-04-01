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

query = "Select * from campaign;"
df = pd.read_sql(query,mydb)

df.head()


# An Overview of campaigns by outcome
query = 'SELECT outcome, COUNT(*) AS total_campaign, SUM(goal) AS total_goal, SUM(pledged) AS total_pledged, SUM(backers) AS total_backers \
        FROM campaign \
        GROUP BY outcome;'
df = pd.read_sql(query,mydb)
num_campaign = df['total_campaign']
outcomes = df['outcome']
plt.pie(num_campaign, labels=outcomes, autopct='%.0f%%')
plt.show()
plt.clf()

# Box plot of goal by state
query = "SELECT goal, outcome FROM campaign WHERE outcome = 'successful' OR outcome = 'failed' ORDER BY outcome"
df = pd.read_sql(query,mydb)
df['loggoal'] = np.log10(df['goal'])  # take log on the goal

sns.boxplot(x='outcome', y='loggoal', data=df)
plt.title('log(goal) by state')
plt.show()

# find the median goal for each state
df.groupby('outcome')['goal'].median().T

# Box plot of goal by state in board game category
query = "SELECT goal, outcome FROM campaign \
        WHERE (sub_category_id = 14 OR sub_category_id = 70) AND (outcome = 'successful' OR outcome = 'failed') \
        ORDER BY outcome"
df = pd.read_sql(query,mydb)
df['loggoal'] = np.log10(df['goal'])  # take log on the goal

sns.boxplot(x='outcome', y='loggoal', data=df)
plt.title('log(goal) by state')
plt.show()
# find the median goal for each state
df.groupby('outcome')['goal'].median().T


# Let's look at the distribution of backers by state
query = "SELECT backers, outcome FROM campaign WHERE outcome = 'successful' OR outcome = 'failed' ORDER BY outcome"
df = pd.read_sql(query,mydb)
df['logbackers'] = np.log10(df['backers'])  # take log on the backers
sns.boxplot(x='outcome', y='logbackers', data=df)
plt.show()

# Let's look at the distribution of backers by state in board game category
query = "SELECT backers, outcome FROM campaign \
        WHERE (sub_category_id = 14 OR sub_category_id = 70) AND (outcome = 'successful' OR outcome = 'failed') \
        ORDER BY outcome"
df = pd.read_sql(query,mydb)
df['logbackers'] = np.log10(df['backers'])  # take log on the backers
sns.boxplot(x='outcome', y='logbackers', data=df)
plt.show()


# Study the correlation between the number of backers and the goal
query = "SELECT backers, goal, outcome FROM campaign WHERE outcome = 'successful' OR outcome = 'failed' ORDER BY outcome"
df = pd.read_sql(query,mydb)
df['logbackers'] = np.log10(df['backers'])  # take log on the backers
df['loggoal'] = np.log10(df['goal'])  # take log on the goal

sns.relplot(data=df, x='loggoal', y='logbackers', col='outcome')
plt.show()

"""
Seems like there are campaigns with goal set to infinity.
We need to clean that up
"""
df['log_backers_succ'] = df['log_backers'][df['outcome'] == 'successful']
df['log_goal_succ'] = df['log_goal'][df['outcome'] == 'successful']

reg = LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(df['log_goal_succ'].dropna().values.reshape(-1,1), df['log_backers_succ'].dropna().values.reshape(-1,1), test_size=0.2, random_state=0)
reg.fit(X_train, Y_train)

Y_pred = reg.predict(X_test)
plt.plot(X_test, Y_pred, color='blue', linewidth=2)

p = sns.jointplot(x='log_goal_succ', y='log_backers_succ', data=df, kind='reg', color='black')
#plt.plot(X_test, Y_pred, color='blue', linewidth=2)
plt.plot([np.log10(15000), np.log10(15000)], [0, 2.29444], 'b', label='Goal: $15000, Backers: 197')
plt.plot([0, np.log10(15000)], [2.29444, 2.29444],'b')
p.fig.suptitle('Backers by Goals for Successful Campaigns')
plt.xlabel('Log10(Goal) $')
plt.ylabel('Log10(Backers)')
plt.legend()
plt.show()


# Study the correlation between the number of backers and the goal
query = "SELECT backers, goal, outcome FROM campaign \
        WHERE (sub_category_id = 14 OR sub_category_id = 70) AND (outcome = 'successful' OR outcome = 'failed') \
        ORDER BY outcome"
df = pd.read_sql(query,mydb)
df['logbackers'] = np.log10(df['backers'])  # take log on the backers
df['loggoal'] = np.log10(df['goal'])  # take log on the goal

sns.relplot(data=df, x='loggoal', y='logbackers', col='outcome')
plt.show()

df['logbackers_successful'] = df['logbackers'][df['outcome'] == 'successful']
df['loggoal_successful'] = df['loggoal'][df['outcome'] == 'successful']
sns.jointplot(x='loggoal_successful', y='logbackers_successful', data=df, kind='reg')
plt.show()
plt.clf()
p = sns.lmplot(x='loggoal_successful', y='logbackers_successful', data=df)
plt.show()
plt.clf()

# Try linear regression model
reg = LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(df['loggoal_successful'].dropna().values.reshape(-1,1), df['logbackers_successful'].dropna().values.reshape(-1,1), test_size=0.2, random_state=0)
reg.fit(X_train, Y_train)
#To retrieve the intercept:
print(reg.intercept_)

#For retrieving the slope:
print(reg.coef_)

Y_pred = reg.predict(X_test)

#plt.scatter(X_test, Y_test,  color='gray')
sns.lmplot(x='loggoal_successful', y='logbackers_successful', data=df)
plt.plot(X_test, Y_pred, color='blue', linewidth=2)
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

def goal2backers (goal, c, m):
    backers = c + m*np.log10(goal)
    return np.power(10,backers)


print(goal2backers(15000, reg.intercept_[0], reg.coef_[0][0]))

# Check for NaN in dataset
query = "SELECT * FROM campaign"
df = pd.read_sql(query,mydb)
df.isnull().sum().sum()
df.max()

df[df['name'] == '100 Risings']
df[df['currency_id'] == 1].sort_values('name')

"""
---------------------------------------------------------------------
"""


# initial data cleaning
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

# study how does pledged amount changed with campaign duration
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

# median campaign goals by state
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

# TimeSeries Analysis
df['launched_year'] = df['launched'].dt.year
sns.boxplot(x='launched_year', y='log_pledged', data=df, color='coral')
plt.title('Campaign Pledged by Year')
plt.xlabel('Launched Year')
plt.ylabel('Log10(Pledged) $')
plt.show()

sns.boxplot(x='launched_year', y='log_backers', data=df, color='coral')
plt.title('Campaign Backers by Year')
plt.xlabel('Launched Year')
plt.ylabel('Log10(Backers)')
plt.show()


# median campaign goals by state in board game category
df_boardgame = df[df['sub_category_id'] == 14]
sns.boxplot(x='outcome', y='log_goal', data=df_boardgame)
plt.title('log(goal) by state')
plt.show()

df_boardgame.groupby('outcome')['goal'].median().T

# It is better to study the success_rate according to campaign goals. I can create bins of goals 
#sns.distplot(df, x='log_goal', y='outcome')
#plt.show()

"""
df_successful = df[df['outcome'] == 'successful']
binwidth = 0.5
n, bins, patches = plt.hist(df_successful['goal'], bins=np.arange(,8+binwidth,binwidth), facecolor='blue', alpha=0.5)
plt.show()
plt.plot(np.arange(0,7.5+binwidth,binwidth), n/df.sum()['id'])
plt.show()
"""

sns.distplot(df_boardgame['backers'], norm_hist=False)
plt.show()

sns.distplot(df_boardgame['log_goal'], norm_hist=False)
plt.show()

sns.distplot(df['log_goal'], norm_hist=False)
plt.show()

# relationship between goal and pledged amount in the whole dataset
sns.jointplot(x='log_goal', y='log_pledged', data=df, kind='reg')
plt.show()

# relationship between goal and pledged amount in the tabletop games category
sns.jointplot(x='log_goal', y='log_pledged', data=df_boardgame, kind='reg')
plt.show()

# relationship between goal and pledged amount for successful campaigns in the whole dataset
sns.jointplot(x='log_goal', y='log_pledged', data=df[df['outcome'] == 'successful'], kind='reg')
plt.show()

# relationship between goal and pledged amount for successful campaigns in the whole dataset
sns.jointplot(x='log_goal', y='log_pledged', data=df[df['outcome'] == 'failed'], kind='reg')
plt.show()

# filter campaign with pledged amount at least $15,000
sns.jointplot(x='log_goal', y='log_pledged', data=df[df['pledged'] >= 15000], kind='reg')
plt.show()

"""
--- Focus on studying the trend of backers within the board game category
"""
sns.distplot(df_boardgame['log_backers'])
plt.plot([2.23413,2.23413], [0,0.519652], label='Backers: 171.45')
plt.ylabel('Density')
plt.xlabel('Log10(Backers)')
plt.title('Backers Distribution for Tabletop Games')
plt.legend()
plt.show()

sns.countplot(x='log_backers', data=df_boardgame)
plt.show()

sns.boxplot(x='launched_year', y='log_backers', data=df_boardgame, color='coral')
plt.title('Campaign Backers by Year')
plt.xlabel('Launched Year')
plt.ylabel('Log10(Backers)')
plt.show()

sns.set_style("whitegrid")
sns.boxplot(x='launched_year', y='log_pledged', data=df_boardgame, color='coral')
plt.title('Amount Pledged by Year for Tabletop Games')
plt.xlabel('Launched Year')
plt.ylabel('Log10(Pledged)')
plt.show()

sns.boxplot(x='launched_year', y='log_pledged', data=df_boardgame, color='coral')
plt.title('Amount Pledged by Year')
plt.xlabel('Launched Year')
plt.ylabel('Log10(Pledged)')
plt.show()