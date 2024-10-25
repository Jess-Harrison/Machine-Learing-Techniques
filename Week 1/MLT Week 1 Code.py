import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('./titanic.csv')
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()
data = data.dropna()
print(data)
sumry = data.describe()

# survied passanger
survived = data.loc[data['Survived'] == 1]
totalsurvived = survived['Survived'].count()
percentage = totalsurvived / data['Survived'].count() * 100
print('percentage', percentage)

sns.set_style('whitegrid')
sns.countplot(x='Survived', data=data)
plt.show()

# average age
age = data['Age']
ave_age = np.mean(age)

# survivors by gender
survived_gender = pd.DataFrame(survived['Sex'])
survived_gender_count = survived_gender.value_counts()
# res =locals().update(survived_gender_count)
print(survived_gender_count['male'])

male = survived_gender_count['male']
female = survived_gender_count['female']
print(male)
print(female)

# Age distribution
sur_male = survived.loc[survived['Sex'] == 'male']
avg_sur_male_age = np.mean(sur_male['Age'])
print(avg_sur_male_age)
sur_female = survived.loc[survived['Sex'] == 'female']
avg_sur_female_age = np.mean(sur_female['Age'])
ds = np.log1p(survived['Age'])
sns.histplot(data=survived, x=ds, kde=True)
plt.show()

# class for non-surviver
non_sur = data.loc[data['Survived'] == 0]
non_sur_class = non_sur['Pclass']
res1 = dict(Counter(list(zip(non_sur_class))))
print()
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
a = res1.keys().ToArray()
b = res1.values().ToArray()
print()

ax.bar(a, b)
plt.show()

# embarkment
survior_embarkment = survived['Embarked']

survior_embarkment = dict(survior_embarkment)
survior_embarkment.values()
survior_embarkment.keys()
# print(survior_embarkment.count())

# sibling/spouse for survived
sur_sibsp = survived['SibSp']
sur_sibsp = dict(sur_sibsp)
sur_sibsp.keys()
sur_sibsp.values()

# sibling/spouse relation to class
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(sur_sibsp, survived['Pclass'])
plt.show()

# parents/childern relation to survival
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(sur_sibsp, survived['Survived'])
plt.show()
