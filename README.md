# Kaggle
import numpy as np
import pandas as pd
import matplotlib as plt


train_set = pd.read_csv(r'C:\Users\xycmj\Desktop\kaggle\train.csv')
test_set = pd.read_csv(r'C:\Users\xycmj\Desktop\kaggle\test.csv')
pd.set_option('display.max_columns',12)
# print(train_set.head())
train_drop = train_set.drop(['PassengerId','Survived'],axis=1)
# print(train_drop.head())

#提取名字中的称号属性
test_set['Title'] = test_set.Name.apply(lambda name:name.split(',')[1].split('.')[0].strip())
# print(test_set['Title'])
# print(test_set.groupby(test_set.Age.isnull()).Title.value_counts())
#提取Train集中的称号属性
train_drop['Title'] = train_drop.Name.apply(lambda name:name.split(',')[1].split('.')[0].strip())
# print(train_drop.groupby(train_drop.Age.isnull()).Title.value_counts())

print(train_drop['Age'].groupby([train_drop['Sex'],train_drop['Title']]).median())
median = train_drop['Age'].groupby([train_drop['Sex'],train_drop['Title']]).median()
print(median['female'])
print(train_drop.info())
def new_age(col):
    age = col[0]
    sex = col[1]
    title = col[2]
    if pd.isnull(age):
        return median[sex][title]
    return age
train_drop.Age = train_drop[['Age','Sex','Title']].apply(new_age, axis=1)
# print(train_drop.info())
# print(train_drop.loc[:7])
train_drop.Cabin = train_drop.Cabin.fillna('U')
print(train_drop[:10])
most_embarked = train_drop.Embarked.value_counts().index[0]
train_drop.Embarked=train_drop.Embarked.fillna(most_embarked)
print(train_drop.info())
