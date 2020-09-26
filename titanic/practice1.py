"""
性別、乗り込み地点、料金、年齢から学習
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# データの読み込み
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
gender_submission = pd.read_csv('data/gender_submission.csv')

# データの確認（提出用データの形状、学習データ、テストデータ）
gender_submission.head()
train.head()
test.head()
data = pd.concat([train, test], sort=False)
data.tail()
print(len(train), len(test), len(data))
# 欠損値の確認
data.isnull().sum()

# 特徴量エンジニアリング
# 性別：male->0, female->1
data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
# 乗り込み地点：欠損値をSで補間、S->0, C->1, Q->2
data['Embarked'].fillna('S', inplace=True)
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
# 料金：欠損値を平均値で補間
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
# 年齢：欠損値を平均値±標準偏差の間のランダム値で補間
age_avg = data['Age'].mean()
age_std = data['Age'].std()
data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)

# 学習に使用しないカラムを除去
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)

# 学習データとテストデータに分割
train = data[:len(train)]
test = data[len(train):]
y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)

# 機械学習
clf = LogisticRegression(penalty='l2', solver='sag', random_state=0)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

# 提出用ファイル作成
sub = pd.read_csv('data/gender_submission.csv')
sub['Survived'] = list(map(int, y_predict))
sub.to_csv('data/submission.csv', index=False)
