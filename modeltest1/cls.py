import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import pickle

# SVC dung cho phan loai classifier

data = pd.read_csv('diabetes.csv')
# print(data)
# print(data.head(10))
# plt.figure(figsize=(8, 8))
# plt.show()
# sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
# plt.savefig("correlation.png")

target = "Outcome"
print(data[target].value_counts())
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#tien xu ly du lieu
# 1 cot.
scaler = StandardScaler()
result = scaler.fit_transform(data[["Pregnancies"]])
# for i, j in zip(data[["Pregnancies"]].values, result):
#     print("before {}, after{}".format(i, j))

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# cls_rfc = RandomForestClassifier(n_estimators=300, criterion="entropy") #thu cong uoc tinh parameters
# cls = SVC()
param_grid = {
    "n_estimators": [50, 100, 200],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 2, 5, 10],
    "min_samples_leaf": [1, 2, 5]
}
cls = GridSearchCV(RandomForestClassifier(random_state=263), param_grid=param_grid, scoring="accuracy", verbose=2, cv=6, n_jobs=6)
# goi model co trong folder cls
# cls = pickle.load(open("classifier.pkl",'rb'))
cls.fit(x_train, y_train)
pickle.dump(cls, open("classifier.pkl", "wb"))
y_predict = cls.predict(x_test)
# for i, j in zip(y_predict, y_test):
#     print("Predict: {}, Actual: {}".format(i, j))
print(cls.best_score_)
print(cls.best_params_)
print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))