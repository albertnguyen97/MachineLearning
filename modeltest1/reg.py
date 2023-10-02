import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lazypredict.Supervised import LazyRegressor

data = pd.read_csv("datasets/StudentScore.xls")
print(data.info())
target = "math score"
x = data.drop(target, axis=1)
y = data[target]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(data['gender'].unique())
print(data['race/ethnicity'].unique())

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),  # change missing data = mean
    ("scaler", StandardScaler())  # chuan hoa du lieu
])

education_levels = ['high school', 'some high school', "some college", "associate's degree", "bachelor's degree",
                    "master's degree"]  # ds ma hoa sap xep order
gender_values = ["female", "male"]
lunch_values = data["lunch"].unique()
test_values = data["test preparation course"].unique()


order_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_levels, gender_values, lunch_values, test_values])) # chuan hoa du lieu ordinal
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False))

])

# show single transformer
# result = num_transformer.fit_transform(x_train[["reading score", "writing score"]])
# for i, j in zip(x_train[["reading score", "writing score"]].values, result):
#     print("Before: {}, After: {}".format(i, j))
#
# result = order_transformer.fit_transform(x_train[["parental level of education"]])
# for i, j in zip(x_train[["parental level of education"]].values, result):
#     print("before {}, after {}".format(i, j))
#
# result = nom_transformer.fit_transform(x_train[["race/ethnicity"]])
# for i, j in zip(x_train[["parental level of education"]].values, result):
#     print("before {}, after {}".format(i, j))
# ------------------

preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("ord_features", order_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_features", nom_transformer, ["race/ethnicity"]),
])

# train model
# reg = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("regressor", RandomForestRegressor())
# ])
# param_grid = {
#     "regressor__n_estimators": [50, 100, 200],
#     "regressor__criterion": ["squared_error", "absolute_error", "poisson"],
#     "preprocessor__num_features__imputer__strategy": ["mean", "median"]
#     # "max_depth": [None, 2, 5, 10],
#     # "min_samples_leaf": [1, 2, 5]
# }
# reg_cv = GridSearchCV(reg, param_grid=param_grid, scoring="r2", verbose=2, cv=6)
#
# reg_cv.fit(x_train, y_train)
# y_predict = reg_cv.predict(x_test)
# # for i, j in zip(y_test, y_predict):
# #     print("Actual {}, Predict {}".format(i, j))
# print(reg_cv.best_score_)
# print(reg_cv.best_params_)
#
# print("R2: {}".format(r2_score(y_test, y_predict)))
# print("R2: {}".format(mean_absolute_error(y_test, y_predict)))
# print("R2: {}".format(mean_squared_error(y_test, y_predict)))
#
# sample = pd.DataFrame([["male", "group C", "some college", "standard", "completed", 80, 78]],
#                       columns=['gender', "race/ethnicity", "parental level of education", "lunch",
#                       "test preparation course", "reading score", "writing score"])
#
# print(reg_cv.predict(sample))
# tien xu ly cho lazy classifier
reg_pre = Pipeline(steps=[
    ("preprocessor", preprocessor)
])
x_train = reg_pre.fit_transform(x_train)
x_test = reg_pre.transform(x_test)

reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(x_train, x_test, y_train, y_test)
print(models)