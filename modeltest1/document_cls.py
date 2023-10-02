import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def my_func(loc):
    if loc == "New York":
        return "NY"
    result = re.findall("\s[A-Z]{2}$", loc)
    if len(result) > 0:
        return result[0][1:]
    else:
        return loc


data = pd.read_excel("/home/nevergiveup/PycharmProjects/deeplearning/machinelearning/modeltest1/datasets/final_project.ods", engine="odf", dtype=str)
# print(data["career_level"].value_counts())
data.dropna(axis=0, inplace=True)
data['location'] = data['location'].apply(my_func)
# print(len(data['location'].unique()))

target = "career_level"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
# print(data["industry"].value_counts())
# stratify
# print(y_train.value_counts())
# print("-----------------------")
# print(y_test.value_counts())
# vectorizer = TfidfVectorizer(decode_error="ignore", ngram_range=(1, 2))
# result = vectorizer.fit_transform(x_train["description"])
# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# print(result.shape)
# print(result)

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words=["you", "english"]), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=0.005, max_df=0.995), "description"),
    ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry", TfidfVectorizer(stop_words="english"), "industry")
])

cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier())
])

cls.fit(x_train, y_train)
y_predict = cls.predict(x_test)

print(classification_report(y_test, y_predict))