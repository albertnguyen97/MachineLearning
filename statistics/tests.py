import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sn

data = pd.read_csv('diabetes.csv')
result = data.describe()
count_class = data.groupby('Age').size()
print(count_class)

scaler = RobustScaler()
before = data[['Pregnancies']].values
after = scaler.fit_transform(data[['Pregnancies']])
for b, a in zip(before, after):
    print("before: {}, after: {}".format(b, a))

correlations = data.corr()
sn.heatmap(correlations, annot=True)
scatter_matrix(data)

data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
plt.show()