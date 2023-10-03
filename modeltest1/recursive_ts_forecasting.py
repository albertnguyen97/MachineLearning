import pandas as pd
import matplotlib.pyplot as plt


def create_data(input_data, input_window_size):
    i = 1
    while i < input_window_size:
        input_data["co2_{}".format(i)] = input_data["co2"].shift(-i)
        i += 1
    input_data["target"] = input_data["co2"].shift(-i)
    input_data = input_data.drop("time")
    return input_data


data = pd.read_csv("datasets/co2.csv")
print(data.info())
data["time"] = pd.to_datetime(data["time"])
print(data["time"])
data["co2"] = data["co2"].interpolate(inplace=True)  # noi suy for change nan value
# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# print(data.info())
# ax.set_xlabel("Time")
# ax.set_ylabel("Co2")
# plt.show()
window_size = 5
data = create_data(data, window_size)