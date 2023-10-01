# data1 = [-10, -21, 25, 46, 33, -23, -25, 1, 2, -1, -5]
#
# pos_count = sum([1 for i in data1 if i > 0])
# neg_count = sum([1 for i in data1 if i < 0])
#
# print(pos_count, neg_count)


data2 = [4, 6, 4, 3, 3, 4, 3, 4, 3, 8]
data2_dict = {key: data2.count(key) for key in data2}
k = 3

for key, value in data2_dict.items():
    if value > k:
        print(key)