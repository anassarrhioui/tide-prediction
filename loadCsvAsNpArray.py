import csv
import numpy as np

with open("Achill_Island_MODELLED-1-2017.csv", 'r') as f:
    data = list(csv.reader(f))

data = np.array(data, dtype=np.float)

# x_train
print(data[:, 0])
# y_train
print(data[:,1])