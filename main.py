import numpy as np
import pandas
from model import Hebbian
import matplotlib.pyplot as plt

train_data = pandas.read_csv('mnist_train.csv', header=None)
test_data = pandas.read_csv('mnist_test.csv', header=None)
id_train = train_data.loc[:,0].values
id_test = test_data.loc[:,0].values
pure_train_data = train_data.drop(0, axis=1).values / 255
pure_test_data = test_data.drop(0, axis=1).values / 255

data_size, input_size = pure_train_data.shape

max_number_seen = 9
cats = max_number_seen+1
LGN_size = 100
param = 5
lr = 0.05
max_activated = 3
L4_size = max_activated*cats
biases = np.asarray([max_activated*1./cats]*L4_size)
model = Hebbian(cats, input_size, LGN_size, L4_size, param, lr, max_activated, biases, True)
epochs = 1

model.train(pure_train_data, epochs, id_train)
accuracy = model.test(pure_test_data, id_test)

print(accuracy)