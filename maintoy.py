import numpy as np
import pandas
from toymodel import Hebbian
import matplotlib.pyplot as plt

data_size = 1000
cats = 2
pure_train_data = np.random.randint(cats, size=data_size)
pure_test_data = np.random.randint(cats, size=1000)
id_train = pure_train_data
id_test = pure_test_data

LGN_size = 20
param = 2.5
lr = 0.1
max_activated = 5
L4_size = max_activated*cats
model = Hebbian(cats, LGN_size, L4_size, param, lr, max_activated, False)
epochs = 1

model.train(pure_train_data, epochs, id_train)
accuracy = model.test(pure_test_data, id_test)

print(accuracy)