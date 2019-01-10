import trainingData
import numpy as np

batch_size = 5
for i in range(0, 100):
    X, Y = trainingData.next_batch(batch_size)
    print(np.shape(X), np.shape(Y))