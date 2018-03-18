import mxnet as mx
import numpy as np
import logging


'''
Implementation of https://medium.com/@julsimon/an-introduction-to-the-mxnet-api-part-3-1803112ba3a8
'''
logging.basicConfig(level=logging.INFO)

n_features = 100
n_samples = 1000

n_training_samples = 800
n_categories = 10

# Data generation
X = mx.nd.uniform(low=0, high=1, shape=(n_samples, n_features))
Y = mx.nd.empty(shape=(n_samples))

for i in range(n_samples - 1):
    Y[i] = np.random.randint(low=0, high=n_categories)

# Data splitting
X_train = mx.nd.crop(data=X, begin=(0,0), end=(n_training_samples, n_features - 1))
Y_train = mx.nd.crop(data=Y, begin=0, end=n_training_samples)

X_val = mx.nd.crop(data=X, begin=(n_training_samples,0), end=(n_samples, n_features - 1))
Y_val = mx.nd.crop(data=Y, begin=n_training_samples, end=n_samples)

# Network
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=64)
activation_1 = mx.symbol.Activation(fc1, name='activation_1', act_type='relu')
fc2 = mx.symbol.FullyConnected(activation_1, name='fc2', num_hidden=n_categories)
out = mx.symbol.Softmax(fc2, name='softmax')
mod = mx.mod.Module(out)

# Iterator
batch = 100
train_iterator = mx.io.NDArrayIter(data=X_train, label=Y_train, batch_size=batch)

# Provide data
mod.bind(data_shapes=train_iterator.provide_data, label_shapes=train_iterator.provide_label)
mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
mod.fit(train_data=train_iterator, num_epoch=50)

# Validation
validate_iterator = mx.io.NDArrayIter(data=X_val, label=Y_val, batch_size=batch)

total_correct = 0
for preds, i_batch, batch in mod.iter_predict(validate_iterator):
    predictions = preds[0].asnumpy().argmax(axis=1)
    label = batch.label[0].asnumpy().astype(int)
    correct_predictions = np.sum(label==predictions)
    total_correct += correct_predictions

print(total_correct)
