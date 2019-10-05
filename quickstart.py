# Tensorflow 2.0 Quick Start

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
assert(tf.__version__ == "2.0.0")

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# Load everyone's favorite dataset
mnist = tf.keras.datasets.mnist
(xtr, ytr), (xte, yte) = mnist.load_data()
xtr, xte = xtr / 255., xte / 255.

print(xtr.shape, xte.shape)

# Add channels dimension
xtr = xtr[..., tf.newaxis]
xte = xte[..., tf.newaxis]

# Set up dataset parameters
BATCH_SIZE = 32
NUM_EPOCHS = 5

# Random sample from the first x samples using shuffle(x)
# Batch the dataset to add an outer dimension
train_set = tf.data.Dataset.from_tensor_slices((xtr, ytr))
train_set = train_set.shuffle(10000)
train_set = train_set.batch(BATCH_SIZE)
test_set = tf.data.Dataset.from_tensor_slices((xte, yte)).batch(BATCH_SIZE)

# Specify the model
class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2D(BATCH_SIZE, 3, activation='relu')
        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(10, activation='softmax')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = CNN()
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Metrics accumulation, this is pretty useful.
train_loss = tf.keras.metrics.Mean(name='train_loss') # Weighted mean
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')

# When you annotate a function with tf.function, you can still call it like any other function.
# But it will be compiled into a graph, which means you get the benefits of faster execution,
# running on GPU or TPU, or exporting to SavedModel.
@tf.function
def train_step(images, labels):
    # GradientTape records operations for automatic differentiation.
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_function(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    loss = loss_function(labels, predictions)
    test_loss(loss)
    test_accuracy(labels, predictions)

def reset_all_states(*args):
    for item in args:
        item.reset_states()

# Execute the training and test loops.
# Similar to PyTorch, where the Dataset class implements an iterator.
for epoch in range(NUM_EPOCHS):
    for train_images, train_labels in train_set:
        train_step(train_images, train_labels)
    for test_images, test_labels in test_set:
        test_step(test_images, test_labels)
    
    template = 'Epoch: {} | Loss: {} | Accuracy: {} | Test Loss: {} | Test Accuracy: {}'
    print(template.format(epoch + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))
    
    reset_all_states(train_loss, train_accuracy, test_loss, test_accuracy)
