import tensorflow as tf
import numpy as np
import csv
import cv2

learning_rate = 0.001
epochs = 2
batch_size = 1000
display_freq = 100
img_h = img_w = 28
img_size_flat = img_h * img_w
n_classes = 4
h1 = 200

images = []
labels = []
dataset = np.load("dataset.npy")

for data in dataset:
    images.append(data['image'])
    labels.append(data['label'])
images = np.array(images)
labels = np.array(labels)

def weight_variable(name, shape):
    initializer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initializer)


def bias_variable(name, shape):
    initializer = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b' + name,
                           dtype=tf.float32,
                           initializer=initializer)


def fc_layer(x, num_nodes, name, use_relu=True):
    in_dim = x.get_shape()[1]
    W = weight_variable(name, shape=[in_dim, num_nodes])
    b = bias_variable(name, [num_nodes])
    layer = tf.matmul(x, W)
    layer += b
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


X = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')

y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y')

fc1 = fc_layer(X, h1, 'FC1', use_relu=True)

output_logits = fc_layer(fc1, n_classes, 'OUT', use_relu=False)

# Define the loss function, optimizer, and accuracy

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y, logits=output_logits), name='loss')

optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate, name='Adam-op').minimize(loss)

correct_prediction = tf.equal(
    tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')

accuracy = tf.reduce_mean(
    tf.cast(correct_prediction, tf.float32), name='accuracy')

cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

num_tr_iter = int(images.shape[0] / batch_size)
for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch+1))
    curr = 0
    final = batch_size
    for iteration in range(num_tr_iter):
        batch_x, batch_y = images[curr:final], labels[curr:final]
        curr += batch_size
        final += batch_size
        feed_dict_batch = {X: batch_x, y: batch_y}
        sess.run(optimizer, feed_dict=feed_dict_batch)

        if iteration % display_freq == 0:
            loss_batch, acc_batch = sess.run([loss, accuracy],
                                            feed_dict=feed_dict_batch)
            print("Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                format(iteration, loss_batch, acc_batch))

# Testing

test_image = cv2.imread('+.png')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image = 255 - test_image
test_image = cv2.resize(test_image, (28,28))
print(test_image.shape)
# test_image = test_image.flatten()

feed_dict_test = {X: test_image.reshape(1,-1), y: [[1,0,0,0]]}
loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
print('---------------------------------------------------------')
print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
print('---------------------------------------------------------')

sess.close()