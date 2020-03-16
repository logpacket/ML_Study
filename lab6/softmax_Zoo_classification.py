import tensorflow as tf
import numpy as np

xy = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, -1]

nb_classes = 7
Y_one_hot = tf.one_hot(y_data.astype(np.int32), nb_classes)

W = tf.Variable(tf.random.normal((16, nb_classes)), name='weight')
b = tf.Variable(tf.random.normal((nb_classes,)), name='bias')
variables = [W, b]


def logit_fn(X):
    return tf.matmul(X, W) + b


def hypothesis(X):
    return tf.nn.softmax(logit_fn(X))


def cost_fn(X, Y):
    logits = logit_fn(X)
    cost_i = tf.keras.losses.categorical_crossentropy(y_true=Y, y_pred=logits, from_logits=True)
    cost = tf.reduce_mean(cost_i)
    return cost


def prediction(X, Y):
    pred = tf.argmax(hypothesis(X), 1)
    correct_prediction = tf.equal(pred, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


optimizer = tf.optimizers.SGD(learning_rate=0.1)

for step in range(1001):
    optimizer.minimize(lambda: cost_fn(x_data, Y_one_hot), var_list=variables)
    if step % 100 == 0:
        print(step, cost_fn(x_data, Y_one_hot).numpy(), prediction(x_data, Y_one_hot).numpy())
