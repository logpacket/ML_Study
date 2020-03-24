import tensorflow as tf
import numpy as np

xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_train = xy[:, 0:-1]
y_train = xy[:, [-1]]

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))

W = tf.Variable(tf.random.normal((4, 1)), dtype=tf.float32)
b = tf.Variable(tf.random.normal((1,)), dtype=tf.float32)


def linearReg_fn(features):
    hypothesis = tf.matmul(features, W) + b
    return hypothesis


def loss_fn(hypthesis, labels):
    cost = tf.reduce_mean(tf.square(hypthesis - labels))
    return cost


optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5)


for step in range(101):
    for features, labels in iter(dataset):
        features = tf.cast(features, tf.float32)
        labels = tf.cast(labels, tf.float32)
        hypo_value = linearReg_fn(features)
        loss_value = loss_fn(linearReg_fn(features), labels)
        optimizer.minimize(lambda: loss_fn(linearReg_fn(features), labels), var_list=[W, b])
    print("iter : {}, Loss:{:.4f}, Prediction:{}".format(step, loss_value, hypo_value))