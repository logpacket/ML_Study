import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W_init = tf.random_normal_initializer()
W = tf.Variable(initial_value=W_init(shape=[1], dtype=tf.dtypes.float32))

hypothesis = lambda W: X * W

cost = lambda W: tf.reduce_mean(tf.square(hypothesis(W) - Y))

W_val = []
cost_val = []

for i in range(-30, 50):
    W.assign([i * 0.1])  # 0.1 = learning rate
    curr_cost = cost(W)
    curr_W = W.numpy()
    W_val.append(curr_W)
    cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.show()
