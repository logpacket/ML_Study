import tensorflow as tf

x_data = [[1., 2.], [2., 3.], [3., 1.], [4., 3.], [5., 3.], [6., 2.]]
y_data = [[0.], [0.], [0.], [1.], [1.], [1.]]
constant = [[1.], [1.], [1.], [1.], [1.], [1.]]

W = tf.Variable(initial_value=tf.random_normal_initializer()(shape=[2, 1], dtype=tf.float32), name="weight")
b = tf.Variable(initial_value=tf.random_normal_initializer()(shape=[1], dtype=tf.float32), name="bias")

hypothesis = lambda X: tf.sigmoid(tf.matmul(X, W)+b)
cost = lambda: -tf.reduce_mean(y_data * tf.math.log(hypothesis(x_data)) + (tf.math.subtract(constant, y_data) * tf.math.log(1 - hypothesis(x_data))))
optimizer = tf.optimizers.SGD(learning_rate=0.01)

predicted = lambda X: tf.cast(hypothesis(X) > 0.5, dtype=tf.float32)
accuracy = lambda X, Y: tf.reduce_mean(tf.cast(tf.equal(predicted(X), Y), dtype=tf.float32))

for step in range(10001):
    cost_val = cost().numpy()
    optimizer.minimize(cost, var_list=[W, b])
    if step % 200 == 0:
        print(step, cost_val)

h = hypothesis(x_data)
c = predicted(x_data)
a = accuracy(x_data, y_data)
print("\nHypothesis:\n", h.numpy(), "\nCorrect:\n", c.numpy(), "\nAccuracy:\n", a.numpy())