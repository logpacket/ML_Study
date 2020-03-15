import tensorflow as tf

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

W = tf.Variable(initial_value=tf.random_normal_initializer()(shape=[3, 1], dtype=tf.float32), name="weight")
b = tf.Variable(initial_value=tf.random_normal_initializer()(shape=[1], dtype=tf.float32), name="bias")

hypothesis = lambda X, W: tf.matmul(X, W) + b

cost = lambda: tf.reduce_mean(tf.square(hypothesis(x_data, W) - y_data))

optimzer = tf.optimizers.SGD(learning_rate=1e-5)

for step in range(2001):
    cost_val = cost()
    hy_val = hypothesis(x_data, W)
    optimzer.minimize(cost, var_list=[W, b])
    if step % 20 == 0:
        print(step, "Cost:", cost_val.numpy(), "\nPrediction:\n", hy_val.numpy())
