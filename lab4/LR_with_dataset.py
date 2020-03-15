import tensorflow as tf

x_record = [0., 0., 0.]
y_record = [0.]

map_func = lambda *x: tf.stack(x)
x_dataset = tf.data.experimental.CsvDataset("./lab4/data.csv", x_record, header=True, select_cols=[0, 1, 2]).map(map_func).batch(6)
y_dataset = tf.data.experimental.CsvDataset("./lab4/data.csv", y_record, header=True, select_cols=[3]).map(map_func).batch(6)

W = tf.Variable(initial_value=tf.random_normal_initializer()(shape=[3, 1], dtype=tf.float32), name="weight")
b = tf.Variable(initial_value=tf.random_normal_initializer()(shape=[1], dtype=tf.float32), name="bias")

hypothesis = lambda X, W: tf.matmul(X, W) + b

cost = lambda: tf.reduce_mean(tf.square(hypothesis(x_data, W) - y_data))

optimzer = tf.optimizers.SGD(learning_rate=1e-5)

for x_data, y_data in zip(x_dataset, y_dataset):
    for step in range(4001):
        cost_val = cost()
        hy_val = hypothesis(x_data, W)
        optimzer.minimize(cost, var_list=[W, b])
        if step % 20 == 0:
            print(step, "Cost:", cost_val.numpy(), "\nPrediction:\n", hy_val.numpy())
