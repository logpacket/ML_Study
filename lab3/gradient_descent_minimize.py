import tensorflow as tf

X_data = [1., 2., 3.]
Y_data = [1., 2., 3.]

init = tf.random_normal_initializer()
W = tf.Variable(initial_value=init(shape=[1], dtype= tf.dtypes.float32), name= "weight")

hypothesis = lambda W, X: W * X
cost = lambda W, X, Y: tf.reduce_sum(tf.square(hypothesis(W, X) - Y))

learning_rate = 0.1
gradient = lambda W, X, Y: tf.reduce_mean((W * X - Y) * X) * 2
descent = lambda W, X, Y: W - learning_rate * gradient(W, X, Y)
update = lambda W, X, Y: W.assign(descent(W, X, Y))

for step in range(21):
    update(W, X_data, Y_data)
    print(step, cost(W, X_data, Y_data).numpy(), W.numpy())
