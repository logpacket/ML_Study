import tensorflow as tf

x_train = [[1, 2, 1],
           [1, 3, 2],
           [1, 3, 4],
           [1, 5, 5],
           [1, 7, 5],
           [1, 2, 5],
           [1, 6, 6],
           [1, 7, 7]]

y_train = [[0, 0, 1],
           [0, 0, 1],
           [0, 0, 1],
           [0, 1, 0],
           [0, 1, 0],
           [0, 1, 0],
           [1, 0, 0],
           [1, 0, 0]]

x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))

W = tf.Variable(tf.random.normal((3, 3)))
b = tf.Variable(tf.random.normal((3,)))


def softmax_fn(features):
    hypothesis = tf.nn.softmax(tf.matmul(features, W) + b)
    return hypothesis


def loss_fn(hypothesis, labels):
    cost = tf.reduce_mean(-tf.reduce_sum(labels * tf.math.log(hypothesis), axis=1))
    return cost


is_Decay = True
starter_learning_rate = 0.1

if (is_Decay):
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=starter_learning_rate,
                                                                   decay_steps=1000,
                                                                   decay_rate=0.96,
                                                                   staircase=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate)
else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=starter_learning_rate)


def accuracy_fn(hypothesis, labels):
    prediction = tf.argmax(hypothesis, 1)
    is_correct = tf.equal(prediction, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    return accuracy


for step in range(1001):
    for features, labels in iter(dataset):
        features = tf.cast(features, tf.float32)
        labels = tf.cast(labels, tf.float32)
        optimizer.minimize(lambda: loss_fn(softmax_fn(features), labels), var_list=[W, b])
        if step % 100 == 0:
            print("iter {} loss{:4f}".format(step, loss_fn(softmax_fn(features), labels)))

x_test = tf.cast(x_test, tf.float32)
y_test = tf.cast(y_test, tf.float32)
test_acc = accuracy_fn(softmax_fn(x_test), y_test)
print("Testset Accuracy:{:.4f}".format(test_acc))
