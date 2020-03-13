import tensorflow as tf

x_train = [1., 2., 3., 4., 5., 6.]
y_train = [3.2, 6.3, 9.6, 12.2, 15.3, 18.1]

w_init = tf.random_normal_initializer()  # 랜덤값 생성기 1
b_init = tf.random_normal_initializer()  # 랜덤값 생성기 2
W = tf.Variable(initial_value=w_init(shape=[1], dtype=tf.dtypes.float32))  # 변수 생성( 초기값 shape=[1]인 실수 형 변수 )
b = tf.Variable(initial_value=b_init(shape=[1], dtype=tf.dtypes.float32))

hypothesis = lambda: x_train * W + b  # 람다 함수로 hypothesis 함수 생성
cost = lambda: tf.reduce_mean(tf.square(hypothesis() - y_train))  # 람다 함수로 cost 함수 생성

opt = tf.optimizers.SGD(learning_rate=0.01)  # optimizer 생성 ( Stochastic Gradient Descent 확률적 경사 하강 )

# learning rate : 각 반복에서의 단계 크기 일반적으로 0.01
for step in range(2001):
    opt.minimize(cost, var_list=[W, b])  # cost 값 최소화
    if step % 20 == 0:
        print(step, cost().numpy(), W.numpy(), b.numpy())  # @.numpy() == Tensor가 가지고 있는 실수 값 출력
