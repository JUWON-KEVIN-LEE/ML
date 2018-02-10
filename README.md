# MachineLearning

<br>

#### 0. Purpose

: 인식된 필기를 보고 파일 형태로 변환시켜주는 앱을 만들고 싶어서 

<br>

#### [Tensorflow Code Here](https://github.com/JUWON-KEVIN-LEE/ML/tree/master/tensorflow)

<br>

### 1. Description

#### 1.1 Various Descriptions

"데이터를 이용해서 명시적으로 정의되지 않은 패턴을 컴퓨터로 학습하여 결과를 만들어내는 학문 분야"  

"Field of study that gives computers the ability to learn without being explicitly programmed." 

"A computer program is said to learn from experience E with respect to some tast T and some performance measure P,  if its performance on T, as measured by P, improves with experience E."  

<br>

Example ) Email Spam Classification Program

T : classifying emails as spam or not spam.

E : Watching you label emails as spam or not spam.

P : The number (or fraction) of emails correctly classified as spam/not spam.

<br>

#### 1.2 머신러닝 변천사

1950년대 **규칙 기반** 고전적 인공지능 시대

: 앨런 튜링, 인공지능을 판별하는 튜링 테스트.

<br>

1957년 이후 **신경망 기반** 신경망 시대

: perceptron 이라는 기초적인 신경망이 개발되었으나, 데이터가 한정적이라 성능이 별로.

기초 이론 부족으로 한정적인 패턴만 학습이 가능했다.

<br>

1990년대 이후 **통계 기반** 머신러닝 시대 / 빅데이터 시대

: 통계학을 접목시켜 대규모 데이터에서 패턴을 찾는 시도가 성과를 낸다.

통계학적 머신러닝은 웹에서 쏟아지는 데이터, 대용량 저장장치, 분산 처리 기술과 결합하여 엄청난

시너지를 만들었다.

<br>

**통합** 딥러닝 시대

: 데이터가 많아지고 연산 능력이 증가하면서 ( GPU ) 머신러닝 연구자들은 예전 신경망 이론을 다시 접목.

훨씬 더 많은 데이터와 새로 개발된 이론을 합치면서 머신러닝만 사용하는 모델을 넘어서는 결과를 얻을 수 있게 된다.

<br>

#### 1.3 Probability Theory

- Probability

  - Conditional Probabilty

    Posterior = Likelihood * Prior Knowledge / Normalizing constant

    P(A|B) = P(A∩B) / P(B) = P(A|B) * P(B) / P(A)

<br>

- MLE [ Maximum Likelihood Estimation, 최대우도추정 ]

  감사하게도 MLE 에 대해 쉽게 설명해주신 포스팅 : [MLE by ratsgo](https://ratsgo.github.io/statistics/2017/09/23/MLE/)

<br>

- MAP

  ​

<br>

- Distribution [ 추후 필요한 때 정리 ]

  - Normal Distribution
  - Beta Distribution
  - Binomial Distribution
  - Multinomial Distribution

<br>

### 2. Supervised vs Unsupervised

#### 2.1 Supervised Learining

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.



![supervised_learning](https://github.com/JUWON-KEVIN-LEE/ML/blob/master/images/supervised_learning.png)

- Methodologies
  - Classfication : estimating a discrete dependent value from observations.
  - Regression : estimating a (continuous) dependent value from observations.
  - etc ...



#### 2.1.1 Linear Regression & Gradient Descent, Convex Function

- **Linear Regression** (선형 회귀)

  **Hypothesis :** H(x) = W [ weight ] * x + b [ bias ]

  **Y :** supervised data set

  ​

- **Gradient Descent** (경사 하강법)

  W := W - ∝ [ learning rate ] * ∂/∂W cost(W, b)

  ​      = W - ∝ * ∂/∂W 1/m * ∑ square(Hypothesis - Y)

  ​      = W - ∝ * 1/m * ∑ x * (Hypothesis - Y)

  ​

- **Convex function** (볼록 함수)

  **cost(W, b) = 1/m * ∑ square(Hypothesis - Y)**

  ![convexity](https://github.com/JUWON-KEVIN-LEE/ML/blob/master/images/convexity.png)

```python
import tensorflow as tf

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Linear regression model
hypothesis = X * W + b
# cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(4001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], 
    feed_dict={X:[1,2,4,0], Y:[0.5,1,2,0,]})
    if step % 40 == 0:
        print(step, cost_val, W_val, b_val)
'''
0 0.661618 [ 0.58980006] [-0.90425473]
40 0.161473 [ 0.71937096] [-0.61585635]
...
4000 9.03187e-13 [ 0.50000066] [ -1.13262649e-06]
                    >>> 0.5          >>> 0
'''
```




#### 2.2 Unsupervised Learining

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

- Methodologies
  - Clustering : estimating sets and affiliations of instances to the sets.
  - Filtering : estimating underlying and fundamental signals from the mixture of signals and noises.
  - etc ...
