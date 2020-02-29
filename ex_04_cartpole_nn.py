'''
CartPole 게임에 Neural Network 적용
'''

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras



def render_policy_net(model, max_steps = 200):
    '''신경망 모델 model과 최대 스텝 회수가 주어지면,
    게임 화면을 출력하는 함수'''

    env = gym.make('CartPole-v1') # 게임 환경 생성
    obs = env.reset() # 게임 환경 초기화
    env.render() # 초기 화면 렌더링

    for step in range(max_steps): # 최대 스텝 횟수만큼 반복
        # observation을 신경망 모델에서 적용해서 예측값을 알아냄
        p = model.predict(obs.reshape(1, -1))
        # 예측값 p를 이용해서 action을 결정(0:왼쪼그, 1: 오른쪽)
        action = int(np.random.random()) > p
        obs, reward, done, info = env.step(action)
        env.render() # 바뀐 환경을 렌더링





i __name__ == '__main__':
    # 신경망생성
    model = keras.Sequential() # keras.models.Sequential()
    # fully - connected 은닉층(hidden layer)를 추가
    model.add(keras.layers.Dense(4, activation='elu', input_shape=(4,)))
    # fully - connected 출력층을 추
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # 신경망 요약정보
    model.summary()

    # ReLU(Rectified Linear Unit)
    # f(x) = x if x > 0,
    #      = 0 otherwise

    # ELU(Exponential Linear Unit)
    # f(x) = x, if x > 0
    #      = alpha * (exp(x) -1), otherwise
    # 0 <= alpha < 1

    #Cartpole 게임환경(enviroment)를 생성
    env = gym.make('CarPole-v1')
    obs = env.reset()   # 환경 초기화
    print(obs)

    # observation을 신경망 모델에 forward 시켜서 예측값을 알아냄
    p = model.predict(obs.reshape(1, -1))
    print('p =', p)

    action = int(np.random.random() > p)
    print('action =', action)
    # p 값이 1에 가까울 수록 np.random.random() > p 는 False 가 될 확률이 높아짐
    # -> action은 0이 될 확률이 높아짐
    # -> action을 0으로 설정한다는 것은 cart를 왼쪽으로 가속도를 주겠다는 의미.
    # p 값이 0에 가까울 수록 np.random.random() > p는 True 가 될 확류링 높아짐.
