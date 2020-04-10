import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from tensorflow.keras import layers

model = tf.keras.Sequential()
# 64개의 유닛을 가진 완전 연결 층을 모델에 추가합니다:
model.add(layers.Dense(64, activation='relu'))
# 또 하나를 추가합니다:
model.add(layers.Dense(64, activation='relu'))
# 10개의 출력 유닛을 가진 소프트맥스 층을 추가합니다:
model.add(layers.Dense(10, activation='softmax'))