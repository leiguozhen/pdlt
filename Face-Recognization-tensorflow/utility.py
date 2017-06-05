import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split


SIZE = 64
MAXFACES = 1000

def get_detector():
    return dlib.get_frontal_face_detector()


def get_camera():
    return cv2.VideoCapture(0)


# 随机改变图片的亮度与对比度，增加图片的多样性
def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    for i in range(0, w):
        for j in range(0, h):
            for c in range(3):# R, G, B
                tmp = int(img[j, i, c]*light + bias)
                tmp = min(tmp, 255)
                tmp = max(tmp, 0)
                img[j, i, c] = tmp
    return img


def process_picture(img, detector, SIZE):
    # 转为灰度图片
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用detector进行人脸检测
    dets = detector(gray_img, 1)
    # 使用enumerate 函数遍历序列中的元素以及它们的下标
    # 下标i即为人脸序号
    # left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
    # top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
    faces_ret = []
    for d in dets:
        x1 = max(0, d.top())
        y1 = max(0, d.bottom())
        x2 = max(0, d.left())
        y2 = max(0, d.right())
        face = img[x1:y1, x2:y2]
        face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
        face = cv2.resize(face, (SIZE, SIZE))
        faces_ret.append(face)
    return faces_ret


def get_faces_from_camera(camera, detector, SIZE):
    success, img = camera.read()
    return process_picture(img, detector, SIZE)


def get_faces_from_picture(detector, img_filename, SIZE):
    # 从文件读取图片
    img = cv2.imread(img_filename)
    return process_picture(img, detector, SIZE)


def save_face(face, folder, counter):
    filename = folder + '/' + str(counter) + '.jpg'
    cv2.imwrite(filename, face)


def show_face(face):
    cv2.imshow('image', face)


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)
    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right




x = tf.placeholder(tf.float32, [None, SIZE, SIZE, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)


def cnnLayer():
    # 第一层
    W1 = weightVariable([3,3,3,32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层
    Wf = weightVariable([8*16*32, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*16*32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512,2])
    bout = weightVariable([2])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

