# coding:utf-8

import numpy as np
import time
from PIL import Image
from skimage import transform
import tensorflow as tf

"""
本项目主要实现的是对人脸图像的识别训练与测试，主要步骤如下：
1、预处理数据（包含读取原图像数据、标准化数据和分割数据）
2、构建训练模型（包含定义输入输出数据、训练模型、模型评估）
3、定义训练方法（包含训练次数、训练评估指数、验证评估指数）
4、保存训练模型（将训练出的模型持久化以便重复调用）
5、测试训练模型（模型调用、数据导入、预测输出）
"""

model_path = 'Model/model.ckpt'  # 模型保存路径
wa, ha = 1140, 942  # 原大图像分辨率
w, h = 57, 47  # 原图像中单个图片分辨率
ws, hs, cs = 64, 64, 1  # 单个图像预处理分辨率及通道数

# 1、预处理数据
# 图像数据读取及处理
def load_data(data_path):
    img = Image.open(data_path)
    # 图像数据矩阵化及灰度化（归一化）
    imgs = np.asarray(img, dtype='float32') / 256
    # 由于原图像非正方形，为了便于神经网络对数据的运算，需要对图像做膨胀处理，转化为宽高相等的正方形
    # 64x64的分辨率与原图分辨率较为接近，且不会因为过多膨胀造成的特征扭曲或丢失
    imgs = transform.resize(imgs, (20*ws, 20*hs))
    print(imgs.shape)
    # 构造人脸数据矩阵
    faces = np.empty((400, ws, hs, ))
    # 填充400个人脸数据（每个数据的范围是 列数 * 宽:(列数 + 1) * 宽, 行数 * 高:(行数 + 1) * 高）
    for row in range(20):  # 遍历20列数据
        for column in range(20):  # 遍历20行数据
            # 遍历每行，并且每行从左到右遍历赋值
            faces[row * 20 + column] = imgs[row * ws:(row + 1) * ws, column * hs:(column + 1) * hs]
    # 设置400个样本图的标签
    label = np.empty((400, ))
    for i in range(40):
        label[i * 10:i * 10 + 10] = i
    label = np.asarray(label, np.int32)
    # 分成训练集、验证集、测试集，大小如下
    train_data = np.empty((320, ws, hs, ))   # 320个训练样本
    train_label = np.empty((320, ))          # 320个训练样本
    valid_data = np.empty((40, ws, hs, ))    # 40个验证样本
    valid_label = np.empty((40, ))           # 40个验证样本
    test_data = np.empty((40, ws, hs, ))     # 40个测试样本
    test_label = np.empty((40, ))            # 40个测试样本
    # 填充训练集、验证集、测试集数据
    for j in range(40):
        train_data[j * 8:j * 8 + 8] = faces[j * 10:j * 10 + 8]
        train_label[j * 8:j * 8 + 8] = label[j * 10:j * 10 + 8]
        valid_data[j] = faces[j * 10 + 8]
        valid_label[j] = label[j * 10 + 8]
        test_data[j] = faces[j * 10 + 9]
        test_label[j] = label[j * 10 + 9]
    return train_data, train_label, valid_data, valid_label, test_data, test_label

# 提取训练集、验证集、测试集数据
x_train, \
y_train, \
x_val, \
y_val, \
x_test, \
y_test = load_data('face_image_data.gif')
# 增加图像通道数维度，这样数据集共4个维度，样本个数、宽度、高度、通道数
x_train = x_train[:, :, :, np.newaxis]
x_val = x_val[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]
print('样本数据集的维度：', x_train.shape, y_train.shape)
print('验证数据集的维度：', x_val.shape, y_val.shape)
print('测试数据集的维度：', x_test.shape, y_test.shape)

# 2、构建训练模型
# 定义占位符
x = tf.placeholder(tf.float32, shape=[None, ws, hs, cs], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

# 定义卷积神经网络
def CNNlayer():
    # 第一个卷积层（64—>32)
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # 第二个卷积层(32->16)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # 第三个卷积层(16->8)
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    re1 = tf.reshape(pool3, [-1, 8 * 8 * 64])
    # 全连接层
    dense1 = tf.layers.dense(inputs=re1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
    dense2 = tf.layers.dense(inputs=dense1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
    logits = tf.layers.dense(inputs=dense2,
                             units=40,
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
    return logits

# 运行卷积神经网络模型
logits = CNNlayer()
print("shape of logits:", logits.shape)
b = tf.constant(value=1, dtype=tf.float32)                   # 定义一个常数
logits_eval = tf.multiply(logits, b, name='logits_eval')      # 常数与矩阵相乘

#  定义模型损失函数与准确率
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 3、定义训练方法
# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

# 样本训练次数
n_epoch = 100
# 每次使用batch_size个数据来更新参数（batch_size大小必须小于等于验证集数据个数，否则无法得出验证损失及准确率信息）
batch_size = 32  # 该数值的设置对训练效果影响较大，需要根据训练效果调整
saver = tf.train.Saver()  # 创建saver
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)  # 配置GPU占用资源，防止过度占用导致运行出错
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  # 定义一个会话
sess.run(tf.global_variables_initializer())  # 模型参数全局初始化
all_average_val_loss = []  # 用于存储数据验证平均损失数据
all_average_val_acc = []  # 用于存储数据验证平均准确率数据

for epoch in range(n_epoch):  # 总开始遍历每一层的epoch
    print("epoch:", epoch + 1)  # 在每层epoch的开始训练之前，打印当前epoch层级
    start_time = time.time()  # 返回每个epoch开始训练时候的时间，用于计算每个epoch以及总共消耗了多少时间
    # 训练
    train_loss, train_acc, n_batch = 0, 0, 0  # 初始值默认为0
    # 利用minibatches函数从训练数据中以打乱顺序的方式提取批数据进行训练，批大小为64
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        # 运行train_op，loss，acc三个运算，利用feed_dict给占位符传输数据进行遍历训练
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        # 将每次迭代的损失值err，准确率ac，批次数n_batch进行累计
        train_loss += err
        train_acc += ac
        n_batch += 1
    print("   train loss: %f" % (np.sum(train_loss) / n_batch))  # 计算平均loss值
    print("   train acc: %f" % (np.sum(train_acc) / n_batch))  # 计算平均acc值
    # 验证
    val_loss, val_acc, n_batch = 0, 0, 0  # 初始值默认为0
    # 利用minibatches函数从验证数据中以不打乱顺序的方式提取批数据进行验证，批大小为32
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        # 利用feed_dict给占位符传输数据进行遍历验证，运行train_op，loss，acc三个op运算
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
        # 将每次迭代的损失值err，准确率ac，批次数n_batch进行累计
        val_loss += err
        val_acc += ac
        n_batch += 1
    print("   validation loss: %f" % (np.sum(val_loss) / n_batch))  # 计算平均loss值
    all_average_val_loss.append(float(np.sum(val_loss) / n_batch))  # 记录验证平局损失
    print("   validation acc: %f" % (np.sum(val_acc) / n_batch))  # 计算平均acc值
    all_average_val_acc.append(float(np.sum(val_acc) / n_batch))  # 记录验证平局准确率
    print("   epoch time: %f" % (time.time() - start_time))  # 计算每个epoch所消耗的时间
    print('-------------------------------------------------------')
print("The min val loss is ", min(all_average_val_loss))
print("The max val acc is ", max(all_average_val_acc))
# 获取验证准确率最高时的训练次数，以便再次训练达到最优效果
for i in range(len(all_average_val_acc)):
    if all_average_val_acc[i] == max(all_average_val_acc):
        print("The max val acc epoch num is ", i+1)

# 4、保存训练模型
saver.save(sess, model_path)
sess.close()

# 5、测试训练模型
with tf.Session() as sess:              # 创建会话，用于执行已经定义的运算
    data = []                           # 定义空白列表，用于保存处理后的验证数据
    data = x_test                       # 将处理过后的验证图像数据保存在前面创建的空白data列表当中
    # 加载之前训练好的模型
    saver = tf.train.import_meta_graph('Model/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('Model'))
    # 获取模型参数
    graph = tf.get_default_graph()            # 获取当前的默认计算图
    x = graph.get_tensor_by_name("x:0")       # 返回给定名称的tensor
    print(x)                                  # 返回加载的模型的参数
    # 加载待预测数据
    feed_dict = {x: data}                     # 利用feed_dict，给占位符传输数据
    logits = graph.get_tensor_by_name("logits_eval:0")      # 返回logits_eval对应的tensor
    print(logits)
    # 对测试数据进行预测
    classification_result = sess.run(logits, feed_dict)      # 利用feed_dict把数据传输到logits进行验证图像预测
    print(classification_result)                            # 打印预测矩阵
    print(tf.argmax(classification_result, 1).eval())        # 打印预测矩阵每一行的最大值的下标
    output = []                                             # 定义空白列表output
    output = tf.argmax(classification_result, 1).eval()      # 选择出预测矩阵每一行最大值的下标，并将字符串str当成有效的表达式来求值并返回计算结果，将其赋值给output
    print(output)
    print(output.shape)
    # 输出预测结果(原数据排列为从上到下，每行从左到右，其对应标签为顺序0~39)
    for i in range(len(output)):                      # 遍历len(output)=40的人脸标签
        print("face", i+1, "prediction:", output[i])  # 输出每种人脸预测值最高的选项（i+1代表测试图像序号，output[i]代表预测的标签值）
