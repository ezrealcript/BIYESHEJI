from PyQt5.QtCore import QThread, pyqtSignal

# 目标检测代码
import numpy as np
import time
import math
import os

from scipy import io
import tensorflow as tf
import matplotlib.pyplot as plt
import os.path as op
import h5py
from sklearn import metrics

latent_dim = 20
epoch = 30

batch_size = 1000


class Data_mat(object):
    def __init__(self, filepath):
        self.data = io.loadmat(filepath)    #输入.mat文件
        self.img = np.array(self.data['data'][:], dtype=np.float64).T  # image二维
        self.tgt = np.array(self.data['d'][:], dtype=np.float64).T  # target
        self.grt = np.array(self.data['labelpic'][:], dtype=np.float64) .T # groundtruth
        self.x = self.img.shape[2]
        self.y = self.img.shape[1]
        self.z = self.img.shape[0]
        # plt.imshow(self.img)  # 输出 残差CEM 的结果
        # plt.savefig("./img.jpg")

# 约束能量最小化
# @ray.remote
def cem_detector(CEM_Data):
    # size = CEM_Data.img.shape  # get the size of image matrix
    R = np.dot(CEM_Data.img, CEM_Data.img.T / CEM_Data.y)  # R = X*X'/size(X,2);   R = (1/N) * X * X'
    w = np.dot(np.linalg.pinv(R), CEM_Data.tgt)  # w = (R+lamda*eye(size(X,1)))\d ;???   w = Rni * d
    w2 = np.dot(CEM_Data.tgt.T, w)   # w2 = d' * Rni * d
    w = np.divide(w, w2)  # w = w1/w2
    result = np.dot(w.T, CEM_Data.img).T  # y=w' * X;
    np.save('./results/result', result)
    # return result


# @ray.remote
def cem_residual_detector(CEM_Data, residual, reconstruct_result, d_minMax):
    size = CEM_Data.img.shape  # get the size of image matrix
    lamda = 10e-1
    # R = np.dot(reconstruct_result.T, reconstruct_result/size[1])# R = X*X'/size(X,2);
    # Tikhonov 正则化，确保 R’ 可逆
    R = np.dot(reconstruct_result.T, reconstruct_result)
    l = np.dot(lamda, np.eye(size[0]))
    R = np.add(R, l)
    R = R / size[1]   # R‘ = 1/N * X' * X + λ * I
    w = np.dot(np.linalg.pinv(R), d_minMax)  # w = (R+lamda*eye(size(X,1)))\d ;
    w2 = np.dot(d_minMax.T, w)
    w = np.divide(w, w2)   # w = (Rni * d) / (d' * Rni * d)
    result = np.dot(w.T, residual.T).T  # y=w'* X;
    np.save('./results/results_residual', result)
    # return result


class CVAE(tf.keras.Model):    # 条件变分自编码器 定义网络
    def __init__(self, latent_dim, input_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim   # 隐藏层维数
        self.input_dim = input_dim   #输入层维数
        self.inference_net = tf.keras.Sequential(        # 构建Sequential网络
            [
                tf.keras.layers.InputLayer(input_shape=(input_dim,)),   # 输入层（维数）
                # tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu),   # Dense(维数，解决死亡relu问题)
                tf.keras.layers.Dense(100, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(100, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu),
                # tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(input_dim, activation='tanh'),   # 将数据压缩到-1到1之间，输出期望为0.
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))   # 用于从“服从指定正态分布的序列”中随机取出指定个数的值
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):    # 编码器
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)  # 切割张量（需要切割的张量，切割的份数，切割的维度）
        return mean, logvar

    def reparameterize(self, mean, logvar):  #重新参数化
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def reparameterize_1(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):    #解码器
        logit = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logit)   # sigmoid函数
            return probs
        return logit


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = np.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

# 计算SAM 光谱角填图
def compute_SAM(x, x_logit, d_input):
    SAM = []
    SAM1 = []
    SAM2 = []
    num = x.shape[0]
    for i in range(num):
        d = tf.transpose(d_input)   #转置
        A = tf.reduce_sum(tf.multiply(x_logit[i, :], d))   # 矩阵求和
        B = tf.norm(x_logit[i, :], ord=2)   # 计算二范数
        C = tf.norm(d, ord=2)
        defen = tf.math.divide(A, B * C + 0.00001)
        defen = tf.acos(defen)  # 计算余弦
        SAM.append(defen)
    for i in range(len(SAM)):
        tf.cond(SAM[i] < SAM[20], lambda: SAM1.append(SAM[i]), lambda: SAM2.append(SAM[i]))
    sam_loss = tf.reduce_mean(SAM1)  # 计算向量某一维度的平均值
    return 0.1 * sam_loss

# 计算CEM损失函数
def compute_CEM(y_true, y_pred, d_input):
    CEM = []
    lamda = 1e-6

    x_trans = tf.transpose(y_true)  # 转置
    x_re_trans = tf.transpose(y_pred)  # 转置
    R = tf.matmul(x_trans, y_true)  # 矩阵相乘
    num = y_true.shape[0]
    print(y_true.shape[0])
    R = tf.math.divide(R, num)  # 除法
    R = R + lamda * np.identity(y_true.shape[1])
    w = tf.matmul(tf.linalg.inv(R), d_input) / (tf.matmul(tf.transpose(d_input), tf.matmul(tf.linalg.inv(R), d_input)))
    for i in range(num):
        cem = tf.matmul(tf.transpose(w), tf.reshape(x_re_trans[:, i], [x_re_trans.shape[0], 1]))
        cem = cem[0, 0]
        CEM.append(cem)
    cem_loss = tf.reduce_mean(CEM)
    return 0.0001 * cem_loss

# 计算mf 匹配滤波
def compute_mf(y_true, y_pred, d_input):
    mf = []
    lamda = 1e-6
    row = y_true.shape[0]  # 行,像素个数
    col = y_true.shape[1]  # 列,光谱

    x_trans = tf.transpose(y_true)
    x_re_trans = tf.transpose(y_pred)
    a = tf.reduce_mean(x_trans)
    r = tf.matmul(x_trans - a, tf.transpose(x_trans - a))
    r = tf.math.divide(r, row)
    r = r + lamda * np.identity(y_true.shape[1])
    w = tf.matmul(tf.linalg.inv(r), d_input - a)

    for i in range(row):
        mf_temp = tf.matmul(tf.transpose(w), tf.reshape(x_re_trans[:, i] - a, [x_re_trans.shape[0], 1]))
        mf_temp = mf_temp[0, 0]
        mf.append(mf_temp)
    mf = tf.nn.top_k(mf, k=10).values   # 找到输入的张量的最后的一个维度的最大的k个值和它的下标
    mf_loss = tf.reduce_mean(mf)
    return 0.000001 * mf_loss


@tf.function
def compute_loss(model, x, d_input):   # 计算损失函数
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cem_loss = compute_CEM(x, x_logit, d_input)
    mse_loss = tf.reduce_mean(tf.square(x_logit - x), axis=-1)
    kl_div = - 0.5 * tf.reduce_sum(1. + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)
    return kl_div + mse_loss + cem_loss


@tf.function
def train_step(model, x, optimizer, d_input):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, d_input)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def mnist_dataset(batch_size):
    x_input = np.array(train_data).astype('float32')
    x_input = 2 * ((x_input - x_input.min()) /
                   (x_input.max() - x_input.min()))
    x_input = (tf.data.Dataset.from_tensor_slices(x_input).batch(batch_size))

    x_input_val = np.array(val_data).astype('float32')
    x_input_val = 2 * ((x_input_val - x_input_val.min()) /
                       (x_input_val.max() - x_input_val.min()))
    x_input_val = (tf.data.Dataset.from_tensor_slices(x_input_val).batch(batch_size))

    return x_input, x_input_val

# 用变分自编码器训练背景样本
def train_func_distributed():
    x_input, x_input_val = mnist_dataset(batch_size)

    input_dim = train_data.shape[1]
    optimizer = tf.optimizers.Adam(1e-4)  # 学习率动态调整
    model = CVAE(latent_dim, input_dim)
    # 训练
    for e in range(1, epoch + 1):
        loss_train = tf.keras.metrics.Mean()
        start_time1 = time.time()
        for train_x in x_input:
            train_step(model, train_x, optimizer, d_input)
            loss_train(compute_loss(model, train_x, d_input))
        end_time1 = time.time()
        loss_train = loss_train.result()
        if e % 1 == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in x_input_val:
                loss(compute_loss(model, test_x, d_input))
            elbo = loss.result()
            # display.clear_output(wait=False)
            # 输出每一维的训练时间
            print('Epoch:{}, Train set ELBO:{}, Test set ELBO:{}, Time:{}'
                  .format(e, loss_train, elbo, end_time1 - start_time1))

            # success_epoh.emit(e, round(loss_train,5), round(elbo,5), round(end_time1 - start_time1, 5))

    # model.save_weights('./model/checkpoint')
    model.load_weights('./model/checkpoint')
    mean, logvar = model.encode(test_data)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    x_logit = x_logit.numpy()

    return x_logit

# 线程处理代码

class NewTaskThread(QThread):

    # 信号，触发信号，更新窗体
    success = pyqtSignal(int, str, str, str)
    error = pyqtSignal(int, str, str, str)
    success_cem = pyqtSignal(int, str, float, str)   # CEM粗检测
    success_zcem = pyqtSignal(str, float, str)  # ZCEM检测
    success_cvae = pyqtSignal(str, float, str)  # CVAE
    success_epoh = pyqtSignal(int, float, float, float)


    def __init__(self, row_index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.row_index = row_index

    def train_func_distributed(self):
        x_input, x_input_val = mnist_dataset(batch_size)

        input_dim = train_data.shape[1]
        optimizer = tf.optimizers.Adam(1e-4)  # 学习率动态调整
        model = CVAE(latent_dim, input_dim)
        # 训练
        for e in range(1, epoch + 1):
            loss_train = tf.keras.metrics.Mean()
            start_time1 = time.time()
            for train_x in x_input:
                train_step(model, train_x, optimizer, d_input)
                loss_train(compute_loss(model, train_x, d_input))
            end_time1 = time.time()
            loss_train = loss_train.result()
            if e % 1 == 0:
                loss = tf.keras.metrics.Mean()
                for test_x in x_input_val:
                    loss(compute_loss(model, test_x, d_input))
                elbo = loss.result()
                # display.clear_output(wait=False)
                # 输出每一维的训练时间
                print('Epoch:{}, Train set ELBO:{}, Test set ELBO:{}, Time:{}'
                      .format(e, loss_train, elbo, end_time1 - start_time1))

                self.success_epoh.emit(e, loss_train, elbo, end_time1 - start_time1)
        # model.save_weights('./model/checkpoint')
        model.load_weights('./model/checkpoint')
        mean, logvar = model.encode(test_data)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        x_logit = x_logit.numpy()
        return x_logit

    def run(self):
        start_time_all = time.time()
        global train_data, d_input, test_data, val_data

        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 强制使用cpu
        # cpu_num = 8  # 指定使用的CPU个数
        # config = tf.ConfigProto(device_count={"CPU": cpu_num},
        #                         inter_op_parallelism_threads=cpu_num,
        #                         intra_op_parallelism_threads=cpu_num,
        #                         log_device_placement=True)
        # # 开始训练
        # with tf.Session(config=config) as sess:
        # # 以下编写自己的代码
        print('Number of CPUs in the system: {}'.format(os.cpu_count()))


        CEM_Data = Data_mat('./fuse_result.mat')  # load data
        target = CEM_Data.tgt
        GT = CEM_Data.grt
        img_x = CEM_Data.x
        img_y = CEM_Data.y
        img_z = CEM_Data.z
        CEM_Data.img = np.reshape(CEM_Data.img, (img_z, img_x * img_y))
        img2 = CEM_Data.img

        # cem 约束能量最小化粗检测
        print('detector:' + 'CEM')
        time_cem_start = time.time()
        cem_detector(CEM_Data)
        time_cem_stop = time.time()
        time_all = time_cem_stop - time_cem_start
        time_all = round(time_all,5)
        print("CEM粗检测：")
        print(time_cem_stop - time_cem_start)  # cem粗检测所用时间
        result = np.load('./results/result.npy')  # cem得到的用于背景训练的样本
        ZCEM = np.array(result).reshape(img_y, img_x)  # 二维排列
        ZCEM = abs(ZCEM)  # 绝对值化
        result_coarse = np.array(ZCEM.T)  # 粗检测结果

        # cem粗检测结果展示
        # Coarse detection
        plt.imshow(result_coarse)
        plt.savefig("./CEMcujaince.jpg")
        # plt.show()
        groundtruth = GT.reshape(img_x * img_y).T
        groundtruth = np.where(groundtruth > 0, 1, 0)  # ground truth 有监督训练的训练集的分类准确性
        result_cem = result_coarse.reshape(img_x * img_y).T
        auc = "%.5f" % metrics.roc_auc_score(groundtruth, result_cem)  # 求正确率
        print('%s_AUC: %s' % ('CEM', auc))
        # CEM粗检测信号
        self.success_cem.emit(self.row_index, "CEM粗检测", time_all, auc)

        # Binarization二值化，获取背景样本
        result_binary = np.array(ZCEM.T)
        threshold, upper, lower = 0.15, 1, 0
        a = np.where(result_binary.T > threshold)  # 与阈值进行比较
        b = np.where(result_binary.T <= threshold)
        result_binary = np.where(result_binary > threshold, upper, lower)
        plt.imshow(result_binary)  # 展示背景样本
        plt.savefig("./CEMbeijingyangben.jpg")
        # plt.show()

        data_r = img2.T
        ratio = 0.75  # 75%
        iMax = np.size(b[1])
        background = data_r[b[0] * img_x + b[1]]
        rowrank = np.arange(background.shape[0])  # 求得行的维数，  np.arange（）返回一个从起点到终点的等差数列
        np.random.shuffle(rowrank)  # 重新排序，得到一个随机序列
        background = background[rowrank, :]
        m = math.ceil(np.dot(ratio, iMax))  # math.ceil（）向上取整
        train_data = background[0:m, :]  # 75%的背景样本作为训练值
        val_data = background[m:iMax, :]

        # 读入目标及测试集
        d = target.astype('float32')
        d_input = 2 * ((d - d.min()) / (d.max() - d.min()))

        test_data = img2.astype('float32')
        test_data = img2.reshape(img_z, img_x, img_y, order='F')
        test_data = test_data.reshape(img_z, img_x * img_y).T
        test_data = 2 * ((test_data - test_data.min()) /
                         (test_data.max() - test_data.min())) - 1

        # 训练  进行背景重构  用变分自编码器训练背景样本
        time1 = time.time()
        x_logit = self.train_func_distributed()
        time2 = time.time()
        time_all2 = time2 - time1
        time_all2 = round(time_all2,5)
        print("ZCEM:")
        print(time2 - time1)
        # self.success_zcem.emit("ZCEM", time_all2)


        data_r_minMax = ((data_r - data_r.min()) / (data_r.max() - data_r.min()))
        reconstruct_result = ((x_logit - x_logit.min()) / (x_logit.max() - x_logit.min()))
        d_minMax = ((d - d.min()) / (d.max() - d.min()))
        residual = []
        residual = np.subtract(data_r_minMax, reconstruct_result)  # 用背景重构图形与原图像作差得到3D残差

        time_cem2_start = time.time()
        cem_residual_detector(CEM_Data, residual, reconstruct_result, d_minMax)  # CEM对残差进行检测，计算自相关矩阵
        time_cem2_stop = time.time()
        print("cem残差：")
        print(time_cem2_stop - time_cem2_start)
        results_residual = np.load('./results/results_residual.npy')
        ZCEM_residual = np.array(results_residual).reshape(img_y, img_x)
        ZCEM_residual = abs(ZCEM_residual.T)
        plt.imshow(ZCEM_residual)  # 输出 残差CEM 的结果
        plt.savefig("./ZCEM.jpg")
        # plt.show()

        lamda = 10e-1
        up = 1 - np.exp(np.dot(-lamda, result_coarse))
        do = 0
        output = np.where(result_coarse >= 0, up, do)  # 非线性函数 抑制背景，保留目标
        result_coarse_nonlinear = ZCEM_residual * output  # 得到最终的检测结果

        print('detector:' + 'CVAE')
        plt.imshow(result_coarse_nonlinear)
        plt.savefig("./CEM CVAE.jpg")
        # plt.show()
        groundtruth = CEM_Data.grt.reshape(CEM_Data.x * CEM_Data.y).T
        groundtruth = np.where(groundtruth > 0, 1, 0)
        result_final = result_coarse_nonlinear.reshape(CEM_Data.x * CEM_Data.y).T
        ZCEM_final = ZCEM_residual.reshape(CEM_Data.x * CEM_Data.y).T
        auc = "%.5f" % metrics.roc_auc_score(groundtruth, ZCEM_final)
        auc1 = "%.5f" % metrics.roc_auc_score(groundtruth, result_final)
        print('%s_AUC: %s' % ('CEM', auc))
        print('%s_AUC: %s' % ('CEM', auc1))
        self.success_zcem.emit("ZCEM", time_all2, auc)

        end_time_all = time.time()
        time_all3 = round(end_time_all - start_time_all, 3)
        print("CVAE:")
        print(end_time_all - start_time_all)
        self.success_cvae.emit("CVAE", time_all3, auc1)

        # 具体线程应该做的事
        self.success.emit(self.row_index, "xx", "xx", "xx")
        self.error.emit(1, "xx", "xx", "xx")

        pass