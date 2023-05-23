# 使用theano作为backend，是用keras_tf1.12作为解释器
#增加network coding

# This code can transmite the codes which represent the ganeral code, so this code can transmited the general data using network coding.


# import os
# os.environ['KERAS_BACKEND']='theano'
#
#
# import numpy as np
from keras.models import Sequential
# from keras.layers.core import Dense, Lambda
# from keras import backend as K
# import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model#导入网络结构可视化模块。 https://blog.csdn.net/baymax_007/article/details/83152108
# #matplotlib inline
# from keras.engine.network import Network
# from keras.layers import *
# from keras import backend
# from keras.models import Model
# from keras.preprocessing import image
# import keras.backend as K
# from keras.backend import slice
# import tensorflow as tf
# import matplotlib.pyplot as plt
#
# import numpy as np
# import os
# import random
# import scipy.misc
# from tqdm import *
# from keras.layers import Reshape




#采用函数形式时如果采用theono作为框架（# os.environ['KERAS_BACKEND']='theano'），会出问题，这里采用tensorflow作为框架，所以替换成下列引入包
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0"

### Imports ###
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
#from keras.engine.topology import Container
from keras.engine.network import Network
from keras.layers import *
from keras import backend
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.backend import slice
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import os
import random
import scipy.misc
from tqdm import *
from keras.layers import Reshape



#Parameters
k = 8                       # number of information bits
N = 16                      # code length
train_SNR_Eb = 1            # training-Eb/No

epochs = 2**16            # number of learning epochs
epochs = 500 # 测试使用
code = 'polar'              # type of code ('random' or 'polar')
design = [128, 64, 32]      # each list entry defines the number of nodes in a layer
batch_size = 256            # size of batches for calculation the gradient
LLR = False                # 'True' enables the log-likelihood-ratio layer
optimizer = 'adam'
loss = 'mse'                # or 'binary_crossentropy'
beta = 1.0

def network_coding_loss(s_true, s_pred):

    print(s_true.shape, s_pred.shape)
    return beta * K.sum(K.square(s_true - s_pred))


train_SNR_Es = train_SNR_Eb + 10*np.log10(k/N)#Es和Eb就是这个关系，网上有很多帖子，本程序下半部分的注释也有相应的网址。
train_sigma = np.sqrt(1/(2*10**(train_SNR_Es/10)))#这里的sigma是方差的意思，方差就是噪声。至于这个公式，应该是随便取的一个ie公式，代表
                                                  #表示的是信号功率train_SNR_Es的一个函数，表示噪声的幅度，不过前面加了
                                                  #train_，难道在训练阶段考虑的噪声，下面的训练阶段不是没有考虑噪声吗？在noise_layers和
                                                  #llr_layers里被调用了，调用的结果看起来是model用了，训练没有用。但是为什么前面加train_呢？
                                                  #那就看看训练语句就可以了。但model.summary()， history = model.fit(x, d, batch_size=batch_size, epochs=epochs, verbose=0, shuffle=True)
                                                  #应该已经将噪声考虑进去了，看来keras的网络结构规定并不是仅仅就这两句话，应该包含modes，modes可能就是规定网络结构的
                                                  #需要好好看看keras。但是这里如果噪声考虑进网络，那整个文章的创新点的理解需要重新来一遍，原先以为没有噪声，只有生成矩阵G的作用
                                                  #好像就不对劲，那还要标签什么的有什么用呢。再一个网络没有抓住噪声的效果。



#Define NN model
def modulateBPSK(x):
    return -2*x +1;

def addNoise(x, sigma):
    w = K.random_normal(K.shape(x), mean=0.0, stddev=sigma)
    return x + w

def ber(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)))

def return_output_shape(input_shape):
    return input_shape

def compose_model(layers):
    model = Sequential()
    for layer in layers:
        model.add(layer)
    return model

def log_likelihood_ratio(x, sigma):
    return 2*x/np.float32(sigma**2)

def errors(y_true, y_pred):
    # return K.sum(K.not_equal(y_true, K.round(y_pred)))
    return tf.reduce_mean(K.cast(K.equal(y_true, K.round(y_pred)), dtype='int32'))#K.sum不能直接对bool类型数据进行求和操作（tensorflow中tf.reduce_sum()也是）， https://blog.csdn.net/qian_fighting/article/details/86170648


# Define modulator
modulator_layers = [Lambda(modulateBPSK,
                          input_shape=(N,), output_shape=return_output_shape, name="modulator")]
modulator = compose_model(modulator_layers)#这个方法被调用了
modulator.compile(optimizer=optimizer, loss=loss)#这个函数没有被其它地方直接调用，可能是keras的特殊用法，就是将modulatot提前编译一下。不编译不能用？

# Define noise
noise_layers = [Lambda(addNoise, arguments={'sigma':train_sigma},
                       input_shape=(N,), output_shape=return_output_shape, name="noise")]
noise = compose_model(noise_layers)
noise.compile(optimizer=optimizer, loss=loss)

# Define LLR
llr_layers = [Lambda(log_likelihood_ratio, arguments={'sigma':train_sigma},
                     input_shape=(N,), output_shape=return_output_shape, name="LLR")]
llr = compose_model(llr_layers)
llr.compile(optimizer=optimizer, loss=loss)

# Define decoder
decoder_layers = [Dense(design[0], activation='relu', input_shape=(N,))]
for i in range(1,len(design)):
    decoder_layers.append(Dense(design[i], activation='relu'))#Dense()方法除非第一层需要指定输入形状，只要其他层的前面有层，只需要只需要指定第一个参数，也就是本层有多少个点，也同时是输出层。https://wenku.baidu.com/view/9b072038a46e58fafab069dc5022aaea988f4174.html
decoder_layers.append(Dense(k, activation='sigmoid'))

#这里定义了一个模型，和后面的model模型是同层次的，有.compile(optimizer=optimizer, loss=loss, metrics=[errors])，以备测试时使用
decoder = compose_model(decoder_layers)
decoder.compile(optimizer=optimizer, loss=loss, metrics=[errors])


###########################################################################################
activation_func = 'relu'


def network_code(Input_size): #


    x1 = Input(shape=(Input_size,))
    x2 = Input(shape=(Input_size,))

    x1_1 = Dense(32, activation=activation_func)(x1) ## TypeError: float() argument must be a string or a number, not 'TensorVariable'
    x2_1 = Dense(32, activation=activation_func)(x2) ##


    x1_1_1 = Lambda(lambda x1_1: x1_1[:, :16])(x1_1)
    x1_1_2 = Lambda(lambda x1_1: x1_1[:, 16:])(x1_1)

    x2_1_1 = Lambda(lambda x2_1: x2_1[:, :16])(x2_1)
    x2_1_2 = Lambda(lambda x2_1: x2_1[:, 16:])(x2_1)


    x3 = concatenate([x1_1_2, x2_1_1]) # 2 + 3

    x4 = Dense(16, activation=activation_func)(x3)
    x5 = Dense(32, activation=activation_func)(x4)


    x5_1 = Lambda(lambda x5: x5[:, :16])(x5)
    x5_2 = Lambda(lambda x5: x5[:, 16:])(x5)


    x6 = concatenate([x1_1_1, x5_1])  # 1 + 5_1
    x7 = concatenate([x2_1_2, x5_2])  # 2 + 5_2

    x_out1 = Dense(Input_size * 2, activation=activation_func)(x6)
    x_out2 = Dense(Input_size * 2, activation=activation_func)(x7)

    return Model(inputs=[x1, x2],outputs=x_out1, name='network_code')#为简便起见，这里只是给出一个sink节点


def make_model(input_size):#make_model的模型实际没被使用编译，只是作为普通函数返回最终我们想要的两个模型

    input = Input(shape=(input_size,))#如果是make_model(16)，这里就得是如此，如果是make_model((16,1)),这里就得是 Input(shape=(input_size))

    out_modulator = modulator(input)#第一层次模型
    # out_modulator1 = out_modulator[:, :8]
    # out_modulator2 = out_modulator[:, 8:]

    out_modulator1, out_modulator2 = Lambda(lambda x: [x[:, :8]])(out_modulator), Lambda(lambda x: [x[:, 8:]])(out_modulator)#keras的model层里不允许有函数，也不允许有切割之类的操作，本质上是函数，将其通过Lamda转化为层layer，参考帖子https://blog.csdn.net/zydyb

    network_code_model = network_code(input_size//2)#第二层次模型
    network_code_model.compile(optimizer='adam', loss=network_coding_loss)

    out_network_code_model = network_code_model([out_modulator1, out_modulator2])



    out_decoder = decoder(out_network_code_model)#第三层次模型


    return Model(inputs=input, outputs=out_decoder), network_code_model#输入时整个网络的输入，本来想像上面那样搞成调制之后的部分，这样和测试的元时代码匹配，但时程序出错了，就改一下。

networkcoding_decoding_model, network_code_model = make_model(16)#不用写batchsize，自动补充，也就是最终传入的数据形式为（batchsize,16)，network_code_model应该是只有网络编码部分的模型

# last_decoding_model.compile(optimizer=optimizer, loss=loss, metrics=[ber])#last_decoding_model的输入是上一层网络的输出，而不是开始的原始消息x，虽然可以这样做，主要是为了联合训练
networkcoding_decoding_model.compile(optimizer=optimizer, loss=loss, metrics=[ber])#networkcoding_decoding_model包括网络编码部分+原始的decoder部分，没有调制modulator部分。那整个模型的输入就是网络编码的输入。

#Data Generation
def half_adder(a, b):#好像就这个方法没被调用，是作者多写的，其他的都被调用了。
    s = a ^ b
    c = a & b
    return s, c


def full_adder(a, b, c):
    s = (a ^ b) ^ c
    c = (a & b) | (c & (a ^ b))
    return s, c


def add_bool(a, b):
    if len(a) != len(b):
        raise ValueError('arrays with different length')
    k = len(a)
    s = np.zeros(k, dtype=bool)
    c = False
    for i in reversed(range(0, k)):
        s[i], c = full_adder(a[i], b[i], c)
    if c:
        warnings.warn("Addition overflow!")
    return s


def inc_bool(a):
    k = len(a)
    increment = np.hstack((np.zeros(k - 1, dtype=bool), np.ones(1, dtype=bool)))
    a = add_bool(a, increment)
    return a


def bitrevorder(x):
    m = np.amax(x)
    n = np.ceil(np.log2(m)).astype(int)
    for i in range(0, len(x)):
        x[i] = int('{:0{n}b}'.format(x[i], n=n)[::-1], 2)
    return x


def int2bin(x, N):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        binary = np.zeros((len(x), N), dtype='bool')
        for i in range(0, len(x)):
            binary[i] = np.array([int(j) for j in bin(x[i])[2:].zfill(N)])
    else:
        binary = np.array([int(j) for j in bin(x)[2:].zfill(N)], dtype=bool)

    return binary


def bin2int(b):
    if isinstance(b[0], list):
        integer = np.zeros((len(b),), dtype=int)
        for i in range(0, len(b)):
            out = 0
            for bit in b[i]:
                out = (out << 1) | bit
            integer[i] = out
    elif isinstance(b, np.ndarray):
        if len(b.shape) == 1:
            out = 0
            for bit in b:
                out = (out << 1) | bit
            integer = out
        else:
            integer = np.zeros((b.shape[0],), dtype=int)
            for i in range(0, b.shape[0]):
                out = 0
                for bit in b[i]:
                    out = (out << 1) | bit
                integer[i] = out

    return integer


def polar_design_awgn(N, k, design_snr_dB):
    S = 10 ** (design_snr_dB / 10)
    z0 = np.zeros(N)

    z0[0] = np.exp(-S)
    for j in range(1, int(np.log2(N)) + 1):
        u = 2 ** j
        for t in range(0, int(u / 2)):
            T = z0[t]
            z0[t] = 2 * T - T ** 2  # upper channel
            z0[int(u / 2) + t] = T ** 2  # lower channel

    # sort into increasing order
    idx = np.argsort(z0)

    # select k best channels
    idx = np.sort(bitrevorder(idx[0:k]))

    A = np.zeros(N, dtype=bool)
    A[idx] = True

    return A


def polar_transform_iter(u):
    N = len(u)
    n = 1
    x = np.copy(u)
    stages = np.log2(N).astype(int)
    for s in range(0, stages):
        i = 0
        while i < N:
            for j in range(0, n):
                idx = i + j
                x[idx] = x[idx] ^ x[idx + n]
            i = i + 2 * n
        n = 2 * n
    return x


# Create all possible information words
d = np.zeros((2 ** k, k), dtype=bool)
for i in range(1, 2 ** k):
    d[i] = inc_bool(d[i - 1])

# Create sets of all possible codewords (codebook)
if code == 'polar':

    A = polar_design_awgn(N, k, design_snr_dB=0)  # logical vector indicating the nonfrozen bit locations
    x = np.zeros((2 ** k, N), dtype=bool)
    u = np.zeros((2 ** k, N), dtype=bool)
    u[:, A] = d#将u中由A指定的列的位置的值替换成d，每一行里，A指定了16个位置中的8个，然后d的每一行也有8个值，正好填写进去，但每个
               #值不一定是1，所以不太好观察，这个是polar码的编码理论里面的，不需要深究。

    for i in range(0, 2 ** k):
        x[i] = polar_transform_iter(u[i])
     #没有噪声；想想最小距离以及信噪比。
elif code == 'random':

    np.random.seed(4267)  # for a 16bit Random Code (r=0.5) with Hamming distance >= 2
    x = np.random.randint(0, 2, size=(2 ** k, N), dtype=bool)#随机编码的编码本，信息是8bit的每行索引值（转化为二进制），一共2^K个胖子，即码字。
    #1，没有最小距离的概念，那怎么体现乡农公式一定能解呢？一定能解不是在最小距离里吗？还是基于最小距离的解法只是乡农公式的一个子集呢？
    #   也就即使错误超过了最小距离，但是信噪比适合，通过香农公式一定能解出来，还是要彻底弄懂香浓公式，尤其是最小距离1/2角度的译码是什么关系？
    #2，这里没考虑噪声，就是没有噪声，网络编码里可以考虑噪声，也可以和这个思路一样，不考虑噪声。


#Train Neural Network
networkcoding_decoding_model.summary()#model.summary()输出模型各层的参数状况

# return Model(inputs=[x1, x2],outputs=x_out1, name='network_code')

# out_modulator = modulator(x)
# out_modulator1 = out_modulator[:, :8]
# out_modulator2 = out_modulator[:, 8:]

m = x.shape[0]
loss_history = []

#训练过程我们没有考虑噪声，原始论文考虑了噪声
for epoch in range(epochs):

    loss_network_coding_only = []#单纯的网络编码中间层的损失记录
    loss_all = []#最后一层的解码器的损失，也是总的损失，二者都需要每次epoch清0。

    t = tqdm(range(0, x.shape[0], batch_size),mininterval=0)

    for idx in t:

        #############################################################################################
        # 为联合训练准备输入输出的实际数据
        out_modulator = modulator.predict(x)#第一层模型产生数据
        out_modulator1 = out_modulator[:, :8]
        out_modulator2 = out_modulator[:, 8:]

        out_network_code_model = network_code_model.predict([out_modulator1, out_modulator2])#第二层模型（单独网络编码）产生数据

        # out_last_decoding_model = last_decoding_model.predict(out_network_code_model)#条语句没有必要，因为联合训练只需要上一层的输出作为本层的输入，这里last_decoding_model已经是最后一层，给出他的正常传播输出out_last_decoding_model没有意义，因为不会有下层网络联合训练需要用到它。

        # 进行实际的练习和训练，会用到前面的“为联合训练准备输入输出的实际数据”。

        # loss_network_coding_only.append(network_code_model.train_on_batch(x=[out_modulator1, out_modulator2],  y=x))#这两个是错误的信息输入，没有考虑清楚联合训练的问题，数据都是上一个模型的输出，下一层的输入
        # loss_all.append(last_decoding_model.train_on_batch(x=x, y=d))

        #*****************************************************************                                         # 第一层网络modulator没有要求训练，它的训练隐含在后续各层了，如果本层训练就是提特殊要求。

        loss_network_coding_only.append(network_code_model.train_on_batch(x=[out_modulator1, out_modulator2], y=x))#单纯网络编码的训练，让本层模型的输入输出尽量相等，有利于后面模型的解码。输入是上一层网络的输出,y作为标签不必是上一层的输出

        # loss_all.append(networkcoding_decoding_model.train_on_batch(x=out_network_code_model, y=d))#不能是(x=x, y=d))，输入是上一层网络的输出out_network_code_model，,y作为标签不必是上一层的输出

        loss_all.append(networkcoding_decoding_model.train_on_batch(x=x,  y=d))  # 输入是整个网络的输入，这个是整个网络的训练


        # # Update learning rate
        # K.set_value(autoencoder_model.optimizer.lr, lr_schedule(epoch))
        # #K.set_value(autoencoder_model2.optimizer.lr, lr_schedule(epoch))
        # K.set_value(reveal_model1.optimizer.lr, lr_schedule(epoch))
        # K.set_value(reveal_model2.optimizer.lr, lr_schedule(epoch))
        # K.set_value(reveal_model3.optimizer.lr, lr_schedule(epoch))

        t.set_description('Epoch {} | Batch: {:3} of {}. loss_network_coding_only {:10.2f} | loss_all {:10.2f} '.format(epoch + 1, idx, m, np.mean(loss_network_coding_only), np.mean(loss_all)))

    networkcoding_decoding_model.save_weights('models/model_'+ str(epoch))#整个网络训练的部分


    loss_history.append(np.mean(loss_all))



#Test NN
# test_batch = 1000
test_batch = 256
# num_words = 100000      # multiple of test_batch
# num_words = 3000
num_words = 256*10

SNR_dB_start_Eb = 0 #https://blog.csdn.net/wordwarwordwar/article/details/80299808
SNR_dB_stop_Eb = 5 #https://blog.csdn.net/li200503028/article/details/17026745
SNR_points = 20

SNR_dB_start_Es = SNR_dB_start_Eb + 10 * np.log10(k / N) #https://blog.csdn.net/sinat_38151275/article/details/79869891
SNR_dB_stop_Es = SNR_dB_stop_Eb + 10 * np.log10(k / N)

sigma_start = np.sqrt(1 / (2 * 10 ** (SNR_dB_start_Es / 10)))#http://blog.sina.com.cn/s/blog_61aa4e9d0101dze3.html
sigma_stop = np.sqrt(1 / (2 * 10 ** (SNR_dB_stop_Es / 10)))

sigmas = np.linspace(sigma_start, sigma_stop, SNR_points)

nb_errors = np.zeros(len(sigmas), dtype=int)
nb_bits = np.zeros(len(sigmas), dtype=int)

for i in range(0, len(sigmas)):

    for ii in range(0, np.round(num_words / test_batch).astype(int)):

        # Source
        np.random.seed(0)
        d_test = np.random.randint(0, 2, size=(test_batch, k))#一次batch_size=1000个消息，随机产生的，用于测试，每行8bit，类似于训练阶段的d

        # Encoder
        if code == 'polar':
            x_test = np.zeros((test_batch, N), dtype=bool)
            u_test = np.zeros((test_batch, N), dtype=bool)
            u_test[:, A] = d_test

            for iii in range(0, test_batch):
                x_test[iii] = polar_transform_iter(u_test[iii])

        elif code == 'random':
            x_test = np.zeros((test_batch, N), dtype=bool)#类似于训练阶段的x
            for iii in range(0, test_batch):
                x_test[iii] = x[bin2int(d_test[iii])]#https://www.mathworks.com/matlabcentral/fileexchange/56478-int2bin-bin2int
                #随机编码的实质是把x作为一个随机密码本，每一行的16个bit随机产生，然后把每一行对应的索引转化为二进制，这个二进制就是信息，
                #8bit，也就是(K,N)码的编码信息。
        # Modulator (BPSK)
        # s_test = -2 * x_test + 1#s_test为调制之后的信号，将发送至信道。，这里注释掉，因为总的模型输入是开始的信息，而不是调制之后的信号

        # Channel (AWGN)
        # y_test = s_test + sigmas[i] * np.random.standard_normal(s_test.shape)
        # y_test = s_test #测试一下没有噪声时系统的识别率是多少，看看模型正不正确。
        y_test = x_test  # 测试一下没有噪声时系统的识别率是多少，看看模型正不正确。
        y_test = x_test + 0*sigmas[i] * np.random.standard_normal(x_test.shape)#这里有噪声

        # if LLR:
        #     y_test = 2 * y_test / (sigmas[i] ** 2)#参见论文公式5，sigmas的意义看来是噪声功率。，这里也注释掉，不考虑噪声

        # Decoder
        # nb_errors[i] += decoder.evaluate(y_test, d_test, batch_size=test_batch, verbose=0)[1]#模型不再是decoder，而是总模型networkcoding_decoding_model，也可以运行，也是截取一部分。这个好像和之前的训练没有关系，因为没有导入训练的文件的过程阿。所以是0
        nb_errors[i] += networkcoding_decoding_model.evaluate(y_test, d_test, batch_size=test_batch, verbose=0)[1]
        #1，利用前面训练函数“history = model.fit()”产生的训练网络进行预测，同时直接将评价结果给出。本函数进去d_test（每行8bit），
        #   出来一个对应的码字（16bit），记为y_test‘，将之和y_test对比，就出测试的结果了。 2022.9.11。修改，应该进的是y_test，出来的是d_test，然后d_test和标签（原始被传输信息）作对比。输入输出维度可以再decoder这个类断点调试看。

        #2，nb_errors衡量的具体是什么，跟decoder.evaluate的具体定义有关系，但维度大致是20*m，m应该是每个信噪比（一共20个）对应的总误差，
        #   从后面BER=nb_errors/nb_bits来看，应该是按bit累计的误差，具体内容不是很重要。每次最内层的循环，针对每个i，每次最多增加batch_size
        #   个，也就是1个错误就是16bit一个消息出错了，针对每个i，内层总的错误最多最极端为num_words个，涉及的信息的bit为num_words*8个
        #3，keras看来比tensorflow更简单，很多细节都直接诶封装了，不过这样让人怀疑，这个结果真的好使吗？有点惊心动魄，关键就看着
        #   这一步了，如果不行，那就是方案不行了。不知道网络编码那个思路直接代进来会怎么样？
        #2022.9.11。 train是用到的模型是model_layers = modulator_layers + noise_layers + llr_layers + decoder_layers。test时用到的是 decoder_layers。测试时的信号经过信道已经调制modulator_layers和经过了噪声noise_layers

        # 2022.12.15日，之前注释的有点糊涂，为什么 nb_errors[i] 都是0呢？是不是networkcoding_decoding_model.evaluate方法不好使，所以弄了下面的计算方法？networkcoding_decoding_model就是总模型阿，不是单纯网络编码模型
        # 应该不是0才对阿。那后面的图形就不用“nb_errors/nb_bits”，而得用“number_error = len(np.nonzero(error)[0])/(len(error)*8)”吧？不知道想的对不对，时间很长了，又看了3天才行。
        # 正常来讲这个语句networkcoding_decoding_model.evaluate肯定不行，因为压根没有和前面的训练语句练习即来，没有加载阿，在原始程序里，先history = model.fit(x, d, batch_size=batch_size, epochs=epochs, verbose=0, shuffle=True)，然后
        #然后nb_errors[i] += decoder.evaluate(y_test, d_test, batch_size=test_batch, verbose=0)[1]，model的后半部分就是decoder


        # nb_errors[i] += model.evaluate(y_test, d_test, batch_size=test_batch, verbose=0)[1]
        # 下面的错误测试过程就和训练的结果联系起来了，
        networkcoding_decoding_model.load_weights('models/model_100')
        decoded = networkcoding_decoding_model.predict(y_test)#这里不考虑噪声
        decoded = np.round(decoded)#就近取整
        error = decoded - d_test
        number_error = len(np.nonzero(error)[0])/(len(error)*8)
        print(number_error)
        nb_bits[i] += d_test.size
        #d_test为1000*8，也就是统计一次测试多少bit，为下面计算BER做准备。对于每个i，一次8000个，弄np.round(num_words / test_batch).astype(int)遍
        #其实总共参与的bit数目为num_words*8个。


#Load MAP
result_map = np.loadtxt('map/{}/results_{}_map_{}_{}.txt'.format(code,code,N,k), delimiter=', ')
#这是由其他方法得出来的数据，作者没有给出githubu的源码，应该不难，是通信仿真里的东西，不是这篇论文的重点。
#想通过这个角度观看polar码的信噪比和噪声的关系看来是不行了。
sigmas_map = result_map[:,0]
nb_bits_map = result_map[:,1]
nb_errors_map = result_map[:,2]


#Plot Bit-Error-Rate
legend = []

plt.plot(10*np.log10(1/(2*sigmas**2)) - 10*np.log10(k/N), nb_errors/nb_bits)
#以“10*np.log10(1/(2*sigmas**2)) - 10*np.log10(k/N)”为横轴，意为信噪比E_b/N_0，以“nb_errors/nb_bits”为竖轴，意为BER。
#nb_errors/nb_bits：num_words个码字（字符）里，错误个数为nb_errors，总的bit数目为nb_bits，就是bit错误率BER，衡量的码的解码能力。
legend.append('NN')

plt.plot(10*np.log10(1/(2*sigmas_map**2)) - 10*np.log10(k/N), nb_errors_map/nb_bits_map)
legend.append('MAP')

plt.legend(legend, loc=3)
plt.yscale('log')
plt.xlabel('$E_b/N_0$')
plt.ylabel('BER')
plt.grid(True)
plt.show()
plot_model(networkcoding_decoding_model,to_file='model.png',show_shapes=True,show_layer_names=False)#网络结构可视化语句
#Image('model.png')
