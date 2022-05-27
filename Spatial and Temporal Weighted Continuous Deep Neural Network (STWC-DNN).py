import tensorflow as tf
import numpy as np
import os
import time
from scipy.stats import pearsonr
import random
from read_data2 import read_txt
from labeltenfold import read_data
from skimage import io
from readtimestationbased import readtxt2, readsampledata
from tensorflow.python.training.moving_averages import assign_moving_average

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cucnum = 11
batch_size = 200




def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device("/cpu:0"):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def maskconv2d(inputs, mask,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           maskoutputs = None):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      assert(data_format=='NHWC' or data_format=='NCHW')
      if data_format == 'NHWC':
        num_in_channels = inputs.get_shape()[-1].value
      elif data_format=='NCHW':
        num_in_channels = inputs.get_shape()[1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      net = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding,
                             data_format=data_format)


      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(net, biases, data_format=data_format)


      if padding == 'SAME':
          if bn:
              outputs = batch_norm_mine(outputs, tf.squeeze(mask, -1), 'bn', is_training)
      else:
          if bn:
              outputs = tf.squeeze(outputs)
              outputs = batch_norm_mine2(outputs, None,    'bn', is_training)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs, maskoutputs




def batch_norm_mine2(inputs,
                    mask,
                    scope,
                    is_training=True,
                    epsilon=1e-3,
                    decay = 0.9
                    ):
    """

    :param inputs: 一个K-D的tensor
    :param mask: 一个(K-1)-D的tensor
    :param scope:
    :param is_training:
    :param epsilon:
    :param decay: 滑动平均的衰减指数
    :return:
    """

    with tf.variable_scope(scope):
        shape = inputs.get_shape().as_list()


        moving_mean = tf.get_variable("moving_mean", shape[-1], initializer=tf.constant_initializer(0.0),
                                     trainable=False)

        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0),
                                     trainable=False)


        def mean_and_var_update():


            axes = list(range(len(inputs.get_shape()) - 1))
            batch_mean, batch_var = tf.nn.moments(inputs, axes, name="moments")  # [depth]

            with tf.control_dependencies([assign_moving_average(moving_mean, batch_mean, decay),
                                          assign_moving_average(moving_var, batch_var, decay)]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, variance = tf.cond(tf.cast(is_training, tf.bool), mean_and_var_update, lambda:(moving_mean, moving_var))



        gamma = tf.get_variable("scale", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        beta = tf.get_variable("shift", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)

        return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)



def batch_norm_mine(inputs,
                    mask,
                    scope,
                    is_training=True,
                    epsilon=1e-3,
                    decay = 0.9
                    ):
    """

    :param inputs: 一个K-D的tensor
    :param mask: 一个(K-1)-D的tensor
    :param scope:
    :param is_training:
    :param epsilon:
    :param decay: 滑动平均的衰减指数
    :return:
    """

    with tf.variable_scope(scope):
        shape = inputs.get_shape().as_list()


        moving_mean = tf.get_variable("moving_mean", shape[-1], initializer=tf.constant_initializer(0.0),
                                     trainable=False)

        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0),
                                     trainable=False)

        def mean_and_var_update():
            valid_indices = tf.to_int32(mask>0)
            valid_input = tf.dynamic_partition(inputs, valid_indices, num_partitions=2)[1]
            axes = list(range(len(valid_input.get_shape()) - 1))

            batch_mean, batch_var = tf.nn.moments(valid_input, axes, name="moments")  # [depth]

            with tf.control_dependencies([assign_moving_average(moving_mean, batch_mean, decay),
                                          assign_moving_average(moving_var, batch_var, decay)]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, variance = tf.cond(tf.cast(is_training, tf.bool), mean_and_var_update, lambda:(moving_mean, moving_var))


        gamma = tf.get_variable("scale", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        beta = tf.get_variable("shift", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)

        return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)



def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=None,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
  """ Fully connected layer with non-linear operation.

  Args:
    inputs: 2-D tensor BxN
    num_outputs: int

  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.variable_scope(scope) as sc:
    num_input_units = inputs.get_shape()[-1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_units, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)
    biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)

    if bn:
      outputs = batch_norm_mine2(outputs, None,    'bn', is_training)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs



def singlepoint( pointself, is_training, bn_decay=None):

    pointself = tf.squeeze(pointself)

    with tf.variable_scope('single'):
        fet =  fully_connected(pointself, 64, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)

        fet =  fully_connected(fet, 256, bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)

        fet =  fully_connected(fet, 256, bn=True, is_training=is_training,
                                      scope='fc3', bn_decay=bn_decay)

        fet =  fully_connected(fet, 256, bn=True, is_training=is_training,
                                      scope='fc4', bn_decay=bn_decay)

    return fet




def wiru(pmc, pointselfxyz,  coxyz,  connectmasksample,bn_decay= None):
    pointselfxyz = tf.expand_dims(pointselfxyz, 1)

    pmc = tf.reshape(pmc, [batch_size, -1, 1])
    connectmasksample = tf.reshape(connectmasksample, [batch_size, -1, 1, 1])
    coxyz = tf.transpose(coxyz, [0, 1, 3, 2])
    coxyz = tf.reshape(coxyz, [batch_size, -1, 14])
    coxyt2 = pointselfxyz - coxyz
    coxyt2 = tf.concat([coxyt2, pmc], 2)

    coxyt2 = tf.expand_dims(coxyt2, 2)
    conv1, maskmaskout = myconv(coxyt2, connectmasksample, None, 32, is_training, bn_decay, kernel_size=1,
                                scope='conv1')
    conv11, _ = myconv(conv1, connectmasksample, None, 64, is_training, bn_decay, kernel_size=1, scope='conv11')

    conv2, _ = myconv(conv11, connectmasksample, None, 256, is_training, bn_decay, kernel_size=1, scope='conv2')

    conv21, _ = myconv(conv2, connectmasksample, None, 256, is_training, bn_decay, kernel_size=1, scope='conv21')

    conv21 = conv21 * connectmasksample
    conv21 = tf.reduce_sum(conv21, 1)
    numsc = tf.reduce_sum(connectmasksample, 1) + 0.0000001
    coxyt4 = conv21 / numsc
    coxyt4 = tf.squeeze(coxyt4)
    return coxyt4, coxyt4

def centroll(bat1, mbat1):
    pool1 = tf.nn.avg_pool(bat1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")
    mask1 = tf.nn.avg_pool(mbat1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")
    cent1 = pool1 / (mask1 + 0.0000001)
    return  cent1

def myconv(inputs, mask, maskace, output_channel, is_training, bn_decay, padding = 'SAME' , bn = True,  kernel_size=3, stride=1, activation=tf.nn.relu, scope=None):
    with tf.variable_scope(scope):

        net, maskout =  maskconv2d(inputs, mask ,  output_channel, [kernel_size, kernel_size],
                                padding= padding, stride=[stride, stride],
                                bn=bn, is_training=is_training,  activation_fn=activation,
                                scope= 'scope', bn_decay=bn_decay,maskoutputs = maskace)

    return net, maskout



def mapsllkd( pointself, mask, is_training, bn_decay=None):
    pointself = pointself * mask
    bat1 = tf.slice(pointself, [0, 0, 0, 0], [200, 8, 8, cucnum])
    bat2 = tf.slice(pointself, [0, 0, 4, 0], [200, 8, 8, cucnum])
    bat3 = tf.slice(pointself, [0, 0, 8, 0], [200, 8, 8, cucnum])
    bat4 = tf.slice(pointself, [0, 4, 0, 0], [200, 8, 8, cucnum])
    bat5 = tf.slice(pointself, [0, 4, 4, 0], [200, 8, 8, cucnum])
    bat6 = tf.slice(pointself, [0, 4, 8, 0], [200, 8, 8, cucnum])
    bat7 = tf.slice(pointself, [0, 8, 0, 0], [200, 8, 8, cucnum])
    bat8 = tf.slice(pointself, [0, 8, 4, 0], [200, 8, 8, cucnum])
    bat9 = tf.slice(pointself, [0, 8, 8, 0], [200, 8, 8, cucnum])

    mbat1 = tf.slice(mask, [0, 0, 0, 0], [200, 8, 8, 1])
    mbat2 = tf.slice(mask, [0, 0, 4, 0], [200, 8, 8, 1])
    mbat3 = tf.slice(mask, [0, 0, 8, 0], [200, 8, 8, 1])
    mbat4 = tf.slice(mask, [0, 4, 0, 0], [200, 8, 8, 1])
    mbat5 = tf.slice(mask, [0, 4, 4, 0], [200, 8, 8, 1])
    mbat6 = tf.slice(mask, [0, 4, 8, 0], [200, 8, 8, 1])
    mbat7 = tf.slice(mask, [0, 8, 0, 0], [200, 8, 8, 1])
    mbat8 = tf.slice(mask, [0, 8, 4, 0], [200, 8, 8, 1])
    mbat9 = tf.slice(mask, [0, 8, 8, 0], [200, 8, 8, 1])

    cent1 = centroll(bat1, mbat1)
    cent2 = centroll(bat2, mbat2)
    cent3 = centroll(bat3, mbat3)
    cent4 = centroll(bat4, mbat4)

    cent6 = centroll(bat6, mbat6)
    cent7 = centroll(bat7, mbat7)
    cent8 = centroll(bat8, mbat8)
    cent9 = centroll(bat9, mbat9)

    cent5real = tf.slice(pointself, [0, 8, 8, 0], [200, 1, 1, cucnum])

    bat1c = bat1 - cent1
    bat2c = bat2 - cent2
    bat3c = bat3 - cent3
    bat4c = bat4 - cent4
    bat5c = bat5 - cent5real
    bat6c = bat6 - cent6
    bat7c = bat7 - cent7
    bat8c = bat8 - cent8
    bat9c = bat9 - cent9

    ceng1c = tf.concat([bat1c, bat2c, bat3c], 2)
    ceng2c = tf.concat([bat4c, bat5c, bat6c], 2)
    ceng3c = tf.concat([bat7c, bat8c, bat9c], 2)
    cengc = tf.concat([ceng1c, ceng2c, ceng3c], 1)

    mceng1 = tf.concat([mbat1, mbat2, mbat3], 2)
    mceng2 = tf.concat([mbat4, mbat5, mbat6], 2)
    mceng3 = tf.concat([mbat7, mbat8, mbat9], 2)
    mceng = tf.concat([mceng1, mceng2, mceng3], 1)


    conv1c, maskmaskoutc = myconv(cengc, mceng, None, 32, is_training, bn_decay, kernel_size=1, scope='conv1c')
    conv11c, _ = myconv(cengc, mceng, maskmaskoutc, 64, is_training, bn_decay, kernel_size=1, scope='conv11c')
    conv12c = conv11c * mceng
    pool1c = tf.nn.avg_pool(conv12c, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

    mask1 = tf.nn.avg_pool(mceng, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")
    mask12 = tf.nn.max_pool(mceng, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

    pool1c = pool1c / (mask1 + 0.0000001)

    pool1 =  pool1c * mask12

    conv2, maskmaskout1 = myconv(pool1, mask1, None, 256, is_training, bn_decay, kernel_size=1, scope='conv2')

    conv21, _ = myconv(conv2, mask1, maskmaskout1, 256, is_training, bn_decay, kernel_size=1, scope='conv21')

    conv22 = conv21 * mask12
    pool2 = tf.nn.avg_pool(conv22, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding="SAME")
    mask2 = tf.nn.avg_pool(mask12, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding="SAME")

    pool2 = pool2 / (mask2 + 0.0000001)

    conv5 = tf.squeeze(pool2)
    return conv5



def wiate(fet,is_training, bn_decay):
    fetavg = tf.reduce_mean(fet, 1)
    fetmax = tf.reduce_max(fet, 1)
    fetmin = tf.reduce_min(fet, 1)
    fetavg = tf.expand_dims(fetavg, -1)
    fetmax = tf.expand_dims(fetmax, -1)
    fetmin = tf.expand_dims(fetmin, -1)
    fet2 = fet - fetavg
    fet2 = fet2 * fet2
    fet2avg = tf.reduce_mean(fet2, 1)
    fet2avg = tf.expand_dims(fet2avg, -1)
    at2 = tf.concat([fetavg,fetmax,fetmin,fet2avg], 1)
    wei1 =  fully_connected(at2, 256, bn=True, is_training=is_training,
                                   scope='fc1', bn_decay=bn_decay, activation_fn=tf.nn.sigmoid)
    return wei1



def mapsltkl(pointself, mask, is_training, bn_decay=None):


    pointself = tf.expand_dims(pointself, 2)
    mask = tf.expand_dims(mask, -1)
    pointself = pointself * mask

    conv1, maskmaskout = myconv(pointself, mask, None, 32, is_training, bn_decay, kernel_size=1,
                                scope='conv1')
    conv11, _ = myconv(conv1, mask, None, 64, is_training, bn_decay, kernel_size=1, scope='conv11')

    conv2, _ = myconv(conv11, mask, None, 256, is_training, bn_decay, kernel_size=1, scope='conv2')

    conv21, _ = myconv(conv2, mask, None, 256, is_training, bn_decay, kernel_size=1, scope='conv21')

    conv22, _ = myconv(conv21, mask, None, 256, is_training, bn_decay, kernel_size=1, scope='conv22')

    conv22 = conv22 * mask
    pool2 = tf.nn.avg_pool(conv22, ksize=[1, 5, 1, 1], strides=[1, 5, 1, 1], padding="SAME")
    mask2 = tf.nn.avg_pool(mask, ksize=[1, 5, 1, 1], strides=[1, 5, 1, 1], padding="SAME")

    pool2 = pool2 / (mask2 + 0.0000001)

    conv5 = tf.squeeze(pool2)
    return conv5

def get_model(pointself, pointselflabel, pointselfxyz, co1,  co1pmtrue, coxyz, maskconnectsample,
              windpre1,   windmask,
              is_training, bn_decay=None):

    pointpara = tf.concat([pointself, pointselfxyz], 1)
    copara = tf.concat([co1, coxyz], 2)
    weiinggeo, coxyt3 = wiru(co1pmtrue, pointpara,  copara, maskconnectsample)

    fet = singlepoint(pointself,  is_training)

    windal = windpre1

    with tf.variable_scope('geo'):
        fet2g = mapsllkd(windal, windmask, is_training)


    with tf.variable_scope('a1'):
        wfet = wiate(fet, is_training, bn_decay)


    fet = wfet * fet2g + wfet * weiinggeo + fet
    fet3 =  fully_connected(fet, 256, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)

    pself =  fully_connected(fet3, 1, bn=False, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay, activation_fn=tf.nn.sigmoid)

    pself = tf.squeeze(pself)
    losssaround4 = tf.contrib.layers.l2_regularizer(1.0)(pointselflabel - pself)
    return pself, losssaround4, coxyt3




if __name__ == "__main__":


    test_batch_size = batch_size
    base_learning_rate = 0.0001
    epoch = 3000

    # selffeature
    pointself = tf.placeholder(tf.float32, [batch_size,  11], 'pointround')
    pointselflabel = tf.placeholder(tf.float32, [batch_size], 'pointself')
    # selfposition+time
    pointselfxyz = tf.placeholder(tf.float32, [batch_size, 3], 'pointselfxyz')

    # stationfeatrue in 5 hour
    co1 = tf.placeholder(tf.float32, [batch_size, 39, 11, 5], 'co1')
    # stationposition in 5 hour
    coxyz = tf.placeholder(tf.float32, [batch_size, 39, 3, 5], 'coxyz')
    # stationpm25 in 5 hour
    co1pmtrue = tf.placeholder(tf.float32, [batch_size, 39, 1, 5], 'co1pmtrue')
    # mask the test stations and missing data
    maskconnectsample = tf.placeholder(tf.float32, [batch_size, 39, 5], 'maskconnectsample')
    # regionfeature
    windpre1 = tf.placeholder(tf.float32, [batch_size, 16, 16, 11], 'windpre1')
    # mask the missing data
    windmask = tf.placeholder(tf.float32, [batch_size, 16, 16, 1], 'windmask')


    is_training = tf.placeholder(tf.bool, [], 'is_training')
    global_step = tf.placeholder(tf.int32, [], 'global_step')

    net_all1,  loss1, coxytdy = get_model( pointself, pointselflabel, pointselfxyz,co1,  co1pmtrue, coxyz,maskconnectsample,
                                           windpre1,   windmask,
                                           is_training, bn_decay=None)


    loss = loss1

    base_lr = tf.constant(base_learning_rate, tf.float64)
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - global_step / epoch), 0.9))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op1 = optimizer.minimize(loss)

    ops = {'train_op1': train_op1,
           'loss1': loss1,
           'output_y1': net_all1,

           'co1': co1,
           'coxyz': coxyz,
           'maskconnectsample': maskconnectsample,
           'co1pmtrue': co1pmtrue,

           'windpre1': windpre1,
           'windmask': windmask,

           'pointself': pointself,
           'pointselflabel': pointselflabel,
           'pointselfxyz': pointselfxyz,
           'is_training': is_training,
           'coxytdy': coxytdy,
           }




