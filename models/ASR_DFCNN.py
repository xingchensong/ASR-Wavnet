import tensorflow as tf
from .modules import Conv2dBlockWithMaxPool,post_net
from utils.mylogger import log
from keras import backend as K
from tensorflow.python.ops import ctc_ops as ctc
from .base_model import _learning_rate_decay,batch_wer,Base_Model

class ASR(Base_Model):
    def __init__(self,hparams,name='ASR'):
        super().__init__(hparams,name=name)
        self._hparams = hparams
        self.name = name

    def build_graph(self):
        '''
        placeholders：
            inputs : [batch_size, max_frames_in_current_batch, n-fft, 1]
            labels : [batch_size, max_length_in_current_batch]
            label_lengths : [batch_size]
            input_lengths : [batch_size]
        '''
        with tf.variable_scope('NET') as scope:
            with tf.variable_scope('NET_Input') as scope:
                self.inputs = tf.placeholder(tf.float32,[None, None, self._hparams.num_mels, 1], 'spec_inputs')
            is_training = self._hparams.is_training
            # print(tf.shape(inputs))
            block = Conv2dBlockWithMaxPool(num_conv=2)

            # TODO: Net architecture
            # block1  [batch_size, max_frames_in_current_batch/2, n-fft/2, 32]
            block1 = block(inputs=self.inputs,kernal_size=3,channels=32,pool_size=2,is_training=is_training,scope='block1',dropout=0.05)
            # block2  [batch_size, max_frames_in_current_batch/4, n-fft/4, 64]
            block2 = block(block1, 3, 64, 2, is_training, scope='block2', dropout=0.1)
            # block3  [batch_size, max_frames_in_current_batch/8, n-fft/8, 128]
            block3 = block(block2, 3, 128, 2, is_training, scope='block3', dropout=0.15)
            # block4  [batch_size, max_frames_in_current_batch/8, n-fft/8, 128]
            block4 = block(block3, 3, 128, 1, is_training, scope='block4', dropout=0.2)
            # block5  [batch_size, max_frames_in_current_batch/8, n-fft/8 = 50, 128]
            block5 = block(block4, 3, 128, 1, is_training, scope='block5', dropout=None)
            # post-net  [batch_size, max_frames_in_current_batch/8, hparams.final_output_dim ]
            y_pred = post_net(block5,is_training)
            self.y_pred2 = tf.argmax(y_pred,2)

            # TODO: Set attr and log info
            self.pred_logits = tf.identity(y_pred,name='pred_logits')
            log('Initialized ASR model. Dimensions: ')
            log('  block1:      ' + ''.join(str(block1.shape)))
            log('  block2:      ' + ''.join(str(block2.shape)))
            log('  block3:      ' + ''.join(str(block3.shape)))
            log('  block4:      ' + ''.join(str(block4.shape)))
            log('  block5:      ' + ''.join(str(block5.shape)))
            log('  postnet out: ' + ''.join(str(y_pred.shape)))

    def add_loss(self):
        with tf.variable_scope('loss') as scope:
            with tf.variable_scope('CTC_Input') as scope:
                self.labels = tf.placeholder(tf.int32, [None, None], 'labels')
                self.input_lengths = tf.placeholder(tf.int32, [None, 1], 'input_lengths') # 刚开始忘了写 1，表示batch_size*1的tensor
                self.label_lengths = tf.placeholder(tf.int32, [None, 1], 'label_lengths')
            # input_length实际上就是y_pred的中间那个维度，为什么不能直接取出来呢？可能构建图的时候不是定值（input的帧数那个维度是None，所以这里直接用的话也是没有值
            # 但是如果decode的时候实际可以通过直接取出来做的，因为decode传入的input是个实值不是placeholer那种tensor了
            # 注意input_length必须小于等于真实的label_length
            self.ctc_loss = K.ctc_batch_cost(y_true=self.labels,y_pred=self.pred_logits
                                         ,input_length=self.input_lengths,label_length=self.label_lengths)
            self.batch_loss = tf.reduce_mean(self.ctc_loss,name='batch_loss')
            # 直接使用tf.nn.ctc_loss 需要传入的labels是sparsetensor，使用keras的ctc会帮助你将dense转成sparse
            # self.ctc_loss = tf.nn.ctc_loss(labels=self.labels,inputs=self.pred_labels,
            #                                sequence_length=self.label_lengths,time_major=False)
            # return  [batch_size] 其中值为概率 P(Y | X)

    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
          global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer') as scope:
            hp = self._hparams
            # TODO: Learning rate decay
            if hp.decay_learning_rate:
                self.learning_rate = _learning_rate_decay(hp.initial_learning_rate, global_step)
                # self.learning_rate = tf.train.exponential_decay(hp.initial_learning_rate,global_step,hp.steps_per_epoch//4, 0.96, staircase=False)
            else:
                self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)

            # TODO: Set optimizer
            optimizer = tf.train.AdamOptimizer(self.learning_rate,hp.adam_beta1,hp.adam_beta2)
            # optimizer = tf.train.AdadeltaOptimizer()
            # TODO: Compute gradients
            gradients, variables = zip(*optimizer.compute_gradients(self.ctc_loss))
            '''
            optimizer.minimize()
            This method simply combines calls `compute_gradients()` and
            `apply_gradients()`. If you want to process the gradient before applying
            them call `compute_gradients()` and `apply_gradients()` explicitly instead
            of using this function.
            '''
            self.gradients = gradients

            # TODO: Clip gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

            # TODO: Apply gradients
            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)

    def add_decoder(self):
        '''Adds ctc_decoder to the model. Sets "decode" field. add_loss must have been called.'''
        with tf.variable_scope('decode') as scope:

            # input_length实际上可以直接取pred_labels的第一维？表示有多少帧
            self.decoded, self.log_probabilities = K.ctc_decode(y_pred=self.pred_logits,input_length=tf.squeeze(self.input_lengths)) # input_length=不能从logits里直接获得

            self.decoded2 = tf.squeeze(tf.identity(self.decoded,name='decoded_labels'))

            self.WER = batch_wer(self.labels,self.decoded2,self.input_lengths,self.label_lengths)
            # 如果没有这一句，那么tf.saved_model.utils.build_tensor_info(model.decoded)这里会报错
            # AttributeError: 'list' object has no attribute 'dtype'，因为decoded是一个tensor的list不是一个tensor
            # 这里用identity转成tensor


            # self.decoded, self.log_probabilities = tf.nn.ctc_beam_search_decoder(
            #     inputs=tf.transpose(self.pred_labels,perm=[1,0,2]),
            #     sequence_length=tf.reshape(tensor=self.label_lengths,shape=[-1]))
            # 注意这里self.label_lengths按照ctc_batch_loss的要求在placeholder里设置成了[None,1]二维tensor
            # 而在ctc_beam_search_decoder中需要reshape成[None]

