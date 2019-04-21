import tensorflow as tf
from .modules import casual_layer,post,res_block,dilated_stack
from utils.mylogger import log
from keras import backend as K
from .base_model import _learning_rate_decay,batch_wer
from tensorflow.python.ops import ctc_ops as ctc

class ASR_wavnet(object):
    def __init__(self,hparams,name=None):
        super().__init__()
        self._hparams = hparams
        self.name = name

    def build_graph(self):
        '''
        placeholders：
            inputs : [batch_size, max_frames_in_current_batch, n-mfcc]
            labels : [batch_size, max_length_in_current_batch]
            label_lengths : [batch_size]
            input_lengths : [batch_size]
        '''
        with tf.variable_scope('NET') as scope:
            with tf.variable_scope('NET_Input') as scope:
                self.inputs = tf.placeholder(tf.float32,[None, None, self._hparams.num_mfccs], 'mfcc_inputs')
            is_training = self._hparams.is_training
            dim = self._hparams.wavnet_filters
            output_size = self._hparams.final_output_dim

            # TODO: Net architecture
            # casual [batch_size, max_frames_in_current_batch, dim]
            casual = casual_layer(inputs=self.inputs,filters=dim,
                                  is_training=is_training)

            # skip_tensor [batch_size, max_frames_in_current_batch,dim]
            skip_tensor = dilated_stack(inputs=casual,num_blocks=3,
                                is_training=is_training,dim=dim)

            # logits  [batch_size, max_frames_in_current_batch, dim]
            # pred  [batch_size, max_frames_in_current_batch, output_size]
            logits, pred = post(inputs=skip_tensor,dim=dim,
                                is_training=is_training,output_size=output_size)

            # TODO: Set attr and log info
            self.pred_logits = tf.identity(logits,name='pred_logits')
            self.pred_softmax = tf.identity(pred,name = 'pred_softmax')
            self.pred_labels = tf.identity(tf.argmax(pred,2),name='pred_labels')
            log('Initialized ASR_wavnet model. Dimensions: ')
            log('  casual:      ' + ''.join(str(casual.shape)))
            log('  skip_tensor: ' + ''.join(str(skip_tensor.shape)))
            log('  logits:      ' + ''.join(str(logits.shape)))
            log('  pred:        ' + ''.join(str(pred.shape)))

    def add_loss(self):
        with tf.variable_scope('loss') as scope:
            with tf.variable_scope('CTC_Input') as scope:
                self.labels = tf.placeholder(tf.int32, [None, None], 'labels')
                self.input_lengths = tf.placeholder(tf.int32, [None, 1], 'input_lengths') # 刚开始忘了写 1，表示batch_size*1的tensor
                self.label_lengths = tf.placeholder(tf.int32, [None, 1], 'label_lengths')

            self.ctc_loss = K.ctc_batch_cost(y_true=self.labels,y_pred=self.pred_softmax
                                         ,input_length=self.input_lengths,label_length=self.label_lengths)
            self.batch_loss = tf.reduce_mean(self.ctc_loss,name='batch_loss')

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
            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)

            # TODO: Compute gradients
            gradients, variables = zip(*optimizer.compute_gradients(self.ctc_loss))
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

            self.decoded, self.log_probabilities = K.ctc_decode(y_pred=self.pred_softmax,input_length=tf.squeeze(self.input_lengths,squeeze_dims=-1)) # input_length=不能从logits里直接获得

            self.decoded1 = tf.convert_to_tensor(self.decoded,dtype=tf.int32,name='decoded_labels')

            self.decoded2 = tf.squeeze(tf.identity(self.decoded),squeeze_dims=0)

            self.WER = batch_wer(self.labels,self.decoded2,self.input_lengths,self.label_lengths)