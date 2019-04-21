import tensorflow as tf
from keras import backend as K

class Base_Model(object):
    def __init__(self,hparams,name=None):
        super().__init__()
        self._hparams = hparams
        self.name = name

    def build_graph(self):
        raise NotImplementedError()

    def add_loss(self):
        raise NotImplementedError()

    def add_optimizer(self, global_step,loss):
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
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            self.gradients = gradients

            # TODO: Clip gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

            # TODO: Apply gradients
            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)


def _learning_rate_decay(init_lr, global_step):
  # Noam scheme from tensor2tensor:
  warmup_steps = 4000.0
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)

def batch_wer(y_true, y_pred, input_length, label_length):
    """Runs CTC loss algorithm on each batch element.

    # Arguments
        y_true: tensor `(samples, max_string_length)`
            containing the truth labels.
        y_pred: tensor `(samples, time_steps, num_categories)` (samples, max_string_length)
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_pred`.
        label_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_true`.

    # Returns

    """
    label_length = tf.to_int32(tf.squeeze(label_length, axis=-1))
    input_length = tf.to_int32(tf.squeeze(input_length, axis=-1))
    sparse_labels = tf.to_int32(K.ctc_label_dense_to_sparse(y_true, label_length))
    sparse_pred = tf.to_int32(K.ctc_label_dense_to_sparse(y_pred, input_length))

    WER = tf.reduce_mean(tf.edit_distance(sparse_pred,sparse_labels,normalize=True))
    return  WER