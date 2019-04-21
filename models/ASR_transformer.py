import tensorflow as tf
from keras import regularizers
from keras.layers import Softmax
# noinspection PyPep8Naming
from keras import backend as K

from utils.mylogger import log
from .base_model import Base_Model,_learning_rate_decay
from modulesLib.extras import ReusableEmbedding, TiedOutputEmbedding
from modulesLib.position import TransformerCoordinateEmbedding
from modulesLib.transformer import TransformerACT, TransformerBlock

class ASR_transformer(Base_Model):
    """
        A model which is similar to the one described by OpenAI in paper
        "Improving Language Understanding by Generative Pre-Training", except
        that it relies L2 regularization of the word embedding matrix
        (instead of the dropout), and uses Universal Transformer architecture.
    """
    def __init__(self,hparams,name='ASR_transformer'):
        super().__init__(hparams,name)

    def build_graph(self):
        hp = self._hparams

        with tf.variable_scope('NET') as scope:
            with tf.variable_scope('NET_input') as scopes:
                self.word_ids = tf.placeholder(dtype=tf.int32,shape=[None,hp.max_seq_length,],name='input_word_id')

            # TODO: init
            l2_regularizer = (regularizers.l2(hp.l2_reg_penalty) if hp.l2_reg_penalty
                              else None)
            embedding_layer = ReusableEmbedding(
                hp.input_vocab_size, hp.word_embedding_size,
                input_length=hp.max_seq_length,
                name='bpe_embeddings',
                # Regularization is based on paper "A Comparative Study on
                # Regularization Strategies for Embedding-based Neural Networks"
                # https://arxiv.org/pdf/1508.03721.pdf
                embeddings_regularizer=l2_regularizer)
            output_layer = TiedOutputEmbedding(
                projection_regularizer=l2_regularizer,
                projection_dropout=hp.embedding_dropout,
                name='word_prediction_logits')
            coordinate_embedding_layer = TransformerCoordinateEmbedding(
                hp.transformer_depth,
                name='coordinate_embedding')
            transformer_act_layer = TransformerACT(name='adaptive_computation_time')
            transformer_block = TransformerBlock(
                name='transformer', num_heads=hp.num_heads,
                residual_dropout=hp.transformer_dropout,
                attention_dropout=hp.transformer_dropout,
                use_masking=True, vanilla_wiring=False)
            output_softmax_layer = Softmax(name='word_predictions')

            # TODO: call
            next_step_input, embedding_matrix = embedding_layer(self.word_ids)
            act_output = next_step_input

            for i in range(hp.transformer_depth):
                next_step_input = coordinate_embedding_layer(next_step_input, step=i)
                next_step_input = transformer_block(next_step_input)
                next_step_input, act_output = transformer_act_layer(next_step_input)

            transformer_act_layer.finalize()
            next_step_input = act_output
            word_predictions = output_softmax_layer(
                output_layer([next_step_input, embedding_matrix]))

            self.output_softmax = tf.identity(word_predictions,name='output_softmax')

    def add_loss(self):
        hp = self._hparams
        with tf.variable_scope('loss') as scope:
            # Penalty for confidence of the output distribution, as described in
            # "Regularizing Neural Networks by Penalizing Confident
            # Output Distributions" (https://arxiv.org/abs/1701.06548)
            self.confidence_penalty = K.mean(
                hp.confidence_penalty_weight *
                K.sum(self.output_softmax * K.log(self.output_softmax), axis=-1))
