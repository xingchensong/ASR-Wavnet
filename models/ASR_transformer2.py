from utils.mylogger import log
from .base_model import Base_Model,_learning_rate_decay
from .modules import normalize,embedding,multihead_attention,feedforward,label_smoothing
import tensorflow as tf

class ASR_transformer2(Base_Model):

    def __init__(self,hparams,name='ASR_transformer2',is_training = True):
        super().__init__(hparams,name)
        self.is_training = is_training

    def build_graph(self):
        hp = self._hparams
        with tf.variable_scope("NET") :
            with tf.variable_scope("NET_input") :
                # input placeholder
                self.x = tf.placeholder(tf.int32, shape=(None, None))
                self.y = tf.placeholder(tf.int32, shape=(None, None))
                self.de_inp = tf.placeholder(tf.int32, shape=(None, None))

            # Encoder
            with tf.variable_scope("encoder"):
                # embedding
                self.en_emb = embedding(self.x, vocab_size=hp.input_vocab_size, num_units=hp.hidden_units,
                                        scale=True, scope="enc_embed")
                self.enc = self.en_emb + embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                    vocab_size=hp.max_seq_length, num_units=hp.hidden_units, zero_pad=False, scale=False,
                    scope="enc_pe")
                ## Dropout
                self.enc = tf.layers.dropout(self.enc,
                                             rate=hp.embedding_ropout,
                                             training=tf.convert_to_tensor(self.is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(key_emb=self.en_emb,
                                                       que_emb=self.en_emb,
                                                       queries=self.enc,
                                                       keys=self.enc,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.transformer_dropout,
                                                       is_training=self.is_training,
                                                       causality=False)

                ### Feed Forward
                self.enc = feedforward(self.enc, num_units=[4 * hp.hidden_units, hp.hidden_units])

            # Decoder
            with tf.variable_scope("decoder"):
                # embedding
                self.de_emb = embedding(self.de_inp, vocab_size=hp.label_vocab_size, num_units=hp.hidden_units,
                                        scale=True, scope="dec_embed")
                self.dec = self.de_emb + embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.de_inp)[1]), 0), [tf.shape(self.de_inp)[0], 1]),
                    vocab_size=hp.max_length, num_units=hp.hidden_units, zero_pad=False, scale=False,
                    scope="dec_pe")
                ## Dropout
                self.dec = tf.layers.dropout(self.dec,
                                             rate=hp.embedding_dropout,
                                             training=tf.convert_to_tensor(self.is_training))

                ## Multihead Attention ( self-attention)
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.dec = multihead_attention(key_emb=self.de_emb,
                                                       que_emb=self.de_emb,
                                                       queries=self.dec,
                                                       keys=self.dec,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.transformer_dropout,
                                                       is_training=self.is_training,
                                                       causality=True,
                                                       scope='self_attention')

                ## Multihead Attention ( vanilla attention)
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.dec = multihead_attention(key_emb=self.en_emb,
                                                       que_emb=self.de_emb,
                                                       queries=self.dec,
                                                       keys=self.enc,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.transformer_dropout,
                                                       is_training=self.is_training,
                                                       causality=True,
                                                       scope='vanilla_attention')

                        ### Feed Forward
                self.outputs = feedforward(self.dec, num_units=[4 * hp.hidden_units, hp.hidden_units])

            # Final linear projection
            self.logits = tf.layers.dense(self.outputs, hp.label_vocab_size)
            self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (
                       tf.reduce_sum(self.istarget))



    def add_loss(self):
        hp = self._hparams
        with tf.variable_scope('loss'):
            # Loss
            self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=hp.label_vocab_size))
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
            self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

