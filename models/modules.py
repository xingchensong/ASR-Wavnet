import tensorflow as tf
from six.moves import xrange
from hparams import hparams

################################ DFCnn ###################################################
class Conv2dBlockWithMaxPool(object):
    """Conv2d Block
    The output is max_pooled along time.
    """
    def __init__(self,num_conv,activation = tf.nn.relu):
        self.__num_conv = num_conv
        self.__activation = activation

    @property
    def num_conv(self):
        return self.__num_conv

    @property
    def activation(self):
        return self.__activation

    def __call__(self,inputs,kernal_size,channels,pool_size,is_training,activation=None,scope=None,dropout=None):
        """
        Args:
            inputs: with shape -> (batch_size, n-frames, n-fft)
        """

        if activation is not None:
            self.__activation = activation

        with tf.variable_scope(scope or type(self).__name__):
            for index in xrange(1,self.__num_conv+1):
                with tf.variable_scope('inner_conv_%d' % index):
                    conv_k = tf.layers.conv2d(inputs=inputs,filters=channels,kernel_size=kernal_size,
                                              padding='same',activation=self.activation,kernel_initializer='he_normal')
                    norm_k = tf.layers.batch_normalization(inputs=conv_k,training=is_training)

                    inputs = tf.identity(input=norm_k)

            maxpool_output = tf.layers.max_pooling2d(
                inputs=norm_k,
                pool_size=pool_size,
                strides=pool_size,
                padding='valid',
                name='max_pool'
            )
            if dropout is not None:
                maxpool_output = tf.layers.dropout(inputs=maxpool_output,rate=dropout,training=is_training,name='dropout')

            return maxpool_output

def post_net(inputs,is_training):
    with tf.variable_scope('post_net'):
        # 如果直接从inputs.shape[0]取出来batch-size会报错：（但是后面两个维度是固定的，可以直接取出）
        # TypeError: Failed to convert object of type <class 'list'> to Tensor. Contents: [Dimension(None), -1, Dimension(4096)]. Consider casting elements to a supported type.
        # 因为此时还没初始化，所以第零维度是None，现在只能把超参传进去
        x = tf.reshape(tensor=inputs, shape=[hparams.batch_size, -1 ,inputs.shape[-1]*inputs.shape[-2]])  # [batch_size, ?, 5, 128] --> [batch_size, ?, 5*128]
        x = tf.layers.dropout(x,0.3,training=is_training)
        x = tf.layers.dense(inputs=x,units=128,activation=tf.nn.relu,use_bias=True,kernel_initializer="he_normal")  # [batch_size, ?, 128]
        x = tf.layers.dropout(x, 0.3, training=is_training)
        x = tf.layers.dense(inputs=x, units=hparams.final_output_dim, activation=tf.nn.relu, use_bias=True, kernel_initializer="he_normal")  # [batch_size, ?, final_output_dim]
        # 这里?是指T_out(number of steps in the output time series) 和 n-frames(也就是T_in)不一定相同
        # y_pred = x
        y_pred = tf.nn.softmax(logits=x,axis=-1)  # tf.nn.ctc_loss说输入值不要softmax，但是keras的ctc应该要激活？？

        return y_pred


################################ WavNet ################################################
initializer = tf.contrib.layers.xavier_initializer()

def casual_layer(inputs,filters,is_training):
    with tf.variable_scope('casual_layer'):
        x = tf.layers.conv1d(inputs,filters=filters,kernel_size=1,
                             padding='same',name='casual_conv')
        x = tf.layers.batch_normalization(x,-1,training=is_training,name='casual_conv_bn')
        x = tf.keras.layers.Activation("tanh")(x)
        return x


def res_block(inputs,size,rate,block,dim,is_training):
    with tf.variable_scope('block_%d_%d'%(block, rate)):
        conv_filter = tf.layers.conv1d(inputs,filters=dim,kernel_size=size,padding='same',
                                       dilation_rate=rate,name='conv_filter')
        conv_filter = tf.layers.batch_normalization(conv_filter,training=is_training,name='conv_filter_bn',axis=-1)
        conv_filter = tf.keras.layers.Activation("tanh")(conv_filter)

        # keras.layers.Conv1D()
        conv_gate = tf.layers.conv1d(inputs,filters=dim,kernel_size=size,padding='same',
                                       dilation_rate=rate,name='conv_gate')
        conv_gate = tf.layers.batch_normalization(conv_gate, training=is_training, name='conv_gate_bn')
        conv_gate = tf.keras.layers.Activation('sigmoid')(conv_gate)

        out = tf.multiply(conv_filter,conv_gate,name='out')

        conv_out = tf.layers.conv1d(out,filters=dim,kernel_size=1,padding='same',
                                        name='conv_out')
        conv_out = tf.layers.batch_normalization(conv_out, training=is_training, name='conv_out_bn')
        conv_out = tf.keras.layers.Activation('tanh')(conv_out)

        residual = tf.add(inputs,conv_out,name='residual_out')
        return residual,conv_out

def dilated_stack(inputs,num_blocks,is_training,dim):
    with tf.variable_scope('dilated_stack'):
        skip = []
        res = tf.identity(inputs,name='res_input')
        for i in range(num_blocks):
            for r in [1,2,4,8,16]:
                res, s = res_block(res,size=7,rate=r,block=i,is_training=is_training,dim=dim)
                skip.append(s)
        ret = tf.keras.layers.Add()([s for s in skip])
        return ret

def post(inputs,dim,is_training,output_size):
    with tf.variable_scope('post_process'):
        logits = tf.layers.conv1d(inputs,filters=dim,kernel_size=1,
                                  padding='same',name='logits')
        logits = tf.layers.batch_normalization(logits,training=is_training,name='logits_bn')
        logits = tf.keras.layers.Activation('tanh')(logits)

        y_pred = tf.layers.conv1d(logits,filters=output_size,kernel_size=1,kernel_regularizer=tf.keras.regularizers.l2(0.2),
                                  padding='same',activation='softmax',name='y_pred')
        return logits, y_pred

############################## Transformer ############################################

def normalize(inputs,
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs

# TODO： 使用自己训练的emb而不是原文的pos emb
def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=None):
    '''Embeds a given tensor.
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]
     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]
     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs


def multihead_attention(key_emb,
                        que_emb,
                        queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(key_emb, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(que_emb, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs

# TODO: label_smoothing 是对ground truth做的不是对logits做的
def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)

