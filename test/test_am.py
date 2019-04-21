from hparams import hparams as hp
from models import create_model
from utils.mylogger import *
import os
import tensorflow as tf
import audio,librosa
import numpy as np
from tensorflow.python import pywrap_tensorflow


def compute_mfcc2(file):
    wav = audio.load_wav(file)
    # mfcc = p.mfcc(wav,numcep=hp.num_mfccs) # n_frames*n_mfcc
    mfcc = librosa.feature.mfcc(wav,sr=16000,n_mfcc=26)  # n_mfcc * n_frames
    n_frames = mfcc.shape[1]
    return (mfcc.T,n_frames)

file = 'D32_995.wav'
file1 = 'A2_34.wav'
file2 = 'BAC009S0002W0122.wav'
mfcc,length = compute_mfcc2(file1)
mfcc = np.expand_dims(mfcc,0)
print(mfcc.shape)
length = np.asarray(length).reshape(1,1)
print(length.shape,length) # 313

logdir = 'logging/logs-ASR_wavnet/'
checkpoint_path = os.path.join(logdir,'model.ckpt')
log('Checkpoint path: %s' % checkpoint_path)

step = 2100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('logging/logs-ASR_wavnet/model.ckpt-'+str(step)+'.meta')
    restore_path = '%s-%d' % (checkpoint_path,step)
    saver.restore(sess, restore_path)
    log('Resuming from checkpoint: %s ' % restore_path)

    graph = tf.get_default_graph()  # 或者sess.graph
    inputs = graph.get_tensor_by_name('model/NET/NET_Input/mfcc_inputs:0') # 名字一定陶写全否则报错tensor not exist
    inputs_len = graph.get_tensor_by_name('model/loss/CTC_Input/input_lengths:0')
    feed_dict = {inputs:mfcc,inputs_len:length}
    decoded_labels = graph.get_tensor_by_name('model/decode/decoded_labels:0')

    pred = sess.run(decoded_labels,feed_dict)
    print(pred)