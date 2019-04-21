# import tensorflow as tf  # import 严重影响速度
import numpy as np
# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
from tensorflow.contrib.util import make_tensor_proto
# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import librosa
import audio
from keras import backend as K

########################################################### AM

server = '219.223.173.213:9001'
model_name = 'ASR_am'

def decode_ctc(num_result, num2word):
	result = num_result[:, :, :]
	in_len = np.zeros((1), dtype = np.int32)
	in_len[0] = result.shape[1]
	r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
	r1 = K.get_value(r[0][0])
	r1 = r1[0]
	text = []
	for i in r1:
		text.append(num2word[i])
	return r1, text

def compute_mfcc2(file):
    wav = audio.load_wav(file)
    # mfcc = p.mfcc(wav,numcep=hp.num_mfccs) # n_frames*n_mfcc
    mfcc = librosa.feature.mfcc(wav,sr=16000,n_mfcc=26)  # n_mfcc * n_frames
    n_frames = mfcc.shape[1]
    return (mfcc.T,n_frames)

file = 'D32_995.wav'
file1 = 'A12_49.wav'
file2 = 'BAC009S0002W0122.wav'
mfcc,length = compute_mfcc2(file1)
mfcc = np.expand_dims(mfcc,0)
print(mfcc.shape)
length = np.asarray(length).reshape(1,1)
print(length.shape,length) # 313

host, port = server.split(':')
channel = implementations.insecure_channel(host, int(port))
# stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel._channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = model_name
request.model_spec.signature_name = 'predict_AudioSpec2Pinyin'
request.inputs['mfcc'].CopyFrom(make_tensor_proto(mfcc, shape=mfcc.shape, dtype='float'))  # 1是batch_size
request.inputs['len'].CopyFrom(make_tensor_proto(length, shape=length.shape, dtype='int32'))
print('begin')
result = stub.Predict(request, 60.0)

am_vocab = np.load('am_pinyin_dict.npy').tolist()
pred_logits = np.array(result.outputs['logits'].float_val).reshape(1,-1,len(am_vocab)).astype(np.float32)
labels = np.asarray(result.outputs['label'].int_val)
print('label.shape:',labels.shape)
print('label      :',labels)

print('pred_logits:',pred_logits)

r1 , pinyin = decode_ctc(pred_logits,am_vocab)
print('pinyin1     :',pinyin)

pinyin = [am_vocab[i] for i in labels]
print('pinyin2     :',pinyin)

##################################################### LM

server = '219.223.173.213:9002'
model_name = 'ASR_lm'

lm_pinyin_vocab = np.load('lm_pinyin_dict.npy').tolist()
lm_hanzi_vocab = np.load('lm_hanzi_dict.npy').tolist()

# pinyin = ['jin1','tian1','tian1','qi4','zhen1','hao3']
pinyin = [lm_pinyin_vocab.index(i) for i in pinyin]
pinyin = np.asarray(pinyin).reshape(1,-1)
print(pinyin.shape,pinyin )

host, port = server.split(':')
channel = implementations.insecure_channel(host, int(port))
# stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel._channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = model_name
request.model_spec.signature_name = 'predict_Pinyin2Hanzi'
request.inputs['pinyin'].CopyFrom(
    make_tensor_proto(pinyin, shape=pinyin.shape, dtype='int32'))  # 1是batch_size
print('begin')
result = stub.Predict(request, 60.0)
pred_label = np.array(result.outputs['hanzi'].int_val)

print(pred_label.shape,pred_label)
hanzi = [lm_hanzi_vocab[i] for i in pred_label]
print(hanzi)