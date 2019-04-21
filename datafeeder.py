import difflib,os
import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm
from scipy.fftpack import fft
from random import shuffle
from keras import backend as K
from utils import audio
from utils.mylogger import log
from hparams import hparams as hp
import librosa


class DataFeeder():
    '''
    属性：
        wav_lst :    ['data_aishell/wav/dev/S0724/BAC009S0724W0121.wav' , ...]
        pny_lst :    [['guang3','zhou1','shi4','fang2','di4','chan3','zhong1','jie4','xie2','hui4','fen1','xi1'] , ...]
        han_lst :    ['广州市房地产中介协会分析']
        am_vocab :   ['_', 'yi1', ...]
        pny_vocab :  ['<PAD>', 'yi1', ...]
        han_vocab :  ['<PAD>', '一', ...]
    '''
    def __init__(self, args):
        self.data_type = args.data_type  # train test dev
        self.data_path = args.data_path  # 存放数据的顶层目录
        self.thchs30 = args.thchs30      # 是否使用thchs30
        self.aishell = args.aishell
        self.prime = args.prime
        self.stcmd = args.stcmd
        self.data_length = args.data_length # 使用多少数据训练 None表示全部
        self.batch_size = args.batch_size   # batch大小
        self.shuffle = args.shuffle         # 是否打乱训练数据
        self.feature_type = args.feature_type
        self.AM = args.AM
        self.LM = args.LM
        self.source_init()

    def source_init(self):
        print('get source list...')
        read_files = []
        if self.data_type == 'train':
            if self.thchs30 == True:
                read_files.append('thchs_train.txt')
            if self.aishell == True:
                read_files.append('aishell_train.txt')
            if self.prime == True:
                read_files.append('prime.txt')
            if self.stcmd == True:
                read_files.append('stcmd.txt')
        elif self.data_type == 'dev':
            if self.thchs30 == True:
                read_files.append('thchs_dev.txt')
            if self.aishell == True:
                read_files.append('aishell_dev.txt')
        elif self.data_type == 'test':
            if self.thchs30 == True:
                read_files.append('thchs_test.txt')
            if self.aishell == True:
                read_files.append('aishell_test.txt')
        self.wav_lst = []
        self.pny_lst = []
        self.han_lst = []
        for file in read_files:
            print('load ', file, ' data...')
            sub_file = 'datasets/' + file
            with open(sub_file, 'r', encoding='utf8') as f:
                data = f.readlines()
            for line in tqdm(data):
                wav_file, pny, han = line.split('\t')
                self.wav_lst.append(wav_file)
                self.pny_lst.append(pny.split(' '))
                self.han_lst.append(han.strip('\n'))
        if self.data_length:
            self.wav_lst = self.wav_lst[:self.data_length]
            self.pny_lst = self.pny_lst[:self.data_length]
            self.han_lst = self.han_lst[:self.data_length]
        if self.AM:
            print('make am vocab...')
            self.am_vocab = self.mk_am_vocab(self.pny_lst)
        if self.LM:
            print('make lm pinyin vocab...')
            self.pny_vocab = self.mk_lm_pny_vocab(self.pny_lst)
            print('make lm hanzi vocab...')
            self.han_vocab = self.mk_lm_han_vocab(self.han_lst)

    def get_am_batch(self):
        # shuffle只是对index打乱，没有对原始数据打乱，所以wav_lst[i]和pny_lst[i]还是一一对应的
        shuffle_list = [i for i in range(len(self.wav_lst))]
        while 1:
            if self.shuffle == True:
                shuffle(shuffle_list)
            # len(self.wav_lst) // self.batch_size的值 表示一个epoch里有多少step才能把所有数据过一遍
            for i in range(len(self.wav_lst) // self.batch_size):
                wav_data_lst = []    # wav_data_lst里放的是batch_size个频谱图 wav_lst里放的是音频文件地址
                label_data_lst = []
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = shuffle_list[begin:end]
                for index in sub_list:
                    # TODO：计算频谱图
                    if self.feature_type == 'spec':
                        fbank,n_frames = compute_spec(self.data_path + self.wav_lst[index])
                    elif self.feature_type == 'mel':
                        fbank, n_frames = compute_mel2(self.data_path + self.wav_lst[index])
                    else:
                        fbank, n_frames = compute_mfcc2(self.data_path + self.wav_lst[index])
                    # TODO：把语谱图时间维度pad成8的倍数
                    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1])) # 保证是8的倍数，因为几层cnn后要求维度能被8整除
                    pad_fbank[:fbank.shape[0], :] = fbank
                    label = self.pny2id(self.pny_lst[index], self.am_vocab)
                    label_ctc_len = self.ctc_len(label)
                    # 假设num_mel =90 那么最多只能预测十个字的句子？
                    if pad_fbank.shape[0] // 8 >= label_ctc_len:# ctc要求解码长度必须小于等于原始输入长度
                        wav_data_lst.append(pad_fbank)
                        label_data_lst.append(label)
                    else:
                        print(self.data_path + self.wav_lst[index])
                        raise Exception('data not allowed')
                # TODO：对语谱图时间维度进行第二次pad，pad成本次batch中最长的长度
                pad_wav_data, input_length = self.wav_padding(wav_data_lst)
                pad_label_data, label_length = self.label_padding(label_data_lst)
                inputs = {'the_inputs': pad_wav_data,
                          'the_labels': pad_label_data,
                          'input_length': input_length.reshape(-1,1),
                          # 注意这个input_length不是原始频谱图的长度而是经过几层cnn后隐层输出的长度（传给ctc）
                          # 而且是语谱图第一次pad后的长度//8而不是第二次最长pad后//8，尽量保证靠近真实长度//8的值
                          'label_length': label_length.reshape(-1,1),# batch中的每个句子的真实长度
                          }
                # outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )} # 没看懂为什么有这个
                print('genarate one batch mel data')
                yield inputs
        pass

    def get_lm_batch(self):
        batch_num = len(self.pny_lst) // self.batch_size
        for k in range(batch_num):
            begin = k * self.batch_size
            end = begin + self.batch_size
            input_batch = self.pny_lst[begin:end]
            label_batch = self.han_lst[begin:end]
            max_len = max([len(line) for line in input_batch])
            input_batch = np.array(
                [self.pny2id(line, self.pny_vocab) + [0] * (max_len - len(line)) for line in input_batch])
            label_batch = np.array(
                [self.han2id(line, self.han_vocab) + [0] * (max_len - len(line)) for line in label_batch])
            yield (input_batch, label_batch)

    def pny2id(self, line, vocab):
        ids = []
        for pny in line :
            if pny in vocab:
                ids.append(vocab.index(pny))
            else:
                if 'UnkTok' in vocab:
                    ids.append(vocab.index('UnkTok'))
                else:
                    ids.append(vocab.index('_')-1)

        return ids

    def han2id(self, line, vocab):
        ids = []
        for han in line:
            if han in vocab:
                ids.append(vocab.index(han))
            else:
                ids.append(vocab.index('UnkTok'))

        return ids

    def wav_padding(self, wav_data_lst):
        # wav_data_lst里data是pad_fbank不是原始fbank所以后面//8一定能整除，这个训练的时候为什么不能直接在网络中获取呢？？
        wav_lens = [len(data) for data in wav_data_lst] # len(data)实际上就是求语谱图的第一维的长度，也就是n_frames
        # print(wav_lens)
        wav_max_len = max(wav_lens)
        wav_lens = np.array([leng // 8 for leng in wav_lens])
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, hp.num_mels, 1)) # 之前固定200是n-fft，len(wav_data_lst)是batch_size
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
        # print('new_wav_data_lst',new_wav_data_lst.shape,wav_lens.shape)
        return new_wav_data_lst, wav_lens

    def label_padding(self, label_data_lst):
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens

    def mk_am_vocab(self, data):
        vocab = []
        for line in tqdm(data):
            line = line
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        vocab.append('_')
        return vocab

    def mk_lm_pny_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        vocab.append('UnkTok')
        return vocab

    def mk_lm_han_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            line = ''.join(line.split(' '))#能将‘你好吗’直接拆成三个字
            for han in line:
                if han not in vocab:
                    vocab.append(han)
        vocab.append('UnkTok')
        return vocab

    def ctc_len(self, label):
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1 # 这里+1是因为ctc会在重复字符之间填充
        return label_len + add_len

class DataFeeder_wavnet(DataFeeder):
    def __init__(self,args):
        super().__init__(args)

    def get_am_batch(self):
        shuffle_list = [i for i in range(len(self.wav_lst))]
        while 1:
            if self.shuffle == True:
                shuffle(shuffle_list)
            # len(self.wav_lst) // self.batch_size的值 表示一个epoch里有多少step才能把所有数据过一遍
            for i in range(len(self.wav_lst) // self.batch_size):
                wav_data_lst = []  # wav_data_lst里放的是batch_size个频谱图 wav_lst里放的是音频文件地址
                label_data_lst = []
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = shuffle_list[begin:end]
                for index in sub_list:
                    # TODO：计算频谱图
                    if self.feature_type == 'spec':
                        fbank, n_frames = compute_spec(self.data_path + self.wav_lst[index])
                    elif self.feature_type == 'mel':
                        fbank, n_frames = compute_mel(self.data_path + self.wav_lst[index])
                    else:
                        fbank, n_frames = compute_mfcc2(self.data_path + self.wav_lst[index])

                    pad_fbank = fbank
                    label = self.pny2id(self.pny_lst[index], self.am_vocab)
                    label_ctc_len = self.ctc_len(label)
                    # 假设num_mel =90 那么最多只能预测十个字的句子？
                    if pad_fbank.shape[0] >= label_ctc_len:  # ctc要求解码长度必须小于等于原始输入长度
                        wav_data_lst.append(pad_fbank)
                        label_data_lst.append(label)
                    else:
                        print(self.data_path + self.wav_lst[index])
                        raise Exception('data not allowed')
                # TODO：对mfcc时间维度进行pad，pad成本次batch中最长的长度
                pad_wav_data, input_length = self.wav_padding(wav_data_lst)
                pad_label_data, label_length = self.label_padding(label_data_lst)
                inputs = {'the_inputs': pad_wav_data,
                          'the_labels': pad_label_data,
                          'input_length': input_length.reshape(-1, 1),
                          # 注意这个input_length不是原始频谱图的长度而是经过几层cnn后隐层输出的长度（传给ctc）
                          # 而且是语谱图第一次pad后的长度//8而不是第二次最长pad后//8，尽量保证靠近真实长度//8的值
                          'label_length': label_length.reshape(-1, 1),  # batch中的每个句子的真实长度
                          }
                # outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )} # 没看懂为什么有这个
                print('genarate one batch mfcc data')
                yield inputs
        pass

    def wav_padding(self, wav_data_lst):
        wav_lens = np.asarray([len(data) for data in wav_data_lst])  # len(data)实际上就是求语谱图的第一维的长度，也就是n_frames
        wav_max_len = max(wav_lens)
        new_wav_data_lst = np.zeros(
            (len(wav_data_lst), wav_max_len, hp.num_mfccs))  # 之前固定200是n-fft，len(wav_data_lst)是batch_size
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :] = wav_data_lst[i]
        return new_wav_data_lst, wav_lens

class DataFeeder_transformer(DataFeeder):
    def __init__(self,args):
        super().__init__(args)

    def mk_lm_han_vocab(self,data):
        vocab = ['<PAD>','<GO>','<EOS>']
        for line in tqdm(data):
            line = ''.join(line.split(' '))  # 能将‘你好吗’直接拆成三个字
            for han in line:
                if han not in vocab:
                    vocab.append(han)
        return vocab

    def get_lm_batch(self):
        encoder_inputs = [[self.pny_vocab.index(word) for word in line] for line in self.pny_lst]
        decoder_inputs = [[self.han_vocab.index('<GO>')] + [self.han_vocab.index(word) for word in ''.join(line.split(' '))] for line in self.han_lst]
        decoder_targets = [[self.han_vocab.index(word) for word in ''.join(line.split(' '))] + [self.han_vocab.index('<EOS>')] for line in self.han_lst]

        batch_num = len(encoder_inputs) // self.batch_size
        for k in range(batch_num):
            begin = k * self.batch_size
            end = begin + self.batch_size
            en_input_batch = encoder_inputs[begin:end]
            de_input_batch = decoder_inputs[begin:end]
            de_label_batch = decoder_targets[begin:end]
            max_en_len = max([len(line) for line in en_input_batch])
            max_de_len = max([len(line) for line in de_input_batch])
            en_input_batch = np.array(
                [line + [0] * (max_en_len - len(line)) for line in en_input_batch]
            )
            de_input_batch = np.array(
                [line + [0] * (max_de_len - len(line)) for line in de_input_batch]
            )
            de_label_batch = np.array(
                [line + [0] * (max_de_len - len(line)) for line in de_label_batch]
            )
            yield en_input_batch, de_input_batch, de_label_batch


def compute_mfcc2(file):
    wav = audio.load_wav(file)
    # mfcc = p.mfcc(wav,numcep=hp.num_mfccs) # n_frames*n_mfcc
    mfcc = librosa.feature.mfcc(wav,sr=hp.sample_rate,n_mfcc=hp.num_mfccs)  # n_mfcc * n_frames
    n_frames = mfcc.shape[1]
    return (mfcc.T,n_frames)

def compute_mfcc(file):
    wav = audio.load_wav(file)
    mfcc = audio.mfcc(wav).astype(np.float32)
    n_frames = mfcc.shape[1]
    return (mfcc.T,n_frames)

def compute_mel2(file):
    wav = audio.load_wav(file)
    # mel = audio.melspectrogram(wav).astype(np.float32)
    mel = librosa.feature.melspectrogram(wav,sr=hp.sample_rate,n_mels=hp.num_mels,hop_length=256)  # [shape=(n_mels, t)]
    n_frames = mel.shape[1]
    return (mel.T,n_frames)

def compute_mel(file):
    wav = audio.load_wav(file)
    mel = audio.melspectrogram(wav).astype(np.float32)
    n_frames = mel.shape[1]
    return (mel.T,n_frames)

def compute_spec(file):
    wav = audio.load_wav(file)    # np.ndarray [shape=(n,) or (2, n)] 2表示双声道
    spectrogram = audio.spectrogram(wav).astype(np.float32) # np.ndarray [shape=(num_freq, num_frames), dtype=float32]
    n_frames = spectrogram.shape[1]
    return (spectrogram.T, n_frames) # 注意转置后[shape=( num_frames, num_freq), dtype=float32]

# # 获取信号的时频图
# def compute_fbank(file):
#     x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
#     w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
#     fs, wavsignal = wav.read(file) # fs是采样率
#     # wav波形 加时间窗以及时移10ms
#     time_window = 25  # 单位ms 假设采样率16000那么采样点就是400，对应下面p_end = p_start + 400
#     preemphasis = 0.97
#     wav_arr = np.array(wavsignal)
#     range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
#     data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据
#     data_line = np.zeros((1, 400), dtype=np.float)
#     for i in range(0, range0_end):
#         p_start = i * 160 # 窗移是每次160个点，也就是10ms
#         p_end = p_start + 400 # 400就是n-fft，如果采样率是16000那么n-fft就等于窗长
#         data_line = wav_arr[p_start:p_end]
#         data_line = data_line * w  # 加窗
#         data_line = np.abs(fft(data_line))
#         data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
#     data_input = np.log(data_input + 1)
#     # data_input = data_input[::]
#     # TODO：增加norm和预加重
#     return data_input  # [shape=( num_frames, num_freq), dtype=float32]


# word error rate------------------------------------
def GetEditDistance(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            leven_cost += (j2-j1)
        elif tag == 'delete':
            leven_cost += (i2-i1)
    return leven_cost

# 定义解码器------------------------------------
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
