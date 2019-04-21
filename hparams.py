from tensorflow.contrib.training.python.training.hparam import HParams
import os
# Default hyperparameters:
hparams = HParams(

    # TODO: audio
    # num_freq=2048,  # 使用librosa默认值 n-fft 256 一般等于窗长（一个窗口有多少个采样点）,也就是16000/1000*25ms
    num_mels=40, #  通常设为 20-40
    num_mfccs = 26,#一般至少39啊
    # 如果使用快速傅里叶变换（fft）需要保证窗长是2的倍数，上面400会自动扩展为512
    # max_num_frames = None,  # max_num_frames
    sample_rate =16000,
    # frame_length_ms=25, # 25  50  使用librosa默认值
    # frame_shift_ms=10, # 10  12.5 使用librosa默认值
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    # griffin_lim_iters=60,
    # power=1.5,              # Power to raise magnitudes to prior to Griffin-Lim


    # TODO: am
    AM =True,
    data_type='train', # train test dev
    feature_type = 'mel', # spec, mel, mfcc
    data_path= os.path.expanduser('~/corpus_zn/'), # wav文件顶层目录
    thchs30=True,      # 是否使用thchs30数据
    aishell=True,
    prime=False,
    stcmd=False,
    data_length=None,    # 总共使用条语音数据来训练，None表示全部
    shuffle=True,
    wavnet_filters = 192,

    # TODO: lm
    LM = True,
    num_heads = 8,    # 多头注意力机制
    max_seq_length = 100,
    num_blocks = 6,
    # vocab
    input_vocab_size = None, # 数据中有多少不同发音，一般读取所有train.txt获得
    label_vocab_size = None, # 数据中有多少不同字，一般读取所有train.txt获得
    # embedding size
    word_embedding_size=None,
    transformer_dropout = 0.1,
    transformer_depth = None,
    embedding_dropout = 0.6,
    l2_reg_penalty = 1e-6,
    confidence_penalty_weight = 0.1,
    hidden_units = 512,


    # TODO: training
    is_training = True,  # 注意测试的时候或者inference的时候设置为False
    wavenet_filters = 192, # 128
    batch_size=64,#32,
    adam_beta1=0.9,
    adam_beta2=0.999,
    initial_learning_rate=0.002,#0.004, # 0.001
    steps_per_epoch = None, # 283600//16 (283600是总音频数，16为batch_size)
    decay_learning_rate=False,
    max_iters=100,
    final_output_dim = 1292

)

def hparams_debug_string(hpa):
    values = hpa.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)