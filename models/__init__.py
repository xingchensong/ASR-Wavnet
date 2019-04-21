from .ASR_DFCNN import ASR
from .ASR_wavnet import ASR_wavnet
from .ASR_transformer2 import ASR_transformer2
from .ASR_transformer_encoder import ASR_transformer_encoder


def create_model(name, hparams,is_training =True):
  if name == 'ASR':
    return ASR(hparams,name=name)
  elif name == 'ASR_wavnet':
    return ASR_wavnet(hparams,name=name)
  elif name == 'ASR_transformer2':
    return ASR_transformer2(hparams,name=name,is_training =is_training)
  elif name == 'ASR_transformer_encoder':
    return ASR_transformer_encoder(hparams,name=name,is_training =is_training)
  else:
    raise Exception('Unknown model: ' + name)