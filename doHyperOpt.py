import hyperOpt
import hyperopt.pyll.stochastic
import math
from hyperopt import fmin, tpe, hp, Trials
import pdb

def fn_learn_it_all(params):
  new_dict = {}
  new_dict['max_tokens']=params['max_tokens']
  new_dict['encoder_embed_dim']=params['embed_dim']
  new_dict['decoder_embed_dim']=params['embed_dim']
  new_dict['decoder_out_embed_dim']=params['embed_dim']
  encoder_layers = params['encoder_layers']
  convolution_size = params['convolution_size']
  decoder_layers = params['decoder_layers']
  new_dict['encoder_layers'] = "[({}, 3)] * {}".format(convolution_size, encoder_layers)
  new_dict['decoder_layers'] = "[({}, 3)] * {}".format(convolution_size, decoder_layers)
  
  loss = hyperOpt.myRun(new_dict)
  return loss


#print(layerList)
space = {
    'max_tokens': hp.choice('max_tokens', [250, 500, 1000, 2000, 4000, 6000, 8000]),
    'embed_dim': hp.choice('embed_dim', [8, 16, 32, 64, 128, 256]),
    'encoder_layers': hp.choice('encoder_layers', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    'decoder_layers': hp.choice('decoder_layers', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    'convolution_size': hp.choice('convolution_size', [32, 64, 128, 256, 512]),
}

#print(hyperopt.pyll.stochastic.sample(space))

trials = Trials()
best = fmin(fn=fn_learn_it_all, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
print("doHyperOpt: {}".format(best))
print(trials.trials)


