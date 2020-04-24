# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F
import torch
from fairseq import utils

from . import FairseqCriterion, register_criterion

import pdb

@register_criterion('cross_entropy')
class CrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #pdb.set_trace()
        net_output = model(**sample['net_input'])
        #origin
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        #pdb.set_trace()
        loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
                          reduce=reduce)

        #pdb.set_trace()
        
        target = model.get_targets(sample, net_output).view(-1)
        predict = torch.max(net_output[0],2)[1].squeeze(1)
        true_prob = net_output[0][:,:,5:].reshape(net_output[0].shape[0])
        #true_prob = torch.max(net_output[0],2)[0].squeeze(1)
        #true_prob = net_output[0][:,:,5:].reshape(net_output[0].shape[0])
        

        #MSE
        '''
        loss_fn = torch.nn.MSELoss(reduce=reduce, size_average=False)
        loss = loss_fn(predict, target)
        '''

        #pdb.set_trace()
        #RMSE
        '''
        loss_fn = torch.nn.MSELoss(reduce=reduce, size_average=True)
        loss = loss_fn(predict, target)
        loss = torch.sqrt(loss)
        #loss = loss*math.sqrt(target.size()[0])
        loss_fn_show = torch.nn.MSELoss(reduce=reduce, size_average=False)
        loss_show = loss_fn_show(predict, target)
        '''
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'target': target,
            'predict': predict,
            'true_prob': true_prob
        }
        #pdb.set_trace()
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        #pdb.set_trace()
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        target = [log.get('target') for log in logging_outputs]
        predict = [log.get('predict') for log in logging_outputs]
        true_prob= [log.get('true_prob') for log in logging_outputs]
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'target': target,
            'predict': predict,
            'true_prob': true_prob
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        #pdb.set_trace()
        return agg_output
