#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   ernie_mtl.py
Author:   dangjinming(776039904@qq.com)
Date  :   2022/3/16
Desc  :   ernie_crf_v2
"""

import sys
import os
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

import paddle
import paddle.nn as nn
from paddlenlp.transformers import ErnieModel
from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss, ViterbiDecoder
from data_loader import DataLoader
from label_encoder import LabelEncoder

import logging
logging.basicConfig(
    format='"%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.INFO)

class ErnieMTLModel(nn.Layer):
    def __init__(self, ernie, ner_num_classes, cls_num_classes, dropout=None, crf_lr=0.1):
        super().__init__()
        self.ner_num_classes = ner_num_classes
        self.cls_num_classes = cls_num_classes
        self.ernie = ernie  # allow ernie to be config
        self.crf = LinearChainCrf(self.ner_num_classes, crf_lr=crf_lr, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions, False)
        self.ner_classifier = nn.Linear(self.ernie.config["hidden_size"], self.ner_num_classes)
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie.config["hidden_dropout_prob"])
        #classify
        self.cls_classifier = nn.Linear(self.ernie.config["hidden_size"], self.cls_num_classes)

    def forward(self,
                input_ids,
                true_lengths,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, pooled_output = self.ernie(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids)
        #ner
        sequence_output = self.dropout(sequence_output)
        ner_logits = self.ner_classifier(sequence_output)
        _, ner_prediction = self.viterbi_decoder(ner_logits, true_lengths)
        #classify
        pooled_output = self.dropout(pooled_output)
        cls_logits = self.cls_classifier(pooled_output)
        return ner_logits, ner_prediction, cls_logits



if __name__ == "__main__":
    ernie = ErnieModel.from_pretrained("ernie-1.0")
    model = ErnieMTLModel(ernie, ner_num_classes=3, cls_num_classes=3,dropout=0.1)
    from paddlenlp.transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('ernie-1.0')

    inputs = tokenizer("隔壁王老师在哪里!")
    inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}
    inputs["true_lengths"] = paddle.to_tensor([7])
    ner_logits, ner_prediction, cls_logits = model(**inputs)
    print(ner_logits.shape)
    print(ner_prediction.shape)
    print(cls_logits.shape)
