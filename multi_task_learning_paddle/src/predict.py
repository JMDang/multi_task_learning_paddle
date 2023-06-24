#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   predict.py
Author:   dangjinming(776039904@qq.com)
Date  :   2022/3/16
Desc  :   predict入口
"""

import os
import sys
import json
import logging
import configparser
import numpy as np
import paddle
from paddlenlp.transformers import AutoTokenizer,ErnieModel

import dygraph
from data_loader import DataLoader
from label_encoder import LabelEncoder
from models.ernie_mtl import ErnieMTLModel


class Predict:
    """模型预测
    """
    def __init__(self, predict_conf_path):
        self.predict_conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.predict_conf.read(predict_conf_path)
        self.label_encoder_cls = LabelEncoder(label_id_info=self.predict_conf["DATA"]["label_encoder_cls"],
                                              isFile=True)
        self.label_encoder_ner = LabelEncoder(label_id_info=self.predict_conf["DATA"]["label_encoder_ner"],
                                    isFile=True)
        for label_id, label_name in sorted(self.label_encoder_cls.id_label_dict.items(), key=lambda x:x[0]):
            logging.info("%d: %s" % (label_id, label_name))
        for label_id, label_name in sorted(self.label_encoder_ner.id_label_dict.items(), key=lambda x:x[0]):
            logging.info("%d: %s" % (label_id, label_name))

        self.tokenizer = AutoTokenizer.from_pretrained(self.predict_conf["ERNIE"]["pretrain_model"])

    def run(self):
        """执行入口
        """
        if self.predict_conf["RUN"].getboolean("finetune_ernie"):
            predict_conf = self.predict_conf["ERNIE"]
            pre_trainernie = ErnieModel.from_pretrained(self.predict_conf["ERNIE"]["pretrain_model"])
            model_predict = ErnieMTLMode(pre_trainernie,
                                         ner_num_classes=self.label_encoder_ner.size() * 2 - 1,
                                         cls_num_classes=self.label_encoder_cls.size())
            dygraph.load_model(model_predict, self.predict_conf["MODEL_FILE"]["model_best_path"])

            predict_data = []
            text_list = []
            length_list = []
            origin_texts = []
            mark = 0
            tmp_d = {}
            for line in sys.stdin:
                mark = mark + 1
                cols = line.strip("\n").split("\t")
                origin_text = cols[0]
                text = self.tokenizer(" ".join(list(origin_text)))['input_ids']
                text_list.append(text)
                length_list.append(len(text))
                origin_texts.append(origin_text)
                tmp_d[origin_text] = cols

                if mark == 32:
                    predict_data = list(zip(text_list, length_list))
                    ner_pre_label, ner_pre_entity, cls_pre_label, cls_pre_lname = \
                        dygraph.predict(model=model_predict,
                                        predict_data=predict_data,
                                        label_encoder_ner=self.label_encoder_ner,
                                        label_encoder_cls=self.label_encoder_cls,
                                        batch_size=predict_conf.getint("batch_size"),
                                        max_seq_len=predict_conf.getint("max_seq_len"),
                                        max_ensure=True,
                                        with_label=False)
                    for origin_text, entity, pre_lname in zip(origin_texts, ner_pre_entity, cls_pre_lname):
                        out = [origin_text] + tmp_d[origin_text][1:] + [json.dumps(entity, ensure_ascii=False), pre_lname]
                        print("\t".join(out))
                    predict_data = []
                    text_list = []
                    length_list = []
                    origin_texts = []
                    mark = 0
            if mark != 0:
                predict_data = list(zip(text_list, length_list))
                ner_pre_label, ner_pre_entity, cls_pre_label, cls_pre_lname = \
                    dygraph.predict(model=model_predict,
                                    predict_data=predict_data,
                                    label_encoder_ner=self.label_encoder_ner,
                                    label_encoder_cls=self.label_encoder_cls,
                                    batch_size=predict_conf.getint("batch_size"),
                                    max_seq_len=predict_conf.getint("max_seq_len"),
                                    max_ensure=True,
                                    with_label=False)
                for origin_text, entity, pre_lname in zip(origin_texts, ner_pre_entity, cls_pre_lname):
                    out = [origin_text] + tmp_d[origin_text][1:] + [json.dumps(entity, ensure_ascii=False), pre_lname]
                    print("\t".join(out))


if __name__ == "__main__":
    Predict(predict_conf_path=sys.argv[1]).run()
