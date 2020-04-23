from model.relation_representation_model import EntityMarkerEncoder, EntityMarkerClsEncoder
from configure import FLAGS
import sklearn.exceptions
import warnings
import utils
from dataloader.fewrel_data_loader import JSONFileDataLoader
from framework import FewShotREFramework
import sys
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from transformers import AlbertTokenizer, AlbertModel, BertTokenizer, BertModel
# from pytorch_pretrained_bert import BertAdam as Adam
from torch import nn
from torch.optim import Adam
import os
import time
import numpy as np
import torch
import prettytable as pt
from model.interact_proto import InstanceTransformer, InteractiveContrastiveNet


model_name = 'bert'
N = FLAGS.N
K = FLAGS.K
fc_out_size = 256

relation_encoder = None
if FLAGS.rel_rep_model == 'em':
    relation_encoder = EntityMarkerEncoder()
elif FLAGS.rel_rep_model == "emc":
    relation_encoder = EntityMarkerClsEncoder()
else:
    raise NotImplementedError
bert_model = BertModel.from_pretrained(FLAGS.bert_model)
# bert_model = AlbertModel.from_pretrained(FLAGS.bert_model)

tokenizer = BertTokenizer.from_pretrained(
    FLAGS.bert_model, do_basic_tokenize=False)
# tokenizer = AlbertTokenizer.from_pretrained(
#     FLAGS.bert_model, do_basic_tokenize=False)
tokenizer.add_special_tokens(
    {"additional_special_tokens": [FLAGS.e11, FLAGS.e12, FLAGS.e21, FLAGS.e22]})
bert_model.resize_token_embeddings(len(tokenizer))

# ckpt_file_path = "./checkpoint/fewrel/ipl.layer{}-{}".format(FLAGS.layer,"20-2-13")
ckpt_file_path = "./checkpoint/fewrel/ipl.layer{}".format(FLAGS.layer)
print("#######################################")
print(ckpt_file_path)
# ckpt_file_path = FLAGS.semeval_ckpt_file+FLAGS.rel_rep_model


print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
print("Model: {}".format(model_name))

max_length = FLAGS.max_sentence_length
train_data_loader = JSONFileDataLoader(
    './data/train.json', './data/glove.6B.50d.json', max_length=max_length, case_sensitive=True, reprocess=FLAGS.reproc_data, bertTokenizer=tokenizer, na_rate=FLAGS.na_rate)
val_data_loader = JSONFileDataLoader(
    './data/val_wiki.json', './data/glove.6B.50d.json', max_length=max_length, case_sensitive=True, reprocess=FLAGS.reproc_data, bertTokenizer=tokenizer, na_rate=FLAGS.na_rate)
test_data_loader = JSONFileDataLoader(
    './data/val_pubmed.json', './data/glove.6B.50d.json', max_length=max_length, case_sensitive=True, reprocess=FLAGS.reproc_data, bertTokenizer=tokenizer, na_rate=FLAGS.na_rate)
gpu_aval = torch.cuda.is_available()


model = InteractiveContrastiveNet(tokenizer, bert_model,
                            relation_encoder, max_length)
if os.path.exists(ckpt_file_path):
    if FLAGS.paral_cuda[0] >= 0:
        ckpt = torch.load(ckpt_file_path, map_location=lambda storage,
                          loc: storage.cuda(FLAGS.paral_cuda[0]))
    else:
        ckpt = torch.load(
            ckpt_file_path, map_location=lambda storage, loc: storage.cpu())
    model.load_state_dict(ckpt["state_dict"])
    print("######################load fewrel full model from {}#######################".format(
        ckpt_file_path))

# for p in model.rr_model.sentence_encoder.parameters():
#     p.requires_grad = False
if gpu_aval:
    model = model.to(FLAGS.paral_cuda[0])


framework = FewShotREFramework(
    train_data_loader, val_data_loader, None, test_data_loader)
# sentence_encoder = CNNSentenceEncoder(
#     train_data_loader.word_vec_mat, max_length)
if FLAGS.mode == "train":
    framework.train(model, model_name, FLAGS.batch_size, 5, N, K, 1,
                    learning_rate=FLAGS.learning_rate, weight_decay=FLAGS.l2_reg_lambda, optimizer=Adam, ckpt_file=ckpt_file_path)

else:
    with torch.no_grad():
        tabel = pt.PrettyTable(
            ["51_wiki", "55_wiki", "101_wiki", "105_wiki", "51_pubmed", "55_pubmed", "101_pubmed", "105_pubmed"])
        val_step = 1000
        acc1 = framework.eval(model, FLAGS.batch_size, 5, 1, 1,
                              val_step, data_loader=val_data_loader)
        acc2 = framework.eval(model, FLAGS.batch_size, 5, 5, 1,
                              val_step, data_loader=val_data_loader)
        acc3 = framework.eval(model, FLAGS.batch_size, 10, 1, 1,
                              val_step, data_loader=val_data_loader)
        acc4 = framework.eval(model, FLAGS.batch_size, 10, 5, 1,
                              val_step, data_loader=val_data_loader)
        acc5 = framework.eval(model, FLAGS.batch_size, 5, 1, 1,
                              val_step, data_loader=test_data_loader)
        acc6 = framework.eval(model, FLAGS.batch_size, 5, 5, 1,
                              val_step, data_loader=test_data_loader)
        acc7 = framework.eval(model, FLAGS.batch_size, 10, 1, 1,
                              val_step, data_loader=test_data_loader)
        acc8 = framework.eval(model, FLAGS.batch_size, 10, 5, 1,
                              val_step, data_loader=test_data_loader)
        tabel.add_row(
            [round(100*acc1, 4), round(100*acc2, 4), round(100*acc3, 4), round(100*acc4, 4),
             round(100*acc5, 4), round(100*acc6, 4), round(100*acc7, 4), round(100*acc8, 4)])
        print(tabel)
