from model.relation_representation_model import EntityMarkerEncoder, EntityMarkerClsEncoder
from configure import FLAGS
import sklearn.exceptions
import warnings
import utils
from dataloader.fewrel_data_loader import get_loader
from framework import FewShotREFramework
import sys
from transformers import AlbertTokenizer, AlbertModel, BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
import os
import time
import numpy as np
import torch
import prettytable as pt
from model.interact_proto import GlobalTransformedProtoNet_proto_three, GlobalTransformedProtoNet_three, GlobalTransformedProtoNet_new, InstanceTransformer, InteractiveContrastiveNet, GlobalTransformedProtoNet, Proto, ProtoHATT, GlobalTransformedProtoNet_onehot, GlobalTransformedProtoNet_all_query, GlobalTransformedProtoNet_proto_tag, GlobalTransformedProtoNet_proto_tag_cos


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
bert_model = BertModel.from_pretrained(
    FLAGS.bert_model, output_attentions=True)
# bert_model = AlbertModel.from_pretrained(FLAGS.bert_model)

tokenizer = BertTokenizer.from_pretrained(
    FLAGS.bert_model, do_basic_tokenize=False)
# tokenizer = AlbertTokenizer.from_pretrained(
#     FLAGS.bert_model, do_basic_tokenize=False)
tokenizer.add_special_tokens(
    {"additional_special_tokens": [FLAGS.e11, FLAGS.e12, FLAGS.e21, FLAGS.e22]})
bert_model.resize_token_embeddings(len(tokenizer))

# ckpt_file_path = "./checkpoint/fewrel/ipl.layer{}-{}".format(FLAGS.layer,"20-2-13")
ckpt_file_path = "./checkpoint/fewrel/{}".format(FLAGS.ckpt_name)
print("#######################################")
print(ckpt_file_path)
# ckpt_file_path = FLAGS.semeval_ckpt_file+FLAGS.rel_rep_model


print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
print("Model: {}".format(model_name))

max_length = FLAGS.max_sentence_length
# train_data_loader = JSONFileDataLoader(
#     './data/train.json', './data/glove.6B.50d.json', max_length=max_length, case_sensitive=True, reprocess=FLAGS.reproc_data, bertTokenizer=tokenizer, na_rate=FLAGS.na_rate)
# val_data_loader = JSONFileDataLoader(
#     './data/val_wiki.json', './data/glove.6B.50d.json', max_length=max_length, case_sensitive=True, reprocess=FLAGS.reproc_data, bertTokenizer=tokenizer, na_rate=FLAGS.na_rate)
# test_data_loader = JSONFileDataLoader(
#     './data/val_pubmed.json', './data/glove.6B.50d.json', max_length=max_length, case_sensitive=True, reprocess=FLAGS.reproc_data, bertTokenizer=tokenizer, na_rate=FLAGS.na_rate)


# train_data_loader_51 = get_loader(
#     './data/train.json', tokenizer, 5, 1, FLAGS.Q, FLAGS.batch_size, num_workers=2)
# train_data_loader_55 = get_loader(
#     './data/train.json', tokenizer, 5, 5, FLAGS.Q, FLAGS.batch_size, num_workers=2)
train_data_loader_101 = get_loader(
    './data/train.json', tokenizer, 10, 1, FLAGS.Q, FLAGS.batch_size, num_workers=2)
train_data_loader_105 = get_loader(
    './data/train.json', tokenizer, 10, 5, FLAGS.Q, FLAGS.batch_size, num_workers=2)

val_data_loader = get_loader(
    './data/val_pubmed.json', tokenizer, 10, 5, FLAGS.Q, FLAGS.batch_size, num_workers=2)
test_data_loader = get_loader(
    './data/val_pubmed.json', tokenizer, 10, 1, FLAGS.Q, FLAGS.batch_size, num_workers=2)
gpu_aval = torch.cuda.is_available()


# model = InteractiveContrastiveNet(tokenizer, bert_model,
#                                   relation_encoder, max_length)
# model = InstanceTransformer(tokenizer, bert_model,
#                             relation_encoder, max_length)

# model = Proto(tokenizer, bert_model,
#               relation_encoder, max_length)
# model = ProtoHATT(bert_model, relation_encoder, max_length, 5)
if FLAGS.model_name == 'gtp':
    model = GlobalTransformedProtoNet(tokenizer, bert_model,
                                      relation_encoder, max_length)
elif FLAGS.model_name == "onehot":
    model = GlobalTransformedProtoNet_onehot(tokenizer, bert_model,
                                             relation_encoder, max_length)
elif FLAGS.model_name == "all":
    model = GlobalTransformedProtoNet_all_query(
        tokenizer, bert_model, relation_encoder, max_length)
elif FLAGS.model_name == "proto_three":
    model = GlobalTransformedProtoNet_proto_three(
        tokenizer, bert_model, relation_encoder, max_length)

elif FLAGS.model_name == "proto":
    model = Proto(
        tokenizer, bert_model, relation_encoder, max_length)

elif FLAGS.model_name == "tag":
    model = GlobalTransformedProtoNet_proto_tag(
        tokenizer, bert_model, relation_encoder, max_length)
elif FLAGS.model_name == "tag_cos":
    model = GlobalTransformedProtoNet_proto_tag_cos(
        tokenizer, bert_model, relation_encoder, max_length)
elif FLAGS.model_name == "discrim":
    model = GlobalTransformedProtoNet_new(tokenizer, bert_model,
                                          relation_encoder, max_length)
elif FLAGS.model_name == "multi":
    model = GlobalTransformedProtoNet_three(tokenizer, bert_model,
                                            relation_encoder, max_length)
else:
    raise Exception("no such model name:{}" % FLAGS.model_name)
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
    train_data_loader_101, train_data_loader_105, val_data_loader, None, test_data_loader)
# sentence_encoder = CNNSentenceEncoder(
#     train_data_loader.word_vec_mat, max_length)
if FLAGS.mode == "train":
    framework.train(model, model_name, FLAGS.batch_size, 10, N, K, 1,
                    learning_rate=FLAGS.learning_rate, weight_decay=FLAGS.l2_reg_lambda, optimizer=Adam, ckpt_file=ckpt_file_path)

else:
    with torch.no_grad():
        tabel = pt.PrettyTable(
            ["51_wiki", "55_wiki", "101_wiki", "105_wiki", "51_pubmed", "55_pubmed", "101_pubmed", "105_pubmed"])
        val_step = 1000

        val_data_loader = get_loader(
            './data/val.json', tokenizer, 5, 1, 1, FLAGS.batch_size, num_workers=2)
        acc1 = framework.eval(model, FLAGS.batch_size, 5, 1, 1,
                              val_step, data_loader=val_data_loader)

        val_data_loader = get_loader(
            './data/val.json', tokenizer, 5, 5, 1, FLAGS.batch_size, num_workers=2)
        acc2 = framework.eval(model, FLAGS.batch_size, 5, 5, 1,
                              val_step, data_loader=val_data_loader)

        val_data_loader = get_loader(
            './data/val.json', tokenizer, 10, 1, 1, FLAGS.batch_size, num_workers=2)
        acc3 = framework.eval(model, FLAGS.batch_size, 10, 1, 1,
                              val_step, data_loader=val_data_loader)

        val_data_loader = get_loader(
            './data/val.json', tokenizer, 10, 5, 1, FLAGS.batch_size, num_workers=2)
        acc4 = framework.eval(model, FLAGS.batch_size, 10, 5, 1,
                              val_step, data_loader=val_data_loader)

        val_data_loader = get_loader(
            './data/val_pubmed.json', tokenizer, 5, 1, 1, FLAGS.batch_size, num_workers=2)
        acc5 = framework.eval(model, FLAGS.batch_size, 5, 1, 1,
                              val_step, data_loader=val_data_loader)

        val_data_loader = get_loader(
            './data/val_pubmed.json', tokenizer, 5, 5, 1, FLAGS.batch_size, num_workers=2)
        acc6 = framework.eval(model, FLAGS.batch_size, 5, 5, 1,
                              val_step, data_loader=val_data_loader)

        val_data_loader = get_loader(
            './data/val_pubmed.json', tokenizer, 10, 1, 1, FLAGS.batch_size, num_workers=2)
        acc7 = framework.eval(model, FLAGS.batch_size, 10, 1, 1,
                              val_step, data_loader=val_data_loader)

        val_data_loader = get_loader(
            './data/val_pubmed.json', tokenizer, 10, 5, 1, FLAGS.batch_size, num_workers=2)
        acc8 = framework.eval(model, FLAGS.batch_size, 10, 5, 1,
                              val_step, data_loader=val_data_loader)

        tabel.add_row(
            [round(100*acc1, 4), round(100*acc2, 4), round(100*acc3, 4), round(100*acc4, 4),
             round(100*acc5, 4), round(100*acc6, 4), round(100*acc7, 4), round(100*acc8, 4)])
        print(tabel)
