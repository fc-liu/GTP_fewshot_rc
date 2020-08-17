from model.relation_representation_model import EntityMarkerEncoder
from configure import FLAGS
import sklearn.exceptions
import warnings
import utils
from dataloader.eval_dataloader import get_loader
import sys
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
import os
import time
import numpy as np
import torch
import prettytable as pt
from model.interact_proto import GlobalTransformedProtoNet,Proto


N = FLAGS.N
K = FLAGS.K

relation_encoder = EntityMarkerEncoder()

bert_model = BertModel.from_pretrained(
    FLAGS.bert_model, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(
    FLAGS.bert_model, do_basic_tokenize=False, truncation=False)
tokenizer.add_special_tokens(
    {"additional_special_tokens": [FLAGS.e11, FLAGS.e12, FLAGS.e21, FLAGS.e22]})
bert_model.resize_token_embeddings(len(tokenizer))

ckpt_file_path = "./checkpoint/fewrel/{}".format(FLAGS.ckpt_name)

max_length = FLAGS.max_sentence_length

test_data_loader = get_loader(
    FLAGS.test_file, tokenizer, num_workers=1)
gpu_aval = torch.cuda.is_available()


model = GlobalTransformedProtoNet(tokenizer, bert_model,
                                  relation_encoder, max_length)
# model = Proto(tokenizer, bert_model,
#                                   relation_encoder, max_length)
if os.path.exists(ckpt_file_path):
    if FLAGS.paral_cuda[0] >= 0:
        ckpt = torch.load(ckpt_file_path, map_location=lambda storage,
                          loc: storage.cuda(FLAGS.paral_cuda[0]))
    else:
        ckpt = torch.load(
            ckpt_file_path, map_location=lambda storage, loc: storage.cpu())
    model.load_state_dict(ckpt["state_dict"])

if gpu_aval:
    model = model.to(FLAGS.paral_cuda[0])


def eval(model, data_loader=None):
    res = []
    model.eval()
    with torch.no_grad():
        for support, query in data_loader:
            # logits, pred = self.predict(
            #     model, support, query, B, N, K, Q, label)
            support = [support['word'].to(FLAGS.paral_cuda[0]), support['pos1'].to(FLAGS.paral_cuda[0]),
                       support['pos2'].to(FLAGS.paral_cuda[0]), support['mask'].to(FLAGS.paral_cuda[0])]
            query = [query['word'].to(FLAGS.paral_cuda[0]), query['pos1'].to(FLAGS.paral_cuda[0]),
                     query['pos2'].to(FLAGS.paral_cuda[0]), query['mask'].to(FLAGS.paral_cuda[0])]
            _, pred = model(support, query, 1, N, K, 1)
            pred = pred.detach().cpu().item()
            res.append(pred)

    return res


with torch.no_grad():
    res = eval(model,  data_loader=test_data_loader)
    print(res)
