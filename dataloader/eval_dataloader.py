import json
import os
import multiprocessing
import numpy as np
import random
from transformers import BertTokenizer
import torch
from torch.autograd import Variable
import torch.utils.data as data
from configure import FLAGS
import utils
import sys
E11 = FLAGS.e11
E12 = FLAGS.e12
E21 = FLAGS.e21
E22 = FLAGS.e22
# PAD = "[PAD]"

tokenizer = None


class BertEMDataset(data.Dataset):

    def __init__(self, file_name,  bert_tokenizer, na_rate=0, max_length=512):
        super(BertEMDataset, self).__init__()
        self.max_length = max_length
        self.bertTokenizer = bert_tokenizer
        global tokenizer
        tokenizer = self.bertTokenizer
        self.N = 0
        self.K = 0
        self.Q = 0

        if not os.path.exists(file_name):
            raise Exception("[ERROR] Data file doesn't exist")

        self.json_data = json.load(open(file_name, "r"))
        # self.data = {}
        # print("Finish loading file")
        self.task_num = len(self.json_data)

        self.__init_process_data__(self.json_data)
        # print("Finish init process data")

    def __init_process_data__(self, raw_data):

        def insert_and_tokenize(tokenizer, tokens, pos1, pos2, marker1, marker2):
            tokens.insert(pos2[-1]+1, marker2[-1])
            tokens.insert(pos2[0], marker2[0])
            tokens.insert(pos1[-1]+1, marker1[-1])
            tokens.insert(pos1[0], marker1[0])
            tokens = tokens.copy()

            tokens = tokenizer.tokenize(" ".join(tokens))

            return tokens

        def __process_ins(ins):
            pos1 = ins['h'][2][0]
            pos2 = ins['t'][2][0]
            words = ins['tokens']

            if pos1[0] > pos2[0]:
                tokens = insert_and_tokenize(self.bertTokenizer, words, pos2, pos1, [
                    E21, E22], [E11, E12])
            else:
                tokens = insert_and_tokenize(self.bertTokenizer, words, pos1, pos2, [
                    E11, E12], [E21, E22])

            tokens.insert(0, FLAGS.cls)

            pos1 = [tokens.index(FLAGS.e11), tokens.index(FLAGS.e12)]
            pos2 = [tokens.index(FLAGS.e21), tokens.index(FLAGS.e22)]

            if len(tokens) >= self.max_length:
                max_right = max(pos2[-1], pos1[-1])
                min_left = min(pos1[0], pos2[0])
                gap_length = max_right-min_left
                if gap_length+1 > self.max_length:
                    tokens = [FLAGS.cls, FLAGS.e11, FLAGS.e12,
                              FLAGS.e21, FLAGS.e22, FLAGS.sep]
                elif max_right+1 < self.max_length:
                    tokens = tokens[:self.max_length-1]
                else:
                    tokens = tokens[min_left:max_right]
                    tokens.insert(0, FLAGS.cls)
                    pos1[0] = pos1[0]-min_left+1
                    pos1[-1] = pos1[-1]-min_left+1
                    pos2[0] = pos2[0]-min_left+1
                    pos2[-1] = pos2[-1]-min_left+1

            tokens.append(FLAGS.sep)

            ins["pos1"] = pos1
            ins['pos2'] = pos2
            ins["raw_tokens"] = ins['tokens']
            ins['tokens'] = tokens

            if len(tokens) > self.max_length:
                raise Exception("sequence too long")

        for task in raw_data:
            for meta_rel in task['meta_train']:
                for ins in meta_rel:
                    __process_ins(ins)

            __process_ins(task['meta_test'])

        # print("init process data finish")

    def __additem__(self, d, instance):
        word = instance['tokens']
        pos1 = instance['pos1']
        pos2 = instance['pos2']
        # mask = instance['mask']

        pos1 = torch.tensor(pos1).long()
        pos2 = torch.tensor(pos2).long()
        # word = torch.tensor(word).long()
        # mask = torch.tensor(mask).long()
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        # d['mask'].append(mask)

    def __getitem__(self, index):
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}

        task_data = self.json_data[index]

        for meta_rel in task_data['meta_train']:
            for ins in meta_rel:
                self.__additem__(support_set, ins)
        self.__additem__(query_set, task_data['meta_test'])
        return support_set, query_set

    def __len__(self):
        return self.task_num


def compute_max_length(batch_sets):
    max_length = 0
    for set_words in batch_sets:
        for words in set_words['word']:
            if len(words) > max_length:
                max_length = len(words)

    return max_length


def idx_and_mask(batch_sets):
    global tokenizer
    # batch_sets = batch_sets.copy()
    max_length = compute_max_length(batch_sets)
    sets = []
    for set_item in batch_sets:
        set_item = set_item.copy()
        words_list = set_item['word']
        support_word = []
        for words in words_list:
            tokens_dict = tokenizer.encode_plus(
                words, add_special_tokens=False, is_pretokenized=True, max_length=max_length, pad_to_max_length=True)
            tokens_ids = tokens_dict['input_ids']
            mask = tokens_dict['attention_mask']
            tokens_ids = torch.tensor(tokens_ids).long()
            mask = torch.tensor(mask).long()
            support_word.append(tokens_ids)
            set_item['mask'].append(mask)

        set_item['word'] = support_word
        sets.append(set_item)

    return sets


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    raw_support_sets, raw_query_sets = zip(*data)

    # compute max length
    support_sets = idx_and_mask(raw_support_sets)
    query_sets = idx_and_mask(raw_query_sets)

    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(
            batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(
            batch_query[k], 0)

    return batch_support, batch_query


def get_loader(file_path, tokenizer, max_length=FLAGS.max_sentence_length,
               num_workers=1, na_rate=0):
    dataset = BertEMDataset(
        file_path, tokenizer, max_length=max_length, na_rate=na_rate)
    data_loader = data.DataLoader(dataset=dataset,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased", do_basic_tokenize=False)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [FLAGS.e11, FLAGS.e12, FLAGS.e21, FLAGS.e22]})
    # dataset = BertEMDataset("data/sample.json", tokenizer)
    dataloader = get_loader("data/sample.json", tokenizer)
    # for task in dataloader:
    #     print(1)
