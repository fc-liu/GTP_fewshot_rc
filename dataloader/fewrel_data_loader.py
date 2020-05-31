import json
import os
import multiprocessing
import numpy as np
import random
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

    def __init__(self, file_name,  bert_tokenizer, N, K, Q, na_rate=0, max_length=512):
        '''
        file_name: Json file storing the data in the following format
            {
                "P155": # relation id
                    [
                        {
                            # head entity [word, id, location]
                            "h": ["song for a future generation", "Q7561099", [[16, 17, ...]]],
                            # tail entity [word, id, location]
                            "t": ["whammy kiss", "Q7990594", [[11, 12]]],
                            "token": ["Hot", "Dance", "Club", ...], # sentence
                        },
                        ...
                    ],
                "P177":
                    [
                        ...
                    ]
                ...
            }
        max_length: The length that all the sentences need to be extend to.
        case_sensitive: Whether the data processing is case-sensitive, default as False.
        reprocess: Do the pre-processing whether there exist pre-processed files, default as False.
        cuda: Use cuda or not, default as True.
        '''
        super(BertEMDataset, self).__init__()
        self.max_length = max_length
        self.bertTokenizer = bert_tokenizer
        global tokenizer
        tokenizer = self.bertTokenizer
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate

        if not os.path.exists(file_name):
            raise Exception("[ERROR] Data file doesn't exist")

        self.json_data = json.load(open(file_name, "r"))
        # self.data = {}
        print("Finish loading file")
        self.classes = list(self.json_data.keys())

        self.__init_process_data__(self.json_data)
        print("Finish init process data")

    def __init_process_data__(self, raw_data):
        def insert_and_tokenize(tokenizer, tokens, pos1, pos2, marker1, marker2):
            tokens.insert(pos2[-1]+1, marker2[-1])
            tokens.insert(pos2[0], marker2[0])
            tokens.insert(pos1[-1]+1, marker1[-1])
            tokens.insert(pos1[0], marker1[0])
            tokens = tokens.copy()

            tokens = tokenizer.tokenize(" ".join(tokens))

            return tokens

        for rel in self.classes:
            for ins in self.json_data[rel]:
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

                # tokens_dict = self.bertTokenizer.encode_plus(
                #     tokens, add_special_tokens=False, is_pretokenized=True) # max_length=self.max_length,pad_to_max_length=True

                # tokens_ids = tokens_dict['input_ids']
                # mask = tokens_dict['attention_mask']
                # ins['tokens_idx'] = tokens_ids
                # ins['mask'] = mask
                ins["pos1"] = pos1
                ins['pos2'] = pos2
                ins["raw_tokens"] = ins['tokens']
                ins['tokens'] = tokens

                if len(tokens) > self.max_length:
                    raise Exception("sequence too long")

        print("init process data finish")

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
        try:
            sampled_classes = random.sample(self.classes, self.N+1)
            target_classes = sampled_classes[:-1]
        except Exception:
            sampled_classes = random.sample(self.classes, self.N)
            target_classes = sampled_classes

        na_classes = sampled_classes[-1]

        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                list(range(len(self.json_data[class_name]))),
                self.K + self.Q, False)
            count = 0
            for j in indices:
                instance = self.json_data[class_name][j]
                if count < self.K:
                    self.__additem__(support_set, instance)
                else:
                    self.__additem__(query_set, instance)
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                list(range(len(self.json_data[cur_class]))),
                1, False)[0]
            instance = self.json_data[cur_class][index]
            self.__additem__(query_set, instance)
        query_label += [self.N] * Q_na

        return support_set, query_set, query_label

    def __len__(self):
        return sys.maxsize


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
    batch_label = []
    raw_support_sets, raw_query_sets, query_labels = zip(*data)

    # compute max length
    support_sets = idx_and_mask(raw_support_sets)
    query_sets = idx_and_mask(raw_query_sets)

    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(
            batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(
            batch_query[k], 0)
    batch_label = torch.tensor(batch_label)

    return batch_support, batch_query, batch_label


def get_loader(file_path, tokenizer, N, K, Q, batch_size, max_length=FLAGS.max_sentence_length,
               num_workers=1, na_rate=0):
    dataset = BertEMDataset(
        file_path, tokenizer, N, K, Q, max_length=max_length, na_rate=na_rate)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)
