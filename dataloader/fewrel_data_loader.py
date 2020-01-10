import json
import os
import multiprocessing
import numpy as np
import random
import torch
from torch.autograd import Variable
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import utils
import torch.utils.data as data
from configure import FLAGS
E11 = FLAGS.e11
E12 = FLAGS.e12
E21 = FLAGS.e21
E22 = FLAGS.e22
# PAD = "[PAD]"


class FileDataLoader:
    def next_batch(self, B, N, K, Q):
        '''
        B: batch size.
        N: the number of relations for each batch
        K: the number of support instances for each relation
        Q: the number of query instances for each relation
        return: support_set, query_set, query_label
        '''
        raise NotImplementedError


class JSONFileDataLoader(data.Dataset):
    def _load_preprocessed_file(self):
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        processed_data_dir = '_processed_data'
        if not os.path.isdir(processed_data_dir):
            return False
        word_npy_file_name = os.path.join(
            processed_data_dir, name_prefix + '_word.npy')
        pos1_npy_file_name = os.path.join(
            processed_data_dir, name_prefix + '_pos1.npy')
        pos2_npy_file_name = os.path.join(
            processed_data_dir, name_prefix + '_pos2.npy')
        mask_npy_file_name = os.path.join(
            processed_data_dir, name_prefix + '_mask.npy')
        length_npy_file_name = os.path.join(
            processed_data_dir, name_prefix + '_length.npy')
        rel2scope_file_name = os.path.join(
            processed_data_dir, name_prefix + '_rel2scope.json')
        if not os.path.exists(word_npy_file_name) or \
           not os.path.exists(pos1_npy_file_name) or \
           not os.path.exists(pos2_npy_file_name) or \
           not os.path.exists(mask_npy_file_name) or \
           not os.path.exists(length_npy_file_name) or \
           not os.path.exists(rel2scope_file_name):
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name, allow_pickle=True)
        self.data_pos1 = np.load(pos1_npy_file_name)
        self.data_pos2 = np.load(pos2_npy_file_name)
        self.data_mask = np.load(mask_npy_file_name)
        self.data_length = np.load(length_npy_file_name)
        self.rel2scope = json.load(open(rel2scope_file_name))
        if self.data_word.shape[1] != self.max_length:
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def __init__(self, file_name, word_vec_file_name, max_length=40, exp_per_type=-1,
                 case_sensitive=True, reprocess=False, cuda=True, bertTokenizer=None, na_rate=0):
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
        word_vec_file_name: Json file storing word vectors in the following format
            [
                {'word': 'the', 'vec': [0.418, 0.24968, ...]},
                {'word': ',', 'vec': [0.013441, 0.23682, ...]},
                ...
            ]
        max_length: The length that all the sentences need to be extend to.
        case_sensitive: Whether the data processing is case-sensitive, default as False.
        reprocess: Do the pre-processing whether there exist pre-processed files, default as False.
        cuda: Use cuda or not, default as True.
        '''
        self.file_name = file_name
        self.expam_per_type = exp_per_type
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        self.cuda = cuda
        self.bertTokenizer = bertTokenizer
        self.na_rate = na_rate

        # Try to load pre-processed files:
        if reprocess or not self._load_preprocessed_file():
            # Check files
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")

            # Load files
            print("Loading data file...")
            self.ori_data = json.load(open(self.file_name, "r"))
            print("Finish loading")

            # Eliminate case sensitive
            if not case_sensitive:
                print("Elimiating case sensitive problem...")
                for relation in self.ori_data:
                    for ins in self.ori_data[relation]:
                        for i in range(len(ins['tokens'])):
                            ins['tokens'][i] = ins['tokens'][i].lower()
                print("Finish eliminating")

            # Pre-process data
            print("Pre-processing data...")
            self.instance_tot = 0
            for relation in self.ori_data:
                if self.expam_per_type > 0:
                    intances = self.ori_data[relation]
                    intances = random.sample(intances, self.expam_per_type)
                    self.ori_data[relation] = intances
                self.instance_tot += len(self.ori_data[relation])
            self.data_word = []
            self.data_pos1 = []
            self.data_pos2 = []
            self.data_mask = []
            self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
            self.rel2scope = {}  # left close right open
            i = 0
            for relation in self.ori_data:
                self.rel2scope[relation] = [i, i]
                for ins in self.ori_data[relation]:
                    head = ins['h'][0]
                    tail = ins['t'][0]
                    pos1 = ins['h'][2][0]
                    pos1 = [pos1[0], pos1[-1]+1]
                    pos2 = ins['t'][2][0]
                    pos2 = [pos2[0], pos2[-1]+1]
                    words = ins['tokens']
                    cur_ref_data_word = [None]*max_length
                    words_copy = list(words)
                    if pos1[0] < pos2[0]:
                        pos1[-1] = pos1[-1]+1
                        pos2[0] = pos2[0]+2
                        pos2[-1] = pos2[-1]+3
                        words_copy.insert(pos1[0], E11)
                        words_copy.insert(pos1[-1], E12)
                        words_copy.insert(pos2[0], E21)
                        words_copy.insert(pos2[-1], E22)
                        gap_length = pos2[-1]-pos1[0]
                        min_left = pos1[0]
                        max_right = pos2[-1]
                    else:
                        pos2[-1] = pos2[-1]+1
                        pos1[0] = pos1[0]+2
                        pos1[-1] = pos1[-1]+3
                        words_copy.insert(pos2[0], E21)
                        words_copy.insert(pos2[-1]+1, E22)
                        words_copy.insert(pos1[0]+2, E11)
                        words_copy.insert(pos1[-1]+3, E12)
                        gap_length = pos1[-1]-pos2[0]
                        min_left = pos2[0]
                        max_right = pos1[-1]+3
                    j = len(words_copy)
                    if j > max_length:
                        print("#######long sentence length:{}##########".format(j))
                        if gap_length + 6 > max_length:
                            # self.data_mask.pop(i)
                            print(
                                "#################discard long sentence, length between p1 and p2 is :{}############".format(gap_length))
                            continue
                        else:
                            cur_ref_data_word[0:max_right+1 -
                                              min_left] = cur_ref_data_word[min_left:max_right+1]
                            j = len(cur_ref_data_word)
                            pos1[0] = pos1[0]-min_left
                            pos1[-1] = pos1[-1]-min_left
                            pos2[0] = pos2[0]-min_left
                            pos2[-1] = pos2[-1]-min_left

                    cur_ref_data_word[:min(j, max_length)] = words_copy[:min(
                        j, max_length)]
                    seq_len = min(j, max_length)
                    self.data_length[i] = seq_len

                    mask = [1]*seq_len
                    for j in range(j, max_length):
                        if j >= self.data_length[i]:
                            mask.append(0)
                        # else:
                        #     self.data_mask[i][j] = 1
                    i += 1
                    self.data_word.append(cur_ref_data_word)
                    self.data_pos1.append(pos1)
                    self.data_pos2.append(pos2)

                    self.data_mask.append(mask)
                self.rel2scope[relation][1] = i

            self.data_word = np.asarray(self.data_word)
            self.data_mask = np.asarray(self.data_mask)
            self.data_pos1 = np.asarray(self.data_pos1)
            self.data_pos2 = np.asarray(self.data_pos2)
            print("Finish pre-processing")

            print("Storing processed files...")
            name_prefix = '.'.join(file_name.split('/')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)
            np.save(os.path.join(processed_data_dir,
                                 name_prefix + '_word.npy'), self.data_word)
            np.save(os.path.join(processed_data_dir,
                                 name_prefix + '_pos1.npy'), self.data_pos1)
            np.save(os.path.join(processed_data_dir,
                                 name_prefix + '_pos2.npy'), self.data_pos2)
            np.save(os.path.join(processed_data_dir,
                                 name_prefix + '_mask.npy'), self.data_mask)
            np.save(os.path.join(processed_data_dir,
                                 name_prefix + '_length.npy'), self.data_length)
            json.dump(self.rel2scope, open(os.path.join(
                processed_data_dir, name_prefix + '_rel2scope.json'), 'w'))
            print("Finish storing")

    def next_one(self, N, K, Q):
        target_classes = random.sample(self.rel2scope.keys(), N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []
        
        Q_na = int(self.na_rate * Q)
        na_classes = list(filter(lambda x: x not in target_classes,
                                 self.rel2scope.keys()))

        for i, class_name in enumerate(target_classes):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(
                list(range(scope[0], scope[1])), K + Q, False)
            word = self.data_word[indices]
            pos1 = self.data_pos1[indices]
            pos2 = self.data_pos2[indices]
            mask = self.data_mask[indices]
            support_word, query_word, _ = np.split(word, [K, K + Q])
            support_pos1, query_pos1, _ = np.split(pos1, [K, K + Q])
            support_pos2, query_pos2, _ = np.split(pos2, [K, K + Q])
            support_mask, query_mask, _ = np.split(mask, [K, K + Q])
            support_set['word'].append(support_word)
            support_set['pos1'].append(support_pos1)
            support_set['pos2'].append(support_pos2)
            support_set['mask'].append(support_mask)
            query_set['word'].append(query_word)
            query_set['pos1'].append(query_pos1)
            query_set['pos2'].append(query_pos2)
            query_set['mask'].append(query_mask)
            query_label += [i] * Q

        for j in range(Q_na):
            cur_class=np.random.choice(na_classes,1,False)[0]
            scope = self.rel2scope[cur_class]
            indices = np.random.choice(
                list(range(scope[0], scope[1])), 1, False)[0]
            word = self.data_word[indices]
            pos1 = self.data_pos1[indices]
            pos2 = self.data_pos2[indices]
            mask = self.data_mask[indices]
            query_set['word'].append([word])
            query_set['pos1'].append([pos1])
            query_set['pos2'].append([pos2])
            query_set['mask'].append([mask])
        query_label += [N] * Q_na


        support_set['word'] = np.stack(support_set['word'], 0)
        support_set['pos1'] = np.stack(support_set['pos1'], 0)
        support_set['pos2'] = np.stack(support_set['pos2'], 0)
        support_set['mask'] = np.stack(support_set['mask'], 0)
        query_set['word'] = np.concatenate(query_set['word'], 0)
        query_set['pos1'] = np.concatenate(query_set['pos1'], 0)
        query_set['pos2'] = np.concatenate(query_set['pos2'], 0)
        query_set['mask'] = np.concatenate(query_set['mask'], 0)
        query_label = np.array(query_label)

        perm = np.random.permutation(N * Q+Q_na)
        query_set['word'] = query_set['word'][perm]
        query_set['pos1'] = query_set['pos1'][perm]
        query_set['pos2'] = query_set['pos2'][perm]
        query_set['mask'] = query_set['mask'][perm]
        query_label = query_label[perm]

        return support_set, query_set, query_label

    def next_batch(self, B, N, K, Q):
        support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        label = []
        for one_sample in range(B):
            current_support, current_query, current_label = self.next_one(
                N, K, Q)

            support['word'].append(current_support['word'])
            support['pos1'].append(current_support['pos1'])
            support['pos2'].append(current_support['pos2'])
            support['mask'].append(current_support['mask'])
            query['word'].append(current_query['word'])
            query['pos1'].append(current_query['pos1'])
            query['pos2'].append(current_query['pos2'])
            query['mask'].append(current_query['mask'])

            label.append(current_label)
        label = torch.from_numpy(
            np.stack(label, 0).astype(np.int64)).long()
        support['word'] = np.asarray(
            support['word']).reshape(-1, self.max_length)
        support['pos1'] = np.asarray(
            support['pos1']).reshape(-1, 2)
        support['pos2'] = np.asarray(
            support['pos2']).reshape(-1, 2)
        support['mask'] = np.asarray(
            support['mask']).reshape(-1, self.max_length)
        query['word'] = np.asarray(query['word']).reshape(-1, self.max_length)
        query['pos1'] = np.asarray(query['pos1']).reshape(-1, 2)
        query['pos2'] = np.asarray(query['pos2']).reshape(-1, 2)
        query['mask'] = np.asarray(query['mask']).reshape(-1, self.max_length)

        # To cuda
        if self.cuda:
            label = label.to(FLAGS.paral_cuda[0])

        return support, query, label
