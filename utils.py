# import tensorflow as tf
import numpy as np
import torch
from configure import FLAGS
import os
class2label = {'Other': 0,
               'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
               'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
               'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
               'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
               'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
               'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
               'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
               'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
               'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}

label2class = {0: 'Other',
               1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
               3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
               5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
               7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
               9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
               11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
               13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
               15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
               17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}
gpu_aval = torch.cuda.is_available()


def pad_and_mask(batch_samples, max_seq_length):
    sentences = np.asarray(batch_samples)
    max_length = 0
    for sen_tokens in sentences:
        length = len(sen_tokens)
        if length > max_length:
            if length <= max_seq_length:
                max_length = length
            else:
                max_length = max_seq_length
                break
    ret_sentences = []
    sentences_mask = []
    for sen_tokens in sentences:
        if len(sen_tokens) > max_seq_length:
            sen_tokens = sen_tokens[:max_seq_length]
        mask = [1]*len(sen_tokens)
        pad_length = max_length-len(sen_tokens)
        sen_tokens = np.append(
            sen_tokens, [FLAGS.pad]*(pad_length))
        mask.extend([0]*pad_length)
        ret_sentences.append(sen_tokens)
        sentences_mask.append(mask)

    return ret_sentences, sentences_mask


def convert_batch_to_pair_features(tokenizer, batch_support, batch_query, batch_size):
    global gpu_aval
    """
    batch_support: batch_size,N,K,Seq_len
    batch_query: batch_size, Q
    """
    batch_samples = []
    support_sentences = batch_support["word"]
    query_sentences = batch_query["word"]
    pos1 = batch_support['pos1']
    pos2 = batch_support['pos2']
    support_sentences = support_sentences.reshape(
        [batch_size, -1, FLAGS.max_sentence_length]).tolist()
    query_sentences = query_sentences.reshape(
        [batch_size, -1, FLAGS.max_sentence_length]).tolist()
    for i in range(batch_size):
        for query in query_sentences[i]:
            for support in support_sentences[i]:
                q_sen = query.copy()
                q_sen.append('[SEP]')
                q_sen.extend(support)
                batch_samples.append([q_sen, [-1, -1], [-1, -1]])
    tokens, pos1, pos2, mask, seg_id = bert_tokenize_index_and_mask(
        tokenizer, batch_samples, seg_id=True)
    if gpu_aval and FLAGS.use_gpu:
        tokens = torch.LongTensor(tokens).to(FLAGS.paral_cuda[0])
        pos1 = torch.LongTensor(pos1).to(FLAGS.paral_cuda[0])
        pos2 = torch.LongTensor(pos2).to(FLAGS.paral_cuda[0])
        mask = torch.LongTensor(mask).to(FLAGS.paral_cuda[0])
        seg_id = torch.LongTensor(seg_id).to(FLAGS.paral_cuda[0])

    return (tokens, seg_id, mask)


def convert_batch_to_features(tokenizer, batch_samples):
    global gpu_aval
    sentences = batch_samples['word']
    pos1 = batch_samples['pos1']
    pos2 = batch_samples['pos2']
    batch = []
    for i in range(len(pos1)):
        batch.append([sentences[i], pos1[i], pos2[i]])

    tokens, pos1, pos2, mask = bert_tokenize_index_and_mask(
        tokenizer, batch)
    if gpu_aval and FLAGS.use_gpu:
        tokens = torch.LongTensor(tokens).to(FLAGS.paral_cuda[0])
        pos1 = torch.LongTensor(pos1).to(FLAGS.paral_cuda[0])
        pos2 = torch.LongTensor(pos2).to(FLAGS.paral_cuda[0])
        mask = torch.Tensor(mask).to(FLAGS.paral_cuda[0])

    return (tokens, pos1, pos2, mask)


def convert_batch_tokens_to_features(tokenizer, batch_tokens):
    global gpu_aval
    sentences = batch_samples['word']
    pos1 = batch_samples['pos1']
    pos2 = batch_samples['pos2']
    batch = []
    for i in range(len(pos1)):
        batch.append([sentences[i], pos1[i], pos2[i]])

    tokens, pos1, pos2, mask = bert_tokenize_index_and_mask(
        tokenizer, batch)
    if gpu_aval and FLAGS.use_gpu:
        tokens = torch.LongTensor(tokens).to(FLAGS.paral_cuda[0])
        pos1 = torch.LongTensor(pos1).to(FLAGS.paral_cuda[0])
        pos2 = torch.LongTensor(pos2).to(FLAGS.paral_cuda[0])
        mask = torch.Tensor(mask).to(FLAGS.paral_cuda[0])

    return (tokens, pos1, pos2, mask)


def bert_tokenize_index_and_mask(tokenizer, batch_samples, require_wp_tokens=False, seg_id=False):
    sentences_tokens = []
    sentence_indexed_tokens = []
    samples = np.asarray(batch_samples)
    sentences = samples[:, 0]
    pos1 = samples[:, 1]
    pos2 = samples[:, 2]
    pos1 = pos1.tolist()
    pos2 = pos2.tolist()
    removed_sentence_id = []
    batch_seg = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        # sentence = " ".join(sentence)
        tokens = []
        tokens.append("[CLS]")
        for token in sentence:
            if not token:
                continue
            token = tokenizer.tokenize(token)
            if len(token) > 0:
                if token[0] == FLAGS.e11:
                    pos1[i][0] = len(tokens)
                elif token[0] == FLAGS.e12:
                    pos1[i][-1] = len(tokens)+1
                elif token[0] == FLAGS.e21:
                    pos2[i][0] = len(tokens)
                elif token[0] == FLAGS.e22:
                    pos2[i][-1] = len(tokens)+1

                tokens.extend(token)
            else:
                tokens.append(FLAGS.unk)

        if len(tokens) >= FLAGS.max_sentence_length:
            max_right = max(pos2[i][-1], pos1[i][-1])
            min_left = min(pos1[i][0], pos2[i][0])
            gap_length = max_right-min_left
            if gap_length+1 > FLAGS.max_sentence_length:
                removed_sentence_id.append(i)
                # return False
            elif max_right+1 < FLAGS.max_sentence_length:
                tokens = tokens[:FLAGS.max_sentence_length-1]
            else:
                tokens = tokens[min_left:max_right]
                tokens.insert(0, "[CLS]")
                pos1[i][0] = pos1[i][0]-min_left+1
                pos1[i][-1] = pos1[i][-1]-min_left+1
                pos2[i][0] = pos2[i][0]-min_left+1
                pos2[i][-1] = pos2[i][-1]-min_left+1
        tokens.append("[SEP]")
        sentences_tokens.append(tokens)

    if seg_id:
        for sentence in sentences_tokens:
            seg_pos = sentence.index("[SEP]")
            seg = [0]*seg_pos
            seg.extend([1]*(len(sentence)-seg_pos))
            batch_seg.append(seg)

    for idx in removed_sentence_id:
        pos1.pop(idx)
        pos2.pop(idx)
    sentences_tokens, mask = pad_and_mask(
        sentences_tokens, FLAGS.max_sentence_length)
    for sentence in sentences_tokens:
        indexed_tokens = tokenizer.convert_tokens_to_ids(
            sentence)
        sentence_indexed_tokens.append(indexed_tokens)
    sen_length = len(sentence_indexed_tokens[0])
    for seg in batch_seg:
        seg_len = len(seg)
        if seg_len < sen_length:
            seg.extend([1]*(sen_length-seg_len))
    if require_wp_tokens:
        if seg_id:
            return sentence_indexed_tokens, pos1, pos2, mask, batch_seg, sentences_tokens
        else:
            return sentence_indexed_tokens, pos1, pos2, mask, sentences_tokens
    else:
        if seg_id:
            return sentence_indexed_tokens, pos1, pos2, mask, batch_seg
        else:
            return sentence_indexed_tokens, pos1, pos2, mask


def lengths2mask(lengths, max_length, byte=False, negation=False):
    """
    Lengths to Mask, lengths start from 0
    :param lengths:     (batch, )
        tensor([ 1,  2,  5,  3,  4])
    :param max_length:  int, max length
        5
    :param byte:        Return a ByteTensor if True, else a Float Tensor
    :param negation:
        False:
            tensor([[ 1.,  0.,  0.,  0.,  0.],
                    [ 1.,  1.,  0.,  0.,  0.],
                    [ 1.,  1.,  1.,  1.,  1.],
                    [ 1.,  1.,  1.,  0.,  0.],
                    [ 1.,  1.,  1.,  1.,  0.]])
        True:
            tensor([[ 0.,  1.,  1.,  1.,  1.],
                    [ 0.,  0.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  1.]])
    :return:
        ByteTensor/FloatTensor
    """
    batch_size = lengths.size(0)
    assert max_length >= torch.max(lengths).item()
    assert torch.min(lengths).item() >= 0

    range_i = torch.arange(
        1, max_length + 1, dtype=torch.long).expand(batch_size, max_length).to(lengths.device)
    batch_lens = lengths.unsqueeze(-1).expand(batch_size, max_length)

    if negation:
        mask = batch_lens < range_i
    else:
        mask = torch.ge(batch_lens, range_i)

    if byte:
        return mask.detach()
    else:
        return mask.float().detach()


def position2mask(position, max_length, byte=False, negation=False):
    """
    Position to Mask, position start from 0
    :param position:     (batch, )
        tensor([ 1,  2,  0,  3,  4])
    :param max_length:  int, max length
        5
    :param byte:        Return a ByteTensor if True, else a Float Tensor
    :param negation:
        False:
            tensor([[ 0.,  1.,  0.,  0.,  0.],
                    [ 0.,  0.,  1.,  0.,  0.],
                    [ 1.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  1.,  0.],
                    [ 0.,  0.,  0.,  0.,  1.]])
        True:
            tensor([[ 1.,  0.,  1.,  1.,  1.],
                    [ 1.,  1.,  0.,  1.,  1.],
                    [ 0.,  1.,  1.,  1.,  1.],
                    [ 1.,  1.,  1.,  0.,  1.],
                    [ 1.,  1.,  1.,  1.,  0.]])
    :return:
        ByteTensor/FloatTensor
    """
    batch_size = position.size(0)
    try:
        assert max_length >= torch.max(position).item()+1
    except Exception as e:
        print(e)
        return
    assert torch.min(position).item() >= 0

    range_i = torch.arange(0, max_length, dtype=torch.long).expand(
        batch_size, max_length).to(position.device)

    batch_position = position.unsqueeze(-1).expand(batch_size, max_length)

    if negation:
        mask = torch.ne(batch_position, range_i)
    else:
        mask = torch.eq(batch_position, range_i)

    if byte:
        return mask.detach()
    else:
        return mask.float().detach()


def span2mask(start, end, max_length, byte=False):
    """
    :param start: Start Position
    :param end:   End Position
    :param max_length: Max Length, so max length must big or equal than max of end
    :param byte:  Byte
    :return:
        start -> 0
        end -> 1
        max_length -> 5
            => [1, 0, 0, 0, 0]

        start -> 1
        end -> 2
        max_length -> 5
            => [0, 1, 0, 0, 0]

        start -> 1
        end -> 3
        max_length -> 5
            => [0, 1, 1, 0, 0]
    """
    assert torch.max(end).item() <= max_length
    assert torch.min(end).item() > 0
    mask = lengths2mask(end, max_length) - lengths2mask(start, max_length)
    if byte:
        return mask.byte()
    else:
        return mask


# def initializer():
#     return tf.keras.initializers.glorot_normal()


def load_word2vec(word2vec_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(
        np.float32) * np.sqrt(2.0 / len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load word2vec file {0}".format(word2vec_path))
    with open(word2vec_path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            idx = vocab.vocabulary_.get(word)
            if idx != 0:
                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return initW


def load_glove(word2vec_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(
        np.float32) * np.sqrt(2.0 / len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load glove file {0}".format(word2vec_path))
    f = open(word2vec_path, 'r', encoding='utf8')
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        idx = vocab.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = embedding
    return initW


if __name__ == "__main__":
    e = torch.LongTensor([[2, 4], [1, 3]])
    mask = span2mask(e[:, 0], e[:, 1], 5)
    inputs = torch.rand([2, 5])
    res = torch.sum(inputs * mask, 1) / torch.sum(mask, 1).float()
    print(torch.sum(inputs * mask, 1))
    print(res)
    print(inputs)
    print(mask)
