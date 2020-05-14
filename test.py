from dataloader.fewrel_data_loader import get_loader, BertEMDataset
from configure import FLAGS
from transformers import BertTokenizer


if __name__ == "__main__":
    # train_data_loader = JSONFileDataLoader(
    #     './data/train.json', './data/glove.6B.50d.json', max_length=125, case_sensitive=True, reprocess=FLAGS.reproc_data, bertTokenizer=None, na_rate=FLAGS.na_rate)
    # dataloader = get_loader('./data/train.json', 5, 1, 1, 2)
    # for i in range(10):
    #     item = next(dataloader)
    #     print(item)
    tokenizer = BertTokenizer.from_pretrained(
        FLAGS.bert_model, do_basic_tokenize=False)
    # tokenizer = AlbertTokenizer.from_pretrained(
    #     FLAGS.bert_model, do_basic_tokenize=False)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [FLAGS.e11, FLAGS.e12, FLAGS.e21, FLAGS.e22]})
    # bert_model.resize_token_embeddings(len(tokenizer))
    # dataset = BertEMDataset(file_name='./data/val_pubmed.json', bert_tokenizer=tokenizer,
    #                        N=5,K=5,Q=1, max_length=FLAGS.max_sentence_length, na_rate=FLAGS.na_rate)

    loader = get_loader('./data/val_pubmed.json',
                        tokenizer, 5, 2, 1, 3, num_workers=1)
    # tokens = 'fdsvwfgbgfdngdf'
    # tokens = list(tokens)
    for i in range(1):
        item = next(loader)
        print(item)
    # def f(tokens):
    #     tokens.insert(2, 'we')
    # f(tokens)
    # print(tokens)
