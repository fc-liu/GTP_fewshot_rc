import argparse
import sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--name_mapping", default="name_mapping",
                        type=str, help="mapping file of pretrain class to pretrain entity.")
    parser.add_argument("--wiki_dir", default="/home/lfc/data/wiki_v1.3/",
                        type=str, help="directory path of extraced wiki pretrain data")
    parser.add_argument("--wiki_file", default="/home/lfc/data/wiki_file_c1_filt",
                        type=str, help="directory path of extraced wiki pretrain data")

    # SemEval data
    parser.add_argument("--pretrain_path", default="data/TRAIN_FILE.TXT",
                        type=str, help="Path of train data")
    parser.add_argument("--ckpt_dir", default="checkpoint/semeval_pre_train/",
                        type=str, help="Path of train data")
    # parser.add_argument("--ckpt_name", default="gtp-woseg-sepexcludproto-layhead4",
    #                     type=str, help="Path of train data")
    parser.add_argument("--ckpt_name", default="multi",
                        type=str, help="Path of train data")
    parser.add_argument("--fewrel_ckpt_file", default="checkpoint/fewrel/bert_mg_1.pth",
                        type=str, help="Path of train data")
    parser.add_argument("--semeval_ckpt_file", default="checkpoint/semeval/bert.pth",
                        type=str, help="Path of train data")
    parser.add_argument("--pretrain_ckpt_file", default="checkpoint/pretrain/dual_pretrain.pth",
                        type=str, help="Path of train data")
    parser.add_argument("--pretrain_test_path", default="data/TEST_FILE_FULL.TXT",
                        type=str, help="Path of test data")

    parser.add_argument("--max_sentence_length", default=256,
                        type=int, help="Max sentence length in data")

    # bert config
    parser.add_argument("--e11", default="[e11]",
                        type=str, help="start of e1")
    parser.add_argument("--e12", default="[e12]",
                        type=str, help="end of e1")
    parser.add_argument("--e21", default="[e21]",
                        type=str, help="start of e2")
    parser.add_argument("--e22", default="[e22]",
                        type=str, help="end of e2")
    # parser.add_argument("--unk", default="[UNK]",
    #                     type=str, help="oov token")
    # parser.add_argument("--pad", default="[PAD]",
    #                     type=str, help="pad token")

    # albert special tokens
    parser.add_argument("--unk", default="[unk]",
                        type=str, help="oov token")
    parser.add_argument("--pad", default="[pad]",
                        type=str, help="pad token")
    parser.add_argument("--cls", default="[CLS]",
                        type=str, help="pad token")
    parser.add_argument("--sep", default="[SEP]",
                        type=str, help="pad token")

    parser.add_argument("--bert_model", default="/share/model/bert/cased_L-24_H-1024_A-16",  # default="bert-large-uncased",
                        type=str, help="bert model")
    # Model Hyper-parameters
    parser.add_argument("--mode", default="train",
                        type=str, help="train or eval")
    parser.add_argument("--rel_rep_model", default="em",
                        type=str, help="rm or em or emc")
    parser.add_argument("--N", default=5,
                        type=int, help="few-shot N")
    parser.add_argument("--K", default=5,
                        type=int, help="few-shot K")
    parser.add_argument("--Q", default=1,
                        type=int, help="few-shot K")
    # parser.add_argument("--bert_vocab", default="bert-base-uncased/bert-base-uncased-vocab.txt",
    #                     type=str, help="bert vocab")
    parser.add_argument("--hidden_size", default=1024,
                        type=int, help="Dimensionality of BERT hidden (default: 768)")
    parser.add_argument("--bert_layer", default=23,
                        type=int, help="bert layer to use")
    parser.add_argument("--attention_size", default=396,
                        type=int, help="Dimensionality of attention (default: 50)")
    parser.add_argument("--n_head", default=4,
                        type=int, help="head number of gtp")
    parser.add_argument("--fewrel_output_size", default=2024,
                        type=int, help="Dimensionality of relation representation (default: 512)")
    # Misc
    parser.add_argument("--desc", default="",
                        type=str, help="Description for model")
    parser.add_argument("--dropout_keep_prob", default=0.5,
                        type=float, help="Dropout keep probability of output layer (default: 0.5)")
    parser.add_argument("--l2_reg_lambda", default=1e-6,
                        type=float, help="L2 regularization lambda (default: 1e-5)")

    # Training parameters
    parser.add_argument("--epochs", default=1000000000,
                        type=int, help="finish pretrain model on semeval set after this many steps (default: 100)")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="Batch Size (default: 20)")
    parser.add_argument("--num_epochs", default=1000,
                        type=int, help="Number of training epochs (Default: 100)")
    parser.add_argument("--display_every", default=10,
                        type=int, help="Number of iterations to display training information")
    parser.add_argument("--evaluate_every", default=1000,
                        type=int, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument("--num_checkpoints", default=1000,
                        type=int, help="Number of checkpoints to store (default: 5)")
    parser.add_argument("--learning_rate", default=1e-5,
                        type=float, help="Which learning rate to start with (Default: 1.0)")
    parser.add_argument("--decay_rate", default=0.6,
                        type=float, help="Decay rate for learning rate (Default: 0.9)")

    parser.add_argument("--use_gpu", default=True,
                        type=bool, help="use gpu")
    parser.add_argument("--cuda", default='cuda:2',
                        type=str, help="use cuda")
    parser.add_argument("--paral_cuda", default=[5], type=int,
                        help="parallel cuda", nargs='+')
    # Visualization Parameters
    parser.add_argument("--checkpoint_dir", default=None,
                        type=str, help="Visualize this checkpoint")
    parser.add_argument("--seg_emb_size", default=256, type=int,
                        help="segment embedding size")
    # # Misc Parameters
    # parser.add_argument("--allow_soft_placement", default=True,
    #                     type=bool, help="Allow device soft device placement")
    # parser.add_argument("--log_device_placement", default=False,
    #                     type=bool, help="Log placement of ops on devices")
    # parser.add_argument("--gpu_allow_growth", default=True,
    #                     type=bool, help="Allow gpu memory growth")

    # # Embeddings
    # parser.add_argument("--embeddings", default=None,
    #                     type=str, help="Embeddings {'word2vec', 'glove100', 'glove300', 'elmo'}")
    # parser.add_argument("--embedding_size", default=300,
    #                     type=int, help="Dimensionality of word embedding (default: 300)")
    # parser.add_argument("--pos_embedding_size", default=50,
    #                     type=int, help="Dimensionality of relative position embedding (default: 50)")
    # parser.add_argument("--emb_dropout_keep_prob", default=0.7,
    #                     type=float, help="Dropout keep probability of embedding layer (default: 0.7)")
    parser.add_argument("--na_rate", default=0, type=float,
                        help="NOTA rate, need to be divided by 5, due to the implement")
    parser.add_argument("--pair", default=True, type=bool,
                        help="use bert pair")
    parser.add_argument("--desp", default="", type=str,
                        help="description of the model")
    parser.add_argument("--layer", default=1, type=int,
                        help="number of transformer layer")

    parser.add_argument("--test_file", default="data/sample.json", type=str,
                        help="test file path")

    parser.add_argument("--model_name", default="multi", type=str,
                        help="model name")

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)
    # print("")
    args = parser.parse_args()
    # for arg in vars(args):
    #     print("{}={}".format(arg.upper(), getattr(args, arg)))
    # print("")

    return args


FLAGS = parse_args()
# FLAGS.ALPHA = torch.Tensor([FLAGS.ALPHA]).cuda(FLAGS.paral_cuda[0])
