import os
import sklearn.metrics
import numpy as np
import sys
import time
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from configure import FLAGS
import prettytable as pt
import utils

gpu_aval = torch.cuda.is_available()
# device = torch.device("cuda:"+str(FLAGS.paral_cuda[0]))


class FewShotREModel(nn.Module):
    def __init__(self, sentence_encoder):
        '''
        sentence_encoder: Sentence encoder

        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = sentence_encoder
        self.cost = nn.CrossEntropyLoss()
        # self.cost=nn.MultiMarginLoss(margin=1)

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))


class FewShotREFramework:

    def __init__(self, train_data_loader_1shot, train_data_loader_5shot, val_data_loader, test1_data_loader, test2_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader_1shot = train_data_loader_1shot
        self.train_data_loader_5shot = train_data_loader_5shot
        self.val_data_loader = val_data_loader
        self.test1_data_loader = test1_data_loader
        self.test2_data_loader = test2_data_loader
        self.tabel = pt.PrettyTable(
            ["step", "val_wiki", "val_pubmed"])

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    # def predict(self, model, support, query, B, N, K, Q, label):

    #     # support = utils.convert_batch_to_features(model.tokenizer, support)
    #     # query = utils.convert_batch_to_features(model.tokenizer, query)
    #     logits, pred = model(support, query, B, N, K, N*Q)

    #     # batch_samples, seg_ids, mask = utils.convert_batch_to_pair_features(
    #     #     model.tokenizer, support, query, B)
    #     # logits, pred = model(batch_samples, seg_ids, mask, B, N, K, N*Q)
    #     return logits, pred

    def train(self,
              model,
              model_name,
              B,
              N_for_train,
              N_for_eval,
              K,
              Q,
              ckpt_file='./checkpoint/fewrel/bert.pth',
              test_result_dir='./test_result',
              learning_rate=1e-1,
              lr_step_size=1000,
              weight_decay=FLAGS.l2_reg_lambda,
              train_iter=100000,
              val_iter=500,
              val_step=500,
              test_iter=3000,
              cuda=True,
              pretrain_model=None,
              optimizer=optim.Adam):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        test_result_dir: Directory of test results
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path
        '''
        print("Start training...")
        ckpt_file_path = ckpt_file
        parameters_to_optimize = model.parameters()
        optimizer = optimizer(parameters_to_optimize,
                              learning_rate, weight_decay=FLAGS.l2_reg_lambda)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=FLAGS.decay_rate)

        start_iter = 0
        # if pretrain_model:
        #     checkpoint = self.__load_model__(pretrain_model)
        #     model.load_state_dict(checkpoint['state_dict'])
        #     start_iter = checkpoint['iter'] + 1

        # if cuda:
        #     model = model.to(FLAGS.cuda)
        model.train()

        # Training
        best_acc = 0
        # Stop training after several epochs without improvement.
        not_best_count = 0
        iter_loss = 0.0
        iter_right = 0.0
        iter_sample = 0.0
        k_bak = K

        for it in range(start_iter, start_iter + train_iter):
            if it % 2 == 1:
                train_data_loader = self.train_data_loader_1shot
                K = 1
            else:
                train_data_loader = self.train_data_loader_5shot
                K = 5

            support, query, label = next(train_data_loader)
            sup_ids = support['word'].numpy()
            sup_mask = support['mask'].numpy()
            que_word = query['word'].numpy()
            que_mask = query['mask'].numpy()
            # logits, pred = self.predict(
            #     model, support, query, B, N_for_train, K, Q, label)
            label = label.to(FLAGS.paral_cuda[0])
            support = [support['word'].to(FLAGS.paral_cuda[0]), support['pos1'].to(FLAGS.paral_cuda[0]),
                       support['pos2'].to(FLAGS.paral_cuda[0]), support['mask'].to(FLAGS.paral_cuda[0])]
            query = [query['word'].to(FLAGS.paral_cuda[0]), query['pos1'].to(FLAGS.paral_cuda[0]),
                     query['pos2'].to(FLAGS.paral_cuda[0]), query['mask'].to(FLAGS.paral_cuda[0])]
            logits, pred = model(
                support, query, B, N_for_train, K, N_for_train*Q)
            loss = model.loss(logits, label)
            right = model.accuracy(pred, label)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(parameters_to_optimize, 5)
            optimizer.step()
            scheduler.step()
            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1
            sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(
                it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()

            if it % val_step == 0:
                iter_loss = 0.
                iter_right = 0.
                iter_sample = 0.

            # if (it+1) % 10 == 0:
            #     torch.cuda.empty_cache()
            if (it + 1) % val_step == 0:
                K = k_bak
                acc1 = self.eval(model, B, N_for_eval, K, Q,
                                 val_iter, data_loader=self.val_data_loader)
                # acc2 = self.eval(model, B, N_for_eval, K, Q,
                #                  val_iter, data_loader=self.test1_data_loader)
                acc2 = -1
                acc3 = self.eval(model, B, N_for_eval, K, Q,
                                 val_iter, data_loader=self.test2_data_loader)
                self.tabel.add_row(
                    [it, round(100*acc1, 4), round(100*acc3, 4)])
                print(self.tabel)
                model.train()
                if acc1+acc3 > best_acc:
                    print('Best checkpoint')
                    # if not os.path.exists(ckpt_dir):
                    #     os.makedirs(ckpt_dir)
                    # save_path = os.path.join(ckpt_dir, model_name + ".pth.tar")
                    torch.save({'state_dict': model.state_dict()},
                               ckpt_file_path)
                    best_acc = acc1+acc3
                print(
                    "#################best eval accu: %.4f##################" % (best_acc/2))
        print("\n####################\n")
        print("Finish training " + model_name)
        with torch.no_grad():
            test_acc = self.eval(model, B, N_for_eval, K, Q, test_iter)
        print("Test accuracy: {}, best accu:{}".format(test_acc, best_acc))

    def eval(self,
             model,
             B, N, K, Q,
             eval_iter,
             ckpt=None, data_loader=None, na_rate=FLAGS.na_rate):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        # print("")
        model.eval()
        if ckpt is None:
            eval_dataset = data_loader
        else:
            checkpoint = self.__load_model__(ckpt)
            model.load_state_dict(checkpoint['state_dict'])
            eval_dataset = self.test1_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        with torch.no_grad():
            for it in range(eval_iter):
                support, query, label = next(eval_dataset)
                # logits, pred = self.predict(
                #     model, support, query, B, N, K, Q, label)
                label = label.to(FLAGS.paral_cuda[0])
                support = [support['word'].to(FLAGS.paral_cuda[0]), support['pos1'].to(FLAGS.paral_cuda[0]),
                           support['pos2'].to(FLAGS.paral_cuda[0]), support['mask'].to(FLAGS.paral_cuda[0])]
                query = [query['word'].to(FLAGS.paral_cuda[0]), query['pos1'].to(FLAGS.paral_cuda[0]),
                         query['pos2'].to(FLAGS.paral_cuda[0]), query['mask'].to(FLAGS.paral_cuda[0])]
                logits, pred = model(support, query, B, N, K, N*Q)

                # logit = logits.detach().cpu().numpy()
                right = model.accuracy(pred, label)
                iter_right += self.item(right.data)
                iter_sample += 1

                sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(
                    it + 1, 100 * iter_right / iter_sample) + '\r')
                sys.stdout.flush()
            print("")
            accu = iter_right / iter_sample
            # print("####################accuracy: %.4f#####################" % accu)
        return accu
