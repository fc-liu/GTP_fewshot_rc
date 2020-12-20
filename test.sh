val_file="test.json"
for N in 5 10;do
    for K in 1 5;do
        echo ${N}-way ${K}-shot:
            python sample_io.py data/${val_file} 10000 $N $K 12345 input > data/sample.json
            python sample_io.py data/${val_file} 10000 $N $K 12345 output > data/ans.json
            # CUDA_VISIBLE_DEVICES=5 python test.py --N $N --K $K --bert_model /share/model/bert/cased_L-24_H-1024_A-16 --ckpt_name gtp-woseg-sepexcludproto-lay1-head1 --layer 1 --n_head 1 --model_name gtp --paral_cuda 0 > data/res.json
            CUDA_VISIBLE_DEVICES=5 python test.py --N $N --K $K --bert_model /share/model/bert/cased_L-24_H-1024_A-16 --ckpt_name three-less-l2-h4 --layer 2 --n_head 4 --model_name proto_three --paral_cuda 0 > data/res.json
            python evaluate.py data/res.json data/ans.json
        done
    done
done