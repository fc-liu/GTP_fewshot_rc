val_file="val_pubmed.json"
for N in 5 10;do
    for K in 1 5;do
        echo ${N}-way ${K}-shot:
        for seed in 1 12 123 1234 12345; do
            python sample_io.py data/${val_file} 100 $N $K $seed input > data/sample.json
            python sample_io.py data/${val_file} 100 $N $K $seed output > data/ans.json
            # python test.py --N $N --K $K --bert_model /share/model/bert/cased_L-24_H-1024_A-16 --ckpt_name gtp-woseg-tagproto-only-layer1-head4 --layer 1 --n_head 4 --model_name tag_cos > data/res.json
            python test.py --N $N --K $K --bert_model /share/model/bert/cased_L-24_H-1024_A-16 --ckpt_name discrim_k15_h1 --layer 1 --n_head 1 --model_name discrim > data/res.json
            python evaluate.py data/res.json data/ans.json
        done
    done
done