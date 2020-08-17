
val_file="val_pubmed.json"
for N in 5 10;do
    for K in 1 5;do
        python sample_io.py data/${val_file} 100 $N $K 12345 input > data/sample.json
        python sample_io.py data/${val_file} 100 $N $K 12345 output > data/ans.json
        python test.py --N $N --K $K --bert_model /share/model/bert/cased_L-24_H-1024_A-16 --paral_cuda 0 --ckpt_name gtp-10-mix15-add12> data/res.json
        echo ${N} way-${K} shot:
        python evaluate.py data/res.json data/ans.json
    done
done