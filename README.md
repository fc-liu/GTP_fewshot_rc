# Interactive Prototype for Few-Shot Relation Classification

## Data preparation
we use the official FewRel dataset for training and evaluation,
please download from https://github.com/thunlp/FewRel and put data files into ./data directory.

## Usage
### prerequirement
```bash
pip install -r requirements.txt
```
### Train
```bash
python train_demo.py --mode train
```
### Test
```bash
bash test.sh # replace the file path in this file
```

