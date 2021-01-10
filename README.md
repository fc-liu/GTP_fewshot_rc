# Interactive Prototype for Few-Shot Relation Classification

the data is on FewRel official page https://github.com/thunlp/FewRel,
please download and put into ./data directory.

## Usage
### prerequirement
```bash
pip install -r requirements.txt
```
### Train
```bash
python train_demo.py --mode train --layer 1
```
### Test
```bash
bash test-${N}-${K}.sh # test on FewRel 2.0 N-way K-shot task
```
or
```bash
bash test-${N}-${K}-wiki.sh # test on FewRel 1.0 N-way K-shot task
```
