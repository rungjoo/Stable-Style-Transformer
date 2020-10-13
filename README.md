# Stable Style Transformer with Classifier for Text Style Transfer (INLG 2020)
![model](./image/our_model.png)
The overall flow of our model

## Requirements
1. Pytorch 1.2+
2. Python 3.5+
3. [Huggingface Transformer](https://github.com/huggingface/transformers)
4. [BERTScore] (https://pypi.org/project/bert-score/)

## Datasets
1. [Yelp and Amazon Dataset](https://github.com/lijuncen/Sentiment-and-Style-Transfer)
2. [Human reference-DRG](https://github.com/lijuncen/Sentiment-and-Style-Transfer/tree/master/data)
3. [Human reference-DualRL](https://github.com/luofuli/DualRL/tree/master/references)

## Train
Description based on the yelp dataset
"""bash
cd generation_model/yelp
"""
### Step 1: train classifier
"""bash
cd classifier
python3 train.py
"""
### Step 2: train generator
"""bash
python3 train.py
"""
## Evaluation
"""bash
cd evaluation/yelp/my_model/SST/
"""
Check out generalization_eval_new.ipynb
Systems are evaluated using BLEU, classification accuracy, PPL, and BERTscore.
