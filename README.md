# title_generation
Uses attention networks to generate titles of technical papers using abstracts

## Data Sources
* [ARXIV data](https://www.kaggle.com/neelshah18/arxivdataset#arxivData.json)
* [NIPS data](https://www.kaggle.com/benhamner/nips-papers)

## Structure
* **backend**: contains scripts for training and evaluation of models
* **data**: contains data files from training data, models to results

## Setting up develope env and running server
* `python3 -m virtualenv env`
* `source "virtual env path activate"`
* `pip install -r requirements.txt`
* `python manage.py runserver`
* Can go to localhost and access web UI for evaluating latest saved model

## How to train
* `source "virtual env path activate"`
* `cd title_generation/deeptitles`
* `python -m main.backend.train > main/data/results/"dd_mm_HH.log"`
* one can always go to the utility.py file to tune the hyperparams

## How to train in MSI
* *to do* 

---

# Update with each commit

## References
* [Code: PyTorch Attention models](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

## Current model details
* GRU based encoder and attention decoder
* Hidden size=512, dropout=0.1, learning_rate=0.01 (SGD)
* Uses *teacher-forcing* mechanism during training for faster convergence

## Tasks to do
* setup an evaluation metric for prediction
* identify benchmark
* try other architecturs to make prediction better
* try hyper parameter tuning to make prediction better

## Experimentation (for logging progress)
| **Commit**        | **Description**  |
| ------------- |:-------------:|
| 811b1d8      | NIPS Dataset (Simple Encoder + Attn_Decoder) |
