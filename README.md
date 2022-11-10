# Inductive Logical Query Answering in Knowledge Graphs (NeurIPS 2022) #

<p align="center">
<a href="https://arxiv.org/pdf/2210.08008.pdf"><img src="http://img.shields.io/badge/Paper-PDF-red.svg" alt="NeurIPS paper"></a>
<a href="https://doi.org/10.5281/zenodo.7306046"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7306046.svg" alt="InductiveQE dataset"></a>
</p>

![InductiveQE animation](asset/inductive_qe.gif)

This is the official code base of the paper

[Inductive Logical Query Answering in Knowledge Graphs][paper]

[Mikhail Galkin](https://migalkin.github.io),
[Zhaocheng Zhu](https://kiddozhu.github.io),
[Hongyu Ren](http://hyren.me/),
[Jian Tang](https://jian-tang.com)

[paper]: https://arxiv.org/abs/2210.08008

## Overview ##

**Important: the camera-ready NeurIPS'22 version was identified to have datasets with possible test set leakages.
The new version including this repository and updated Arxiv submission have new datasets and experiments where this
issue has been fixed. We recommend to use the latest version of datasets ([2.0 on Zenodo][dataset]) and experiments ([v2 on arXiv][paper]) for further
comparisons.**

[dataset]: https://zenodo.org/record/7306046

Inductive query answering is the setup where at inference time an underlying graph can have new, unseen entities.
In this paper, we study a practical inductive setup when a training graph **is extended** with more nodes and edges
at inference time. That is, an inference graph is always a superset of the training graph.
Note that the inference graph always shares the same set of relation types with the training graph.

The two big implications of the inductive setup:
* test queries involve new, unseen nodes where answers can be both seen and unseen nodes;
* training queries now might have more answers among new nodes.

The two inductive approaches implemented in this repo:
1. **NodePiece-QE** (Inductive node representations): based on [NodePiece](https://github.com/migalkin/NodePiece) and [CQD](https://github.com/pminervini/KGReasoning/).
Train on 1p link prediction, **inference-only** zero-shot logical query answering over unseen entities.
The NodePiece encoder can be extended with the additional GNN encoder (CompGCN) that is denoted as **NodePiece-QE w/ GNN** in the paper.
2. **Inductive GNN-QE** (Inductive relational structure representations): based on [GNN-QE](https://github.com/DeepGraphLearning/GNN-QE).
Trainable on complex queries, achieves higher performance than NodePiece-QE but is more expensive to train.

We additionally provide a dummy Edge-type Heuristic (`model.HeuristicBaseline`) that only considers possible tails of the last relation projection step of any query pattern.

## Data ##

We created 10 new inductive query answering datasets where validation/test graphs extend the training graph and contain new entities:
* Small-scale: 9 datasets based on FB15k-237 with the ratio of *inference-to-train nodes* varies from 106% to 550%, total of 15k nodes for various splits.
* Large-scale: 1 dataset based on OGB WikiKG2 with the fixed ratio of 133% and 1.5M training nodes but with 500K new nodes and 5M new edges at inference.

<details>
<summary>Datasets Description</summary>

Each dataset is a zip archive containing 17 files:

* `train_graph.txt` (pt for wikikg) - original training graph
* `val_inference.txt` (pt) - inference graph (validation split), new nodes in validation are disjoint with the test inference graph
* `val_predict.txt` (pt) - missing edges in the validation inference graph to be predicted.
* `test_inference.txt` (pt) - inference graph (test splits), new nodes in test are disjoint with the validation inference graph
* `test_predict.txt` (pt) - missing edges in the test inference graph to be predicted.
* `train/valid/test_queries.pkl` - queries of the respective split, 14 query types for fb-derived datasets, 9 types for WikiKG (EPFO-only)
* `*_answers_easy.pkl` - easy answers to respective queries that do not require predicting missing links but only edge traversal
* `*_answers_hard.pkl` - hard answers to respective queries that DO require predicting missing links and against which the final metrics will be computed
* `train_answers_valid.pkl` - the extended set of answers for training queries on the bigger validation graph, most of training queries have at least 1 more new answers. This is supposed to be an inference-only dataset to measure faithfulness of trained models
* `train_answers_test.pkl` - the extended set of answers for training queries on the bigger test graph, most of training queries have at least 1 more new answers. This is supposed to be an inference-only dataset to measure faithfulness of trained models
* `og_mappings.pkl` - contains entity2id / relation2id dictionaries mapping local node/relation IDs from a respective dataset to the original fb15k237 / wikikg2
* `stats.txt` - a small file with dataset stats
</details>

All datasets are available on [Zenodo](https://zenodo.org/record/7306046), please refer to v2.0 of the datasets.
The datasets will be downloaded automatically upon the first run.

Additionally, we provide lightweight dumps ([Zenodo](https://zenodo.org/record/7306061)) just of those graphs (without queries and answers) for training simple link prediction and KG completion models.
Please refer to v2.0 of the datasets.

## Installation ##

The dependencies can be installed via either conda or pip. NodePiece-QE and GNN-QE are compatible
with Python 3.7/3.8/3.9 and PyTorch >= 1.8.0.

### From Conda ###

```bash
conda install torchdrug pytorch cudatoolkit -c milagraph -c pytorch -c pyg
conda install pytorch-sparse pytorch-scatter -c pyg
conda install easydict pyyaml -c conda-forge
```

### From Pip ###

```bash
pip install torchdrug torch
pip install easydict pyyaml
pip install wandb tensorboardx
```

Then install `torch-scatter` and `torch-sparse` following the instructions in the [Github repo](https://github.com/rusty1s/pytorch_sparse).
For example, for PyTorch 1.10 and CUDA 10.2:

```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
```


## Usage ##

### NodePiece-QE ###

Conceptually, running NodePiece-QE consists of two parts:
1. Training a neural link predictor using NodePiece (+ optional GNN), saving materialized embeddings of the test graph.
2. Running CQD over the saved embeddings.

**Step 1: Training a Link Predictor**

Use the `NodePiece` model with the task `InductiveKnowledgeGraphCompletion` applied to the dataset of choice.

We prepared 5 configs for FB15k-237-derived datasets in the `config/lp_pretraining` directory, 2 for NodePiece w/o GNN
and 3 for NodePiece w/ GNN, following the reported hyperparameters in the paper.
`_550` configs have a higher `input_dim` so we decided to have a dedicated file for them to send less params to the
training script.

We also provide 2 configs for the WikiKG graph and recommend running pre-training in the multi-gpu mode due to the size
of the graph.

Example of training a vanilla NodePiece on the 175% dataset:

```bash
python script/run.py -c config/lp_pretraining/nodepiece_nognn.yaml --ratio 175 --temp 0.5 --epochs 2000 --gpus [0] --logger console
```

NodePiece + GNN on the 175% dataset:
```bash
python script/run.py -c config/lp_pretraining/nodepiece_gnn.yaml --ratio 175 --temp 1.0 --epochs 1000 --gpus [0] --logger console
```

For datasets of ratios 106-150 use the 5-layer GNN config `config/lp_pretraining/nodepiece_gnn.yaml`.

* Use `--gpus null` to run the scripts on a CPU.
* Use `--logger wandb` to send training logs to wandb, don't forget to prepend env variable `WANDB_ENTITY=(your_entity)`
before executing the python script.

After training, materialized entity and relation embeddings of the test graph will be stored in the `output_dir` folder.

WikiKG training requires a vocabulary of mined NodePiece anchors, we ship a precomputed
vocab `20000_anchors_d0.4_p0.4_r0.2_25sp_bfs.pkl` together with the `wikikg.zip` archive.
You can mine your own vocab playing around with the `NodePieceTokenizer` -- mining is implemented on a GPU and
should be much faster than the original NodePiece implementation.

An example WikiKG link prediction pre-training config should contain `--vocab` param to the mined vocab, e.g.,
```bash
python script/run.py -c config/lp_pretraining/wikikg_nodepiece_nognn.yaml --gpus [0] --vocab /path/to/pickle/vocab.pkl
```

We highly recommend training both no-GNN and GNN versions of NodePiece on WikiKG using several GPUs, for example
```bash
python -m torch.distributed.launch --nproc_per_node=2 script/run.py -c config/lp_pretraining/wikikg_nodepiece_nognn.yaml --gpus [0,1] --vocab /path/to/pickle/vocab.pkl
```

**Step 2: CQD Inference**

Use the pre-trained link predictor to run CQD inference over EPFO queries (negation is not supported in this version of CQD).

Example of running CQD on the pre-trained 200d NodePiece w/ GNN model over the 175% dataset
* Note that we need to specify a 2x smaller embedding dimension of the training model as by default we train a
ComplEx model with two parts - real and complex;
* Use the full path to the **embeddings** of the pre-trained models, they are named smth like `/path/epoch_1000_ents`
and `/path/epoch_1000_rels`, so just use the common prefix `/path/epoch_1000`.

```bash
python cqd/main.py --cuda --do_test --data_path ./data/175 -d 100 -cpu 6 --log_steps 10000 --test_log_steps 10000 --geo cqd --print_on_screen --cqd-k 32 --cqd-sigmoid --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --inductive --checkpoint_path /path/epoch_1000 --skip_tr
```

To evaluate training queries on the bigger test graphs, use the argument `--eval_train`

```bash
python cqd/main.py --cuda --do_test --data_path ./data/175 -d 100 -cpu 6 --log_steps 10000 --test_log_steps 10000 --geo cqd --print_on_screen --cqd-k 32 --cqd-sigmoid --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --inductive --checkpoint_path /path/epoch_1000 --eval_train
```

### GNN-QE ###

To train GNN-QE and evaluate on the valid/test queries and desired dataset ratio, use the `gnnqe_main.yaml` config.
Example on the 175% dataset:

```bash
python script/run.py -c config/complex_query/gnnqe_main.yaml --ratio 175 --gpus [0]
```

Alternatively, you may specify `--gpus null` to run GNN-QE on a CPU.

The hyperparameters are designed for 32GB GPUs, but you may adjust the batch size in the config file
to fit a smaller GPU memory.

To run GNN-QE with multiple GPUs or multiple machines, use the following commands

```bash
python -m torch.distributed.launch --nproc_per_node=2 script/run.py -c config/complex_query/gnnqe_main.yaml --gpus [0,1]
```

```bash
python -m torch.distributed.launch --nnodes=4 --nproc_per_node=4 script/run.py -c config/complex_query/gnnqe_main.yaml --gpus [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
```

To evaluate training queries on the bigger test graphs, use the config `gnnqe_eval_train.yaml` and specify the
checkpoint `--checkpoint` of the trained model. The best performing checkpoint is written in the log files after training the `main` config.
For example, if the best performing 175% model is `model_epoch_1.pth` then the path will be:
```bash
python script/run.py -c config/complex_query/gnnqe_eval_train.yaml --ratio 175 --gpus [0] --checkpoint /path/to/model/model_epoch_1.pth
```

### Heuristic Baseline ###

Finally, we provide configs for the inference-only rule-based heuristic baseline
that only considers possible tails of the last relation projection step of any query pattern.
The two configs are `config/complex_query/heuristic_main.yaml` and `config/complex_query/heuristic_eval_train.yaml`.

To run the baseline on test queries (for example, on the 175% dataset):

```bash
python script/run.py -c config/complex_query/heuristic_main.yaml --ratio 175 --gpus [0]
```

To run the baseline on train queries over bigger test graphs:
```bash
python script/run.py -c config/complex_query/heuristic_eval_train.yaml --ratio 175 --gpus [0]
```

## Citation ##

If you find this project useful in your research, please cite the following paper

```bibtex
@inproceedings{galkin2022inductive,
  title={Inductive Logical Query Answering in Knowledge Graphs},
  author={Mikhail Galkin and Zhaocheng Zhu and Hongyu Ren and Jian Tang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
  url={https://openreview.net/forum?id=-vXEN5rIABY}
}
```