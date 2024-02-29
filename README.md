# Rethinking Label Smoothing on Multi-hop Question

![License](https://img.shields.io/badge/License-Apache%20License%202.0-green)
![python](https://img.shields.io/badge/python-3.8+-blue.svg)

CCL 2023 Best Paper: [Rethinking Label Smoothing on Multi-hop Question](https://link.springer.com/chapter/10.1007/978-981-99-6207-5_5)


## Introduction 📝

Welcome to the official repository for the paper: "[Rethinking Label Smoothing on Multi-hop Question](https://arxiv.org/pdf/2212.09512.pdf)". Our work introduces a cutting-edge Multi-Hop Question Answering (MHQA) framework, Smoothing $\mathbf{R}^3$, which integrates a series of label smoothing techniques aimed at enhancing the multi-hop reasoning process. We propose **F1 Smoothing**, an innovative approach inspired by the widely-used F1 score in Machine Reading Comprehension (MRC) tasks. Additionally, we present **LDLA ( Linear Decay Label Smoothing Algorithm)**, a pioneering label smoothing algorithm that incorporates curriculum learning principles. This repository houses both the implementation code and the datasets utilized in our research.
![Cover](figures/cover.png)


## Quick Links 🔗

- [Rethinking Label Smoothing on Multi-hop Question](#rethinking-label-smoothing-on-multi-hop-question)
  - [Introduction 📝](#introduction-)
  - [Quick Links 🔗](#quick-links-)
  - [Requirements 📚](#requirements-)
  - [Data 💾](#data-)
  - [Reproducing Baselines 🚀](#reproducing-baselines-)
    - [Training the Retriever Model](#training-the-retriever-model)
    - [Training the Reader Model](#training-the-reader-model)
    - [Script Parameters Explained](#script-parameters-explained)
    - [Hardware Recommendations and Data Preprocessing](#hardware-recommendations-and-data-preprocessing)
  - [Evaluation 💻](#evaluation-)
  - [Bug or Questions? 🤔](#bug-or-questions-)
  - [Citation 📖](#citation-)


## Requirements 📚

Please make sure you have the following requirements installed:
- transformers>=4.20.0
- fastNLP==1.0.1
- jsonlines
- ipdb
- pandas
- torch
- ujson


## Data 💾

The dataset required for this project is available through the official [HotpotQA website](https://hotpotqa.github.io/). To facilitate seamless integration with the code, please follow these steps:

1. Visit the [HotpotQA dataset page](https://hotpotqa.github.io/) and download the dataset.
2. Create a folder named `HotpotQAData` in the root directory of this project, at the same level as the `code` folder.
3. Save the downloaded dataset files into the `HotpotQAData` folder.


## Reproducing Baselines 🚀

We've streamlined the process for replicating our baseline models by providing starter scripts in [main.py](code/main.py). Below you'll find detailed instructions for training both the Retriever and Reader models, along with a comprehensive explanation of the script parameters to customize your training runs.

### Training the Retriever Model

Initiate training of the Retriever model using the command below:

```bash
python train.py --task RE --lr 5e-6 --batch-size 16 --accumulation-steps 1 --epoch 8 --seed 41 --re-model Electra
```

### Training the Reader Model

To start training the Reader model, use the following command:

```bash
python train.py --task QA --lr 2e-6 --batch-size 8 --accumulation-steps 2 --epoch 8 --seed 41 --qa-model Deberta
```

### Script Parameters Explained

When running `train.py`, you can customize your training with the following input arguments:

- `task`: Specifies the model to train - Retriever (`RE`) or Reader (`QA`).
- `lr`: Sets the learning rate.
- `batch-size`: Defines the batch size per step.
- `accumulation-steps`: Determines how many steps to accumulate gradients for before updating model weights (Effective batch size = `batch-size` * `accumulation-steps`).
- `epoch`: Number of training epochs.
- `seed`: Random seed for reproducibility.
- `re-model`/`qa-model`: Chooses the backbone model for the Retriever (`Electra`/`Roberta`) or Reader (`Deberta`/`Roberta`) task.
- `LDLA-decay-rate`: Specifies the decay rate for the LDLA algorithm.

Additional parameters like `data-path` for specifying the dataset directory, and `warmupsteps` for setting the number of warmup steps during training, are also supported but not detailed in the initial commands.

### Hardware Recommendations and Data Preprocessing

- **GPU**: Training is optimized for A100 GPUs. Adjust the random seed (`--seed`) based on your system's clock to ensure reproducibility.
- **Model Support**: We support RoBERTa models as an alternative backbone by specifying `--re-model` or `--qa-model` accordingly.
- **Data Preprocessing**: To prevent redundant preprocessing, processed data is cached in the `cache` directory as `.pkl` files. If you alter [preprocess.py](code/preprocess.py), clear the cache to apply new preprocessing changes.
- **Evaluation**: In addition to the metrics outlined in the Evaluation section, we include `cl_acc` to assess the accuracy of answer type classification.

By default, the script is set to save the top 3 Retriever (`RE`) checkpoints based on F1 scores and the top 3 Reader (`QA`) checkpoints based on joint F1 scores. Ensure your training and evaluation aligns with these best practices for optimal results.


## Evaluation 💻

Once training is complete, assess the performance of your model using the official HotpotQA evaluation script. The script, `hotpot_official_evaluate.py`, is available in the `code` directory and allows for a comprehensive analysis of your model's predictions.

Execute the following command to evaluate your model's predictions:

```bash
python code/hotpot_official_evaluate.py --prediction-file model_pred.json --gold-file HotpotQAData/hotpot_dev_distractor_v1.json
```

Replace `model_pred.json` with the path to your model's prediction file. Ensure that the `hotpot_dev_distractor_v1.json` file is located in the `HotpotQAData` folder, as per the setup instructions in the Data section.

The script outputs several key metrics to gauge the performance of your model:

- **sp_em, sp_f1, sp_prec, and sp_recall**: These metrics evaluate the correctness of supporting facts' judgments. They measure the exact match (em), precision (prec), recall, and F1 score (f1) specifically for supporting facts identification.
- **em, f1, prec, and recall**: These metrics assess the accuracy of answer span extraction, evaluating how well the model identifies the exact answers within the text.
- **joint_em, joint_f1, joint_prec, and joint_recall**: These combined metrics provide an overall assessment of your model's performance, taking into account both the accuracy of supporting fact judgments and answer span extraction.


## Bug or Questions? 🤔

We welcome your feedback and questions! If you have any suggestions, or need to get in touch, please don't hesitate to email us at yinzhangyue@126.com. For issues related to the code or any bugs you might encounter, we encourage you to open a new issue on GitHub. As this is an initial release, your constructive feedback is invaluable to us in making improvements. Thank you for your support and involvement!


## Citation 📖

If you are interested in our work, please use the following citation format when referencing our paper:
```bibtex
@InProceedings{yin-etal-2023-rethinking,
author="Yin, Zhangyue
and Wang, Yuxin
and Hu, Xiannian
and Wu, Yiguang
and Yan, Hang
and Zhang, Xinyu
and Cao, Zhao
and Huang, Xuanjing
and Qiu, Xipeng",
editor="Sun, Maosong
and Qin, Bing
and Qiu, Xipeng
and Jing, Jiang
and Han, Xianpei
and Rao, Gaoqi
and Chen, Yubo",
title="Rethinking Label Smoothing on Multi-Hop Question Answering",
booktitle="Chinese Computational Linguistics",
year="2023",
publisher="Springer Nature Singapore",
address="Singapore",
pages="72--87",
isbn="978-981-99-6207-5"
}
```