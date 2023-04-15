# REPEAT 
This is an implemention for our ICSE 2023 paper based on pytorch

Keeping Pace with Ever-Increasing Data: Towards Continual Learning of Code Intelligence Models

by Shuzheng Gao, Hongyu Zhang, Cuiyun Gao and Chaozheng Wang


# Introduction 
REPEAT is a general training method for continual learning of code intelligence models.



# Data
Our processed datasets can be downloaded in [Zenodo](https://zenodo.org/record/7827136#.ZDjMEnZByUl).


# Run the code
Reproduce the results of our method and each baseline.


```markdown
|-- REPEAT
    |-- sum
    |   |-- CodeBERT
    |   |   |-- run_fineune.sh
    |   |   |-- run_emr.sh
    |   |   |-- run_ewc.sh
    |   |   |-- run_multitask.sh
    |   |   |-- run_ours.sh
    |   |   |-- ...
    |   |-- CodeT5
    |   |   |-- run_fineune.sh
    |   |   |-- run_emr.sh
    |   |   |-- run_ewc.sh
    |   |   |-- run_multitask.sh
    |   |   |-- run_ours.sh
    |   |   |-- ...
    |-- svd
    |   |-- CodeBERT
    |   |   |-- ...
    |   |-- CodeT5
    |       |-- ...
    |-- clone
    |   |-- CodeBERT
    |   |   |-- ...
    |   |-- CodeT5
    |       |-- ...
```

For example, if you want to reproduce the results of code summarization on CodeBERT, you can first move to the direcotory

```bash
cd sum/CodeBERT
```

Please first modify the data and model directory. You can also change the model's hyperparameter in each bash file. 


Normal Finetune:

```bash
bash run_finetune.sh
```


EMR method:

```bash
bash run_emr.sh
```


EWC method:

```bash
bash run_ewc.sh
```


Upper bound:

```bash
bash run_multitask.sh
```


Upper bound:

```bash
bash run_ous.sh
```


## Citation  

If you use our code, please kindly cite:

```
@inproceedings{Gao2023repeat,
  title={Keeping Pace with Ever-Increasing Data: Towards Continual Learning of Code Intelligence Models},
  author={Shuzheng Gao, Hongyu Zhang, Cuiyun Gao, and Chaozheng Wang},
  booktitle={ICSE},
  year={2023},
  publisher={IEEE}
}
```





















