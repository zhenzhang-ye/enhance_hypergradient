# Enhance Hypergradient:
This code reproduces the results from the following [paper](https://arxiv.org/abs/2402.16748):

 > **Enhancing Hypergradients Estimation: A Study of Preconditioning and Reparameterization**

## 0. Requirement:
The code is tested in the following cuda environment:
1) cuda/11.2
2) cudnn/v8.8.1.3

The required python version and libraries can be found in conda_env.yml.

## 1. Run the code:
The default setting is reproducing Fig. 3b with a toy dataset:
```
python experiments
```
The following command can run the code with the dataset from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/), which might take a long time:
```
python experiments --demo
```
To reproduce other figures (e.g. Fig. 1a) in the paper, run the following command:
```
python experiments --demo --figure 1a
```
Reproducing 2a requires checking each result one by one. It is not implemented in this repository.
