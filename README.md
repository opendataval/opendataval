# Data-OOB: Out-of-bag Estimate as a Simple and Efficient Data Value

We provide an implementation of the paper "Data-OOB: Out-of-bag Estimate as a Simple and Efficient Data Value" as a part of Supplementary Material for anonymous review at ICML 2023. Please do not distribute files. 

## Quick start

The following sample Python code will conduct comparison experiments using the synthetic binary classification dataset introduced in Section 4.1. As for the real data analysis, please download OpenML datasets first and change the filepath information in the `config.py` file. To replicate the numerical experiments in the manuscript, you may want to take a close look into the `config.py` file.

```
python3 run --exp_id='expno000CR' --run-id=0 --runpath='./'
```

## The core python file 

The Python class `RandomForestClassifierDV` in `ensemble_DV_core.py` helps to compute the proposed method. 



