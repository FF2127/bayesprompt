# BayesPrompt
Code for the paper "BayesPrompt: Prompting Large-Scale Pre-Trained Language Models on Few-shot Inference via Debiased Domain Abstraction". 
### Installation
##### Create a virtual environment

```bash
conda create -n bayesprompt python=3.8
conda activate bayesprompt
```
##### Requirements

- numpy == 1.20.3
- pandas == 1.3.4
- pytorch_lightning == 1.3.1
- PyYAML == 5.4.1
- scikit_learn == 0.24.2
- torch == 1.10.1+cu111
- transformers == 4.7.0

### Getting Started
##### An Example for SemEval

```bash
bash scripts/semeval.sh
```

### Acknowledgements

The code is based on [KnowPrompt](https://github.com/zjunlp/KnowPrompt) and [SVGD](https://github.com/dilinwang820/Stein-Variational-Gradient-Descent), thank you very much.


### Citation
If you find this repo useful for your research, please cite the following paper: 
```bash
BayesPrompt: Prompting Large-Scale Pre-Trained Language Models on Few-shot Inference via Debiased Domain Abstraction
```
