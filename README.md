# BayesPrompt
Code for the paper "BayesPrompt: Prompting Large-Scale Pre-Trained Language Models on Few-shot Inference via Debiased Domain Abstraction". 
### Requirements

- Python 3.8
- numpy == 1.20.3
- pandas == 1.3.4
- pytorch_lightning == 1.3.1
- PyYAML == 5.4.1
- scikit_learn == 0.24.2
- torch == 1.10.1+cu111
- transformers == 4.7.0
- torchmetrics == 0.5.0
- wandb == 0.13.11

### Getting Started

For a quick start, we perform the GMM and SVGD operations in advance and store the results in the “updated_datasetname” folder. If you want to do this from scratch, please use "transformer_full.py".

##### An Example of SemEval

```bash
bash scripts/semeval.sh
```
We also provide other related data files for download on [Google Drive](https://drive.google.com/file/d/1F3upnwi84msO7mMd0xfARBvbXX-1Mr-h/view?usp=sharing).

### Acknowledgements

The code is based on [KnowPrompt](https://github.com/zjunlp/KnowPrompt) and [SVGD](https://github.com/dilinwang820/Stein-Variational-Gradient-Descent), thank you very much.


### Citation
If you find this repo useful for your research, please consider citing the following paper: 
```bash
@misc{li2024bayesprompt,
      title={BayesPrompt: Prompting Large-Scale Pre-Trained Language Models on Few-shot Inference via Debiased Domain Abstraction}, 
      author={Jiangmeng Li and Fei Song and Yifan Jin and Wenwen Qiang and Changwen Zheng and Fuchun Sun and Hui Xiong},
      year={2024},
      eprint={2401.14166},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
