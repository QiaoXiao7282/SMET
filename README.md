# 


## Install

Tested on Linux. 

```bash
conda create --prefix /projects/0/einf3822/.conda/dst_llm_py10 python=3.10
pip install transformers==4.38.0
pip install datasets==2.19.1
pip install huggingface-hub==0.27.0
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install wandb
pip install loguru
pip install numpy==1.26.4
```


## Pre-training

Run the following script:
```bash
bash scripts/train_llm.sh
```


## Acknowledgement
This repository is built upon the [MixLN](https://github.com/pixeli99/MixLN/tree/main) repo, 
which is based on [GaLore](https://github.com/jiaweizzhao/GaLore). Thanks for their great work!
