#!/bin/bash
conda create --name ADRGD python=3.7.15 -y
conda activate ADRGD
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install scipy==1.7.3 torchattacks==3.3.0 numpy==1.21.6