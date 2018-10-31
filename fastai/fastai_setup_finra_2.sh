#!/usr/bin/env bash

# from: http://files.fast.ai/setup/paperspace 

# as of 2018.08.16
# NVidia driver: NVIDIA-Linux-x86_64-396.44.run 
# CUDA driver: cuda_9.2.148_396.37_linux.run 
# CUDA Patches: cuda_9.2.148.1_linux.run 
# cuDNN zipfile: cudnn-9.2-linux-x64-v7.1.tgz 
# Anaconda: Anaconda3-5.2.0-Linux-x86_64.sh 

# Make sure all above files are copied to this folder 
cd ~/nvidia-install 

sudo yum install -y gcc kernel-devel-$(uname -r)
# Install git just in case
sudo yum install git -y

# Install NVidia driver
sudo /bin/bash NVIDIA-Linux-x86_64-396.44.run -s

# Optimize NVidia driver for G3
# https://docs.aws.amazon.com/AWSEC2/latest/WindowsGuide/optimize_gpu.html 
nvidia-smi -q | head
sudo nvidia-persistenced
sudo nvidia-smi -ac 2505,1177

# Install CUDA
sudo sh cuda_9.2.148_396.37_linux.run --silent --toolkit --override
#ls /usr/local/cuda-9.2/
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64
export PATH=$PATH:/usr/local/cuda-9.2/bin
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64' >> ~/.bashrc
echo 'export PATH=$PATH:/usr/local/cuda-9.2/bin' >> ~/.bashrc
# Install CUDA patches
sudo sh cuda_9.2.148.1_linux.run --silent --accept-eula

# Install cuDNN
tar -xzvf cudnn-9.2-linux-x64-v7.2.1.38.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-9.2/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-9.2/lib64/
sudo chmod a+r /usr/local/cuda-9.2/lib64/libcudnn*

sudo yum install unzip -y

# set -e
# set -o xtrace

# DEBIAN_FRONTEND=noninteractive

# sudo rm /etc/apt/apt.conf.d/*.*
# sudo apt update
# sudo apt install unzip -y
# sudo apt -y upgrade --force-yes -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"
# sudo apt -y autoremove
# sudo ufw allow 8888:8898/tcp
# sudo apt -y install --force-yes -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" qtdeclarative5-dev qml-module-qtquick-controls
# sudo add-apt-repository ppa:graphics-drivers/ppa -y
# sudo apt update
# mkdir downloads
# cd ~/downloads/
# wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
# sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
# sudo apt update
# sudo apt install cuda -y
# wget http://files.fast.ai/files/cudnn-9.1-linux-x64-v7.tgz
# tar xf cudnn-9.1-linux-x64-v7.tgz
# sudo cp cuda/include/*.* /usr/local/cuda/include/
# sudo cp cuda/lib64/*.* /usr/local/cuda/lib64/
# wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
# bash Anaconda3-5.0.1-Linux-x86_64.sh -b
# cd
git clone https://github.com/fastai/fastai.git
cd fastai/
echo 'export PATH=~/anaconda3/bin:$PATH' >> ~/.bashrc
export PATH=~/anaconda3/bin:$PATH
source ~/.bashrc

conda create -n fastai python=3.6 -y
source activate fastai

# echo 'source activate fastai' >> ~/.bashrc
# source ~/.bashrc

conda install pytorch torchvision cuda92 -c pytorch -y
# conda install -c conda-forge spacy -y
# python -m spacy download en
pip install --upgrade pip
pip install Cython pandas sklearn nltk gensim jupyterthemes matplotlib h5py tensorflow keras torchtext plotly
pip install opencv-python seaborn graphviz sklearn-pandas isoweek pandas_summary ipywidgets
jupyter nbextension enable --py widgetsnbextension
conda install -c conda-forge bcolz -y 
python -m ipykernel install --user --name myenv --display-name "fastai"


cd ..
# mkdir data
cd data
# wget http://files.fast.ai/data/dogscats.zip
unzip -q dogscats.zip
cd ../fastai/courses/dl1/
ln -s ~/data ./

jupyter notebook --generate-config
echo "c.NotebookApp.ip = '*'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension --sys-prefix
echo
echo ---
echo - YOU NEED TO REBOOT YOUR PAPERSPACE COMPUTER NOW
echo ---
