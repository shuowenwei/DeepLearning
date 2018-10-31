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

# Install Anaconda, Tensorflow, Keras, and other libs
wget https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh
bash Anaconda3-5.2.0-Linux-x86_64.sh -b
echo 'export PATH=$PATH:$HOME/anaconda3/bin' >> ~/.bashrc
source ~/.bashrc
conda create -n pytorch python=3.6 -y
source activate pytorch
conda install pytorch torchvision cuda92 -c pytorch -y
conda install -c conda-forge spacy -y
python -m spacy download en
pip install --upgrade pip
pip install Cython pandas sklearn nltk gensim jupyterthemes matplotlib h5py tensorflow keras torchtext plotly
# Add "pytorch" environment to jupyter notebook
conda install ipykernel -y
python -m ipykernel install --user --name myenv --display-name "PyTorch"
# Back to home directory
cd ..