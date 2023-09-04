conda create -n FA python=3.8 -y
conda activate FA
conda install -c conda-forge ninja -y
conda install -c conda-forge jupyterlab -y 
conda install -c conda-forge pandas -y 
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
git clone https://github.com/Dao-AILab/flash-attention.git
# pip install flash-attn --no-build-isolation  
# FA2 install
cd ./flash-attention
git checkout tags/v1.0.9
python setup.py install
jupyter notebook --allow-root