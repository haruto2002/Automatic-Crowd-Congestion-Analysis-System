conda install -y pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y numpy=1.23.5 -c intel
conda install -y lightning=2.2.0 -c conda-forge
conda install -y pandas==2.2.1 -c conda-forge
conda install -y matplotlib=3.8.3 -c conda-forge
conda install -y albumentations=1.3.1 -c conda-forge
conda install -y optuna=3.5.0 -c conda-forge
conda install -y botorch=0.10.0 -c conda-forge 
conda install -y cmaes=0.10.0 -c conda-forge
conda install -y pyDOE2=1.3.0 -c conda-forge
conda install -y plotly=5.19.0 -c conda-forge
conda install -y nvitop=1.3.2 -c conda-forge
conda install -y seaborn=0.13.2 -c conda-forge
conda install -y omegaconf=2.3.0 -c conda-forge
conda install -y tensorboard=2.16.2 -c conda-forge
conda install -y black=24.2.0 -c conda-forge
conda install -y h5py=3..0 -c conda-forge
conda install -y click=8.1.3 -c conda-forge
conda install -y moviepy=1.0.3 -c conda-forge
conda install -y timm=0.9.11 -c conda-forge
conda install -y pycocotools=2.0.6 -c conda-forge
pip install kaleido==0.2.1
pip install mean_average_precision
pip install wandb

conda install conda-forge::pytorch-model-summary

conda install conda-forge::lap
pip install cython-bbox
pip install hydra-core