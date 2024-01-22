conda create -y -n xequinet python=3.8 numpy scipy h5py
source activate xequinet
conda install pytorch==2.0.1 cudatoolkit==11.7 -c pytorch
conda install pyg -c pyg
pip install tqdm pyscf e3nn pytorch-warmup
pip install pydantic==1.10.8
conda deactivate 
