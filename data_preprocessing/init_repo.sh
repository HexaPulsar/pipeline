


conda create -n dataset python=3.10 --file requirements.txt
conda activate dataset

#pipeline repo (alerce public repo)
git clone git@github.com:alercebroker/pipeline.git

#batch processing repo (alerce private repo)
git clone git@github.com:alercebroker/batch_processing.git
 

#- compile Mexican hat and P4J

cd /mhps
#-march=x86-64
python setup.py build_ext --inplace
pip install .

cd ../P4J
python setup.py build_ext --inplace
pip install .

