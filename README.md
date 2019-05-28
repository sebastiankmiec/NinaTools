# *ninaeval* Python Package

NinaTools, and the PythonMyoLinux repository, constitute the work of my final year capstone project (poster below).
![image](https://drive.google.com/uc?export=view&id=1tMUZbA5dVqKowB9Ruh7uCwTAUb2KQM0H)



**The *ninaeval* package can:**

1. Download, process and format the NinaPro dataset (as well as newly created data).
2. Extract features to create training/validation/testing sets.
3. Compare the performance of various classifier models.

**NinaPro dataset source (DB5):**
```
@article{
    author = {Pizzolato, Stefano and Tagliapietra, Luca and Cognolato, Matteo and Reggiani, Monica and M{\"{u}}ller, Henning and Atzori, Manfredo},
     title = {Comparison of Six Electromyography Acquisition Setups on Hand Movement Classification Tasks},
   journal = {Plos One},
      year = {2017}
}
```
&nbsp;


# Setup

1. Create conda environment, install basic dependencies
```
conda create -n myo_env -y
conda activate myo_env

conda install -c conda-forge python=3.5 pytorch pywavelets scipy pandas -y
pip install torchnet tqdm matplotlib scikit-learn appdirs
```

2. Install the *ninaeval* package
```
git clone https://github.com/sebastiankmiec/NinaTools.git
cd NinaTools/
pip install .
```

3. Setup kymatio
```
cd ..
git clone https://github.com/kymatio/kymatio
cd kymatio
python setup.py install
```
&nbsp;


# Usage
Use one of the following, depending on your use case
```
python ninapro_example.py --json=test.json
python new_data_example.py --json=test.json
```
For the first example, you should get the following output
```
1/10. Downloading "https://zenodo.org/record/1000116/files/s2.zip?download=1".
2/10. Downloading "https://zenodo.org/record/1000116/files/s8.zip?download=1".
3/10. Downloading "https://zenodo.org/record/1000116/files/s10.zip?download=1".
4/10. Downloading "https://zenodo.org/record/1000116/files/s6.zip?download=1".
5/10. Downloading "https://zenodo.org/record/1000116/files/s9.zip?download=1".
6/10. Downloading "https://zenodo.org/record/1000116/files/s3.zip?download=1".
7/10. Downloading "https://zenodo.org/record/1000116/files/s1.zip?download=1".
8/10. Downloading "https://zenodo.org/record/1000116/files/s5.zip?download=1".
9/10. Downloading "https://zenodo.org/record/1000116/files/s4.zip?download=1".
10/10. Downloading "https://zenodo.org/record/1000116/files/s7.zip?download=1".
Loading Ninapro data from processed directory...
Extracting dataset features for training, and testing...
100%|████████████████████████████████████████████████████████████████████| 10/10 [00:10<00:00,  1.07s/it]
100%|████████████████████████████████████████████████████████████████████| 10/10 [00:11<00:00,  1.12s/it]
Training classifier on training dataset...
Testing classifier on testing dataset...
0.8943925233644859
```
