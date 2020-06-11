<p align=""left>
<img src="https://img.shields.io/badge/release--date-06%2F2020-green.svg">
</p>

# SC-GAN
The project repository for **SC-GAN: 3D self-attention conditional GAN with spectral normalization for multi-modal neuroimaging synthesis**

## Data
The *data* directory should be organized as the following structure:

```
data
│
└───train
|   |
│   └─── subject 1
|   |         modality_1.nii.gz
|   |         modality_2.nii.gz
|   |         ...
|   |         target.nii.gz
│   └─── subject 2
|   |         modality_1.nii.gz
|   |         modality_2.nii.gz
|   |         ...
|   |         target.nii.gz
│   ...
|
└───val
|   |
│   └─── subject A
|   |         modality_1.nii.gz
|   |         modality_2.nii.gz
|   |         ...
|   |         target.nii.gz
│   └─── subject B
|   |         modality_1.nii.gz
|   |         modality_2.nii.gz
|   |         ...
|   |         target.nii.gz
│   ...
|   
└───test
    |
    └─── subject a
    |         modality_1.nii.gz
    |         modality_2.nii.gz
    |         ...
    └─── subject b
    |         modality_1.nii.gz
    |         modality_2.nii.gz
    |         ...
    ...
```
To evaluate SC-GAN, we used ADNI data. To access ADNI data, please visit: https://ida.loni.usc.edu/

## Requirements

python 3 is required and `python 3.6.9` was used in the study.

Requirements can be found at [requirement.txt](https://github.com/Haoyulance/SC-GAN/blob/master/requirements.txt).

Please use ```pip install requirement.txt``` to install the requirements



## Running the code
SC-GAN was assessed for positron emission tomography (PET), Fractional anisotropy (FA) and mean diffusivity (MD) synthesis from tasks using T1w and FLAIR magnetic resonance images (MRI).

### Training:
Training script is at  **./training**

Use the following command to run the training script:

```python training.py --trainig_size= --gpu= --epoches= --img_size= --data_dir= --modalities= --logdir=```

|configurations|meaning|default|
|---|---|---|
|--training_size|the number of training data to use|None|
|--gpu|gpu ID for training|None|
|--epoches|the number of training epoches|120|
|--img_size|input image size(same for three dimensions)|256|
|--data_dir|data directory|None|
|--modalities|modalities to use in the training. Last one is the name of target modality, the rests are names of input modalities(eg: modality1_modality2_..._target) |flair_t1w_av45|
|--logdir|directory to save tensorboard log |None|

Add more parameters configuration to do hyperparameter tuning:

|parameters|meaning|default values|
|---|---|---|
|--lr|learning rate|0.001|
|--g_reg|generator regularizer|0.001|
|--d_reg|discriminator regularizer|0.001|
|--reg|if using regularization|True|
|--l1_weight|l1 weight|200|
|--B_weight|B-rmse weight|200|

### Testing:

Testing script is at  **./testing**

Use the following command to run the testing script:

```python testing.py --test_size= --gpu= --img_size= --data_dir= --modalities= --logdir= --output=```

|configurations|meaning|default|
|---|---|---|
|--test_size|the number of test data to use|None|
|--gpu|gpu ID for testing|None|
|--img_size|input image size(same for three dimensions)|256|
|--data_dir|data directory|None|
|--modalities|modalities to use in the testing. Last one is the name of target modality, the rests are names of input modalities(eg: modality1_modality2_..._target) |flair_t1w_av45|
|--log_dir|directory to read tensorboard log|None|
|--output|synthesis results directory|None|


## License

If you use this code for your research, please be familiar with the [LICENSE](./LICENSE) and cite our paper.

## Developer
Haoyu Lan  
INI Microstructural imaging Group ([IMG](https://www.ini.usc.edu/img/))   
USC Stevens Neuroimaging and Informatics Institute  
Keck School of Medicine, University of Southern California

## Citation
Lan H, Toga AW, Sepehrband F. SC-GAN: 3D self-attention conditional GAN with spectral normalization for multi-modal neuroimaging synthesis, *under review* [bioRxiv copy](https://www.biorxiv.org/content/10.1101/2020.06.09.143297v1)

