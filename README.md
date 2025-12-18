# MS-SHOT
The official code of paper "**Exploiting Radio Frequency Fingerprints for Device Identification: Tackling Cross-receiver Challenges in the Source-data-free Scenario**". 

### Step 1. Prepare Python envirment

```
conda create --name msshot python=3.13 -y
conda activate msshot
pip install -r requirements.txt
```

### Step 2. Prepare datasets

The processed dataset is in [here](https://drive.google.com/file/d/1tlQk3Jcsq5Tib9FBQXj13EHTAybqmBGA/view?usp=drive_link).

Then you should unzip the dataset.

### Step 3. Try

1. Run source model
```
python main.py -c configs/A_source_only.toml -Rx_s 7-7 -Rx_t 8-8 -data_dir your/dataset/path
```

2. Adapt to target domain
```
python main.py -c configs/SHOT_fbnm_nmlzabs_softlabel_temp0.1_emac.toml -Rx_s 7-7 -Rx_t 8-8 -data_dir your/dataset/path
```