# DynamicTrack

PyTorch implementation of our DynamicTrack tracker.

The results for four tracking benchmarks (UAV123, GOT10K, LaSOT, and TrackingNet) have been released. 

## 1. Dataset Introduction

This code is compatible with four challenging tracking benchmarks, i.e., UAV123, GOT10K, LaSOT, and TrackingNet. 

These datasets can be obtained as follows:
#### UAV123: https://cemse.kaust.edu.sa/ivul/uav123
#### GOT10K: http://got-10k.aitestunion.com/
#### LaSOT: https://cis.temple.edu/lasot/
#### TrackingNet: https://tracking-net.org/

## 2. Results on these four datasets

<img src="https://github.com/chenxlin222/DynamicTrack/blob/main/results/results.png" width="375px"> 

## 3. Running instructions

DynamicTrack is implemented purely based on the PyTorch.

### Install the environment 

    bash install_pytorch17.sh

### Data Preparation

put the tracking datasets (UAV123, GOT10K, LaSOT, and TrackingNet) in ./data.

### Set paths

Run the following command to set paths for this project:

    python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .

### Training

The training for the proposed DynamicTrack consists of two steps: The first step is to train the cascaded backbone, the encoder-decoder Transformer and the prediction head. The second step is to train the adaptive router.

#### (1) The first step

    python tracking/train.py --script dynamictrack --config baseline --save_dir . --mode multiple --nproc_per_node 8

#### (2) The second step

When starting the second step, the following files need to be modified: 

① The line 71 in "tracker_code/lib/train/run_training.py" should be changed to "expr_module = importlib.import_module('lib.train.train_script_step2')"
② "tracker_code/lib/models/dynamictrack/stark_s.py" should changed to the codes for step2.
③ "tracker_code/lib/train/actors/stark_s.py" should changed to the codes for step2.

Then, run the following:
    
    python tracking/train.py --script dynamictrack --config baseline_step2 --save_dir . --mode multiple --nproc_per_node 8

