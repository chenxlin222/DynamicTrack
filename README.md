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

#### The first step

The following parts need to be changed:(1)

