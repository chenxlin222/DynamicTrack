class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/root/autodl-tmp/STARK'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/root/autodl-tmp/STARK/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/root/autodl-tmp/STARK/pretrained_networks'
        self.lasot_dir = '/root/autodl-tmp/STARK/data/lasot'
        self.got10k_dir = '/root/autodl-tmp/STARK/data/got10k/train'
        self.lasot_lmdb_dir = '/root/autodl-tmp/STARK/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/root/autodl-tmp/STARK/data/got10k_lmdb'
        self.trackingnet_dir = '/root/autodl-tmp/STARK/data/trackingnet'
        self.trackingnet_lmdb_dir = '/root/autodl-tmp/STARK/data/trackingnet_lmdb'
        self.coco_dir = '/root/autodl-tmp/STARK/data/coco'
        self.coco_lmdb_dir = '/root/autodl-tmp/STARK/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/root/autodl-tmp/STARK/data/vid'
        self.imagenet_lmdb_dir = '/root/autodl-tmp/STARK/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
