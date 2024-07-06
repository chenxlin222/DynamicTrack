from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/root/autodl-tmp/data/got10k_lmdb'
    settings.got10k_path = '/root/autodl-tmp/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/root/autodl-tmp/data/lasot_lmdb'
    settings.lasot_path = '/root/autodl-tmp/data/lasot'
    settings.network_path = '/root/autodl-tmp/STARK/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/root/autodl-tmp/data/nfs'
    settings.otb_path = '/root/autodl-tmp/data/OTB2015'
    settings.prj_dir = '/root/autodl-tmp/STARK'
    settings.result_plot_path = '/root/autodl-tmp/STARK/test/result_plots'
    settings.results_path = '/root/autodl-tmp/STARK/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/root/autodl-tmp/STARK'
    settings.segmentation_path = '/root/autodl-tmp/STARK/test/segmentation_results'
    settings.tc128_path = '/root/autodl-tmp/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/root/autodl-tmp/data/trackingnet'
    settings.uav_path = '/root/autodl-tmp/data/UAV123'
    settings.vot_path = '/root/autodl-tmp/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

