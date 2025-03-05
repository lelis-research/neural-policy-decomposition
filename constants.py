#Combogrid problem to index
COMBO_TO_IDX = {
    "BL-TR" : 0,
    "BR-TL" : 1,
    "TL-BR" : 2,
    "TR-BL" : 3,
    "test" : 4
}

IDX_TO_COMBO = dict(zip(COMBO_TO_IDX.values(), COMBO_TO_IDX.keys()))

#Default paths
TRAJ_DIR = 'training_data/trajectories'
MODEL_DIR = 'training_data/models/'
OPTION_DIR = 'training_data/options/'