#Combogrid problem to index
COMBO_TO_IDX = {
    "BL-TR" : 0,
    "BR-TL" : 1,
    "TL-BR" : 2,
    "TR-BL" : 3,
    "test" : 4
}

IDX_TO_COMBO = dict(zip(COMBO_TO_IDX.values(), COMBO_TO_IDX.keys()))

#(agent_pos, goal_pos) for fourroom grid size 19 to index
FOUROOM_TO_IDX = {
    ((3, 4), (6, 15)) : 0,
    ((3, 7), (15, 3)) : 1,
    ((15, 7), (3, 15)) : 2

}

IDX_TO_FOUROOM = dict(zip(FOUROOM_TO_IDX.values(), FOUROOM_TO_IDX.keys()))


#Default paths
TRAJ_DIR = 'training_data/trajectories/'
MODEL_DIR = 'training_data/models/'
OPTION_DIR = 'training_data/options/'