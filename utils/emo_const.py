"""
This file contains some wide used variable that common in emotion
"""

# FER/FERPLUS labels and id map
ferlabels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
ferlbl2id = {val: pos for pos, val in enumerate(ferlabels)}

emowlabels = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear']
emowlabels = [o.lower() for o in emowlabels]
emowlbl2id = {val: pos for pos, val in enumerate(emowlabels)}

# from FER+ train set use `prlab.emotion.ferplus.data_helper.calc_train_emotion_distribution`
# calc_train_emotion_distribution(csv_path=csv_path, train_path=train_path)
fer_emo_dis = [
    [0.777334, 0.03261, 0.017447, 0.114091, 0.026224, 0.007145, 0.004379, 0.020772],
    [0.042842, 0.917618, 0.023221, 0.004816, 0.004217, 0.001969, 0.001755, 0.003563],
    [0.058354, 0.035775, 0.787254, 0.01037, 0.014319, 0.003015, 0.088513, 0.0024],
    [0.170124, 0.013292, 0.008632, 0.747902, 0.022808, 0.011237, 0.015923, 0.010082],
    [0.074781, 0.02535, 0.060215, 0.031924, 0.740551, 0.034938, 0.023251, 0.008989],
    [0.08959, 0.017445, 0.016911, 0.088629, 0.125152, 0.620813, 0.007788, 0.033671],
    [0.042815, 0.012239, 0.169483, 0.072557, 0.027228, 0.013045, 0.658964, 0.003671],
    [0.147933, 0.022174, 0.00989, 0.042923, 0.033551, 0.059924, 0.005507, 0.678099]
]

# this order is alphabet order (default when load with folder)
rafdb_labels_names = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
fer2rafdb_mapping = [4, 5, 6, 1, 0, 3, 2]

# mapping from fer_emo_dis
rafdb_emo_dis = [[fer_emo_dis[i][j] for j in fer2rafdb_mapping] for i in fer2rafdb_mapping]

affect_net_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
fer2affectnet_mapping = [0, 1, 3, 2, 6, 5, 4, 7]
affectnet_emo_dis = [[fer_emo_dis[i][j] for j in fer2affectnet_mapping] for i in fer2affectnet_mapping]
