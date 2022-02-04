# %%
import os
#os.environ['CUDA_VISIBLE_DEVICES']='-1'

import tensorflow as tf
from tensorflow import keras

import numpy as np

from depois_model import DePoisModel
from depois_attack import DePoisAttack
from utils import *
import pickle




overall_stats = {"bb": {}, "wb": {}}

attack_boxes = ['bb', 'wb']

if os.path.exists('stats/overall_stats.pkl'):
    overall_stats = pickle.load(open('stats/overall_stats.pkl', 'rb'))
else:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = preprocess_data(x_train, x_test)

    depois_attack = DePoisAttack()
    depois_attack.clone_critic(x_train, y_train)
    depois_attack.clone_classifier(x_train, y_train)

    depois_model = DePoisModel((x_test, y_test))

    for attack_box in attack_boxes:
        for attack_mode in ['CR_only', 'CL_only', 'CR_then_CL', 'CL_then_CR']:
            overall_stats[attack_box][attack_mode] = {}
            budgets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            for eps in budgets:
                print(eps)
                if attack_box == 'wb':
                    x_test_adv, _ = depois_attack.wb_attack(depois_model, (x_test, y_test), eps, attack_mode=attack_mode)
                else:
                    x_test_adv, _ = depois_attack.bb_attack(depois_model, (x_test, y_test), eps, attack_mode=attack_mode)

                stats = depois_model.evaluate(x_test, x_test_adv, y_test, eps)
                #vis_predictions(x_test_adv, y_test, len(y_test))
                overall_stats[attack_box][attack_mode][eps] = stats
                print(stats)

    pickle.dump(overall_stats, open('stats/overall_stats.pkl', 'wb'))

for attack_box in attack_boxes:
    graph_stats(overall_stats[attack_box], attack_box)

graph_wb_vs_bb(overall_stats)
