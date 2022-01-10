import numpy as np
import random
from utility import LossHistory
import keras
from utility import convert_dataset,LossHistory
import pickle
from keras.utils import to_categorical


def load_poisoning_data(file_name):
    file_name = "poisoningData/" + file_name + ".pkl"
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    poisoning_data = {}

    poisoning_data["X"] = np.array(data["X"])
    poisoning_data["Y"]  = np.array(data["Y"])

    return poisoning_data


def save_poisoning_data(file,poisoning_data):
    file = "poisoningData/" + file +".pkl"
    f = open(file,"wb")
    pickle.dump(poisoning_data,f)
    f.close()



def posioning_attack(type,dataset,model_structure,poisoning_fraction,training_info,is_load):
    #input:
    #type: the attacker algorithm
    #dataset: {
    #   "clean_train": clean training dataset
    #   "clean_test": clean test dataset
    #   }
    #model_structure: model of the victim attack
    #poisoning_fraction: generate how many poisoning points
    #output:
    #the posioned dataset
    clean_train = dataset["clean_train"]
    clean_test = dataset["clean_test"]

    x_train,y_train,x_test,y_test =convert_dataset(training_info['model_name'],clean_train["X"],clean_train["Y"],clean_test["X"],clean_test["Y"],training_info)


    if type == 'loss':
        posioning_data = loss_attack(x_train,y_train,x_test,y_test,poisoning_fraction,model_structure,is_load)
    elif type == 'lable_attack':
        posioning_data = label_flip_attack(x_train,y_train,x_test,y_test,poisoning_fraction,is_load)
    elif type == 'random_attack':
        posioning_data = noisy_attack(x_train,y_train,x_test,y_test,poisoning_fraction,is_load)
    elif type == 'gradient_ascent':
        posioning_data = gradient_ascent_attack(x_train,y_train,training_info['dataset_name'],poisoning_fraction,is_load,n_round=training_info['posioned_round'])
    elif type == 'min_max':
        posioning_data = min_max_attack(x_train,y_train,x_test,y_test,poisoning_fraction,model_structure,is_load)
    elif type == 'influence':
        posioning_data = influence_attack(clean_train,clean_test,poisoning_fraction,is_load)
    elif type == 'generative':
        posioning_data = generative_model(x_train,y_train,training_info['dataset_name'],poisoning_fraction,is_load,n_round=training_info['posioned_round'])

    dataset["poisoning_data"] = posioning_data
    return dataset

def label_flip_attack(x_train,y_train,x_test,y_test,fraction,is_load,num_classes=10):
    # This function is generate the lable flip attack. The lable is random selected.
    # input : x_train,y_train,x_test,y_test,poisoning fraction
    # output: x_p,y_p

    sample = x_train.shape[0]
    poisoing_sample = int(sample * fraction)

    x_p = np.copy(x_train[:poisoing_sample])
    y_p = np.copy(y_train[:poisoing_sample])
    y_p = np.random.randint(num_classes, size=y_p.shape[0])
    if len(y_train.shape) != len(y_p.shape):
        y_p = to_categorical(y_p, num_classes=num_classes)
    poisoning_data = {}
    poisoning_data["X"] = x_p
    poisoning_data["Y"] = y_p

    file_name = 'label_flip_' + str(fraction)
    save_poisoning_data(file_name,poisoning_data)
    return  poisoning_data




def loss_attack(x_train,y_train,x_test,y_test,fraction,model,is_load,n_built_in = 5):
    # This function is to generate the posioning data. Output the examples which has large loss value.
    # input : x_train,y_train,x_test,y_test,poisoning fraction
    # output: x_p,y_p

    model.fit(x_train,y_train,epochs=n_built_in,verbose=0)
    n_bad_points = int(fraction*x_train.shape[0])
    poisoning_point = {}
    history = LossHistory()

    loss = model.evaluate(x_train,y_train,verbose = 0,callbacks=[history],batch_size=1)
    max_index = np.argsort(history.losses)[-n_bad_points:]
    x_p = x_train[max_index]
    y_p = y_train[max_index]

    poisoning_point['X'] = x_p
    poisoning_point['Y'] = y_p

    file_name = 'loss_attack_' + str(fraction)
    save_poisoning_data(file_name,poisoning_point)
    print(y_p.shape)
    return poisoning_point


def gradient_ascent_attack(x_train,y_train,datatype,fraction,is_load,n_round,num_classes=10):
    # This function is load the pre-trained poisoning data trained on the gradient_ascent model
    # input : x_train,y_train,x_test,y_test,poisoning fraction
    # output: x_p,y_p
    sample = x_train.shape[0]
    poisoing_sample = int(sample * fraction)

    filename = 'poisoningData/gradient_attack_'+datatype+'.npz'
    data = np.load(filename)

    x_p = data['X'][:,n_round]
    x_p = x_p.reshape(-1,data['X'].shape[-1])
    y_p = data['Y'][:,n_round].reshape(-1)
    y_p = y_p%10
    if len(y_train.shape) != len(y_p.shape) :
        y_p = to_categorical(y_p, num_classes=num_classes)

    if poisoing_sample > x_p.shape[0]:
        n_repeat = int(poisoing_sample/x_p.shape[0])
        x_p = np.repeat(x_p,n_repeat,axis=0)
        y_p = np.repeat(y_p,n_repeat,axis=0)

    poisoning_data= {}
    poisoning_data["X"] = x_p
    poisoning_data["Y"]  = y_p
    return poisoning_data

def generative_model(x_train,y_train,datatype,fraction,is_load,n_round,num_classes=10):
    # This function is load the pre-trained poisoning data trained on the generative model
    # input : x_train,y_train,x_test,y_test,poisoning fraction
    # output: x_p,y_p

    sample = x_train.shape[0]
    poisoing_sample = int(sample * fraction)

    filename = 'poisoningData/generative_attack_'+datatype+'.npz'
    data = np.load(filename)

    x_p = data['X'][:,n_round]
    x_p = x_p.reshape(-1,data['X'].shape[-1])
    y_p = data['Y'][:,n_round].reshape(-1)

    if len(y_train.shape) != len(y_p.shape) :
        y_p = to_categorical(y_p, num_classes=num_classes)

    if poisoing_sample > x_p.shape[0]:
        n_repeat = int(poisoing_sample/x_p.shape[0])
        x_p = np.repeat(x_p,n_repeat,axis=0)
        y_p = np.repeat(y_p,n_repeat,axis=0)

    poisoning_data= {}
    poisoning_data["X"] = x_p
    poisoning_data["Y"]  = y_p
    return poisoning_data

def noisy_attack(x_train,y_train,x_test,y_test,fraction,is_load,num_classes=10):
    # This function is generate the noise attack. Add random pixel to the origin image.
    # input : x_train,y_train,x_test,y_test,poisoning fraction
    # output: x_p,y_p

    sample = x_train.shape[0]
    poisoing_sample = int(sample * fraction)

    x_p = np.copy(x_train[:poisoing_sample])
    y_p = np.copy(y_train[:poisoing_sample])
    noisy = np.random.normal(0,1,x_p.shape)
    x_p = x_p + noisy

    poisoning_data = {}
    poisoning_data["X"] = x_p
    poisoning_data["Y"] = y_p

    file_name = 'noisy_attack_' + str(fraction)
    save_poisoning_data(file_name,poisoning_data)
    print(y_p.shape)
    return  poisoning_data
