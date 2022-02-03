import os
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn import metrics
import tensorflow.keras as keras 

os.makedirs('images', exist_ok=True)
os.makedirs('data/weights', exist_ok=True)

def preprocess_data(x_train, x_test):
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    return x_train, x_test

def load_data():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = preprocess_data(x_train, x_test)
    return (x_train, y_train), (x_test, y_test)


def TrueAndGeneratorData(know_rate, epochs):
    
    (x, y), (X_test, y_test) = load_data()
    trainct = int(know_rate * y.shape[0])   
    train_data = x[0:trainct,:,:]
    train_label = y[0:trainct]
    generator_size = len(y) - trainct

    if (os.path.exists("data/Generator_data_48000_200000.npy")):
        G_data = np.load(f"data/Generator_data_48000_200000.npy")
        G_label = np.load(f"data/Generator_label_48000_200000.npy")
        print("Loaded generator data successfully!")
    else:
        from generator_CGAN_authen import CGAN_data_loss
        CGAN_data_loss(know_rate, epochs)
        
        G_data = np.load(f"data/Generator_data_{generator_size}_{epochs}.npy")
        G_label = np.load(f"data/Generator_label_{generator_size}_{epochs}.npy")
    
    Max = np.max(G_data)
    Min = np.min(G_data)
    G_data = (G_data - Min) / (Max - Min)
    G_data = G_data * (1 * (G_data > 0.3))
    xx = np.concatenate((train_data/255,G_data), axis = 0)
    yy = np.concatenate((train_label,np.squeeze(G_label)), axis =0) # 0 ~ 1
    np.save("data/saved_TrueAndGeneratorData.npy", xx)
    np.save("data/saved_TrueAndGeneratorLabel.npy", yy)
    print('CGAN_loss True And Generator Data saved!')
        

def poi_data(poison_rate):
        
    poison_number = int(poison_rate * 50000)
    data = np.load("dataset/GP_mnist.npz")
    print(list(data.keys()))
    poisoned_x_data = data["X"][:, 0, :,:]
    poisoned_y_data = data["Y"][:, 0]-1
    poisoned_x_data = poisoned_x_data * 255
    poisoned_x_data = poisoned_x_data.reshape(poisoned_x_data.shape[0],28,28, 1)
    poisoned_x_data = (poisoned_x_data.astype(np.float32) - 127.5) / 127.5
    print(poisoned_x_data.min(), poisoned_x_data.max())

    return len(poisoned_x_data), poisoned_x_data, poisoned_y_data


def count_score(validity, poison_number):
    # real 1, poi 0
    label_poisoned_real = np.ones(validity.shape[0])
    label_poisoned_real[-poison_number:] = 0
    
    z_scores = np.mean(validity[0:-poison_number]) - np.std(validity[0:-poison_number])
    f_poisoned_data = np.where(validity<=z_scores)[0]
    
    label_poisoned_fake = np.ones(validity.shape[0])
    label_poisoned_fake[f_poisoned_data] = 0
    
    P = metrics.precision_score(label_poisoned_real,label_poisoned_fake.astype(float),average='binary')
    R = metrics.recall_score(label_poisoned_real,label_poisoned_fake.astype(float),average='binary')
    F1 = (2 * P * R) / (P + R)
    acc = metrics.accuracy_score(label_poisoned_real,label_poisoned_fake.astype(float))
    acc_adv = metrics.accuracy_score(label_poisoned_real[-poison_number:],label_poisoned_fake[-poison_number:].astype(float))
    acc_real = metrics.accuracy_score(label_poisoned_real[0:-poison_number],label_poisoned_fake[0:-poison_number].astype(float))

    return F1, P, R, acc, acc_adv, acc_real



def model_defense(poi_rate,  epochs, will_load_model=False):
    print("starting model defense!")
    (X_train, y_train), (_, _) = load_data()
    #X_train = np.expand_dims(X_train, axis=1)
    
    poison_number, poisoned_x_data, poisoned_y_data = poi_data(poi_rate)
    x_poisoned_raw_test = np.concatenate((X_train[0:50000],poisoned_x_data))   
    y_poisoned_raw_test = np.concatenate((y_train[0:50000],np.squeeze(poisoned_y_data)))
    
    from mimic_model_construction import CWGANGP
    batch_size = 32
    sample_interval = 500
    
    start = time.perf_counter()
    wgan = CWGANGP(epochs, batch_size, sample_interval)
    wgan.train(will_load_model)
    end = time.perf_counter()
    print('Running time for train WGANGP: %s Seconds'%(end-start))

    D_poi = wgan.discriminate_img(x_poisoned_raw_test, y_poisoned_raw_test)
    D_poi = D_poi.flatten()
    print(D_poi.max(), D_poi.min())
    plt.figure()
    plt.title('CWGANGP D_value')  
    plt.hist(D_poi[0:50000],bins = 100,color = 'b')
    plt.hist(D_poi[50000:],bins = 100,color = 'r')
    
    F1, P, R, acc, acc_adv, acc_real = count_score(D_poi, poison_number)
    print('Accuracy of De-pois:')
    print(acc)
    print('Precision of De-pois：')
    print(P)
    print('Recall of De-pois：')
    print(R)
    print('F1 of De-pois：')
    print(F1)
    print('acc_adv：')
    print(acc_adv)
    
    return None



if __name__ == '__main__':
    poi_rate = 0.30
    know_rate = 0.20
    steps = 10000
    #data = poi_data(5)
    #dataset = load_data()
    #generated_data = np.load('data/Generator_data_48000_200000.npy')
    #TrueAndGeneratorData(know_rate, steps)
    model_defense(poi_rate, steps, will_load_model=True)