import tensorflow as tf
import tensorflow_similarity as tfsim
from matplotlib import pyplot as plt
import numpy as np
import time
import os
# from tensorflow_similarity.utils import tf_cap_memory
# tfsim.utils.tf_cap_memory()
from tensorflow_similarity.layers import MetricEmbedding # row wise L2 norm
from tensorflow_similarity.losses import MultiSimilarityLoss  # specialized similarity loss
from tensorflow_similarity.losses import TripletLoss
from tensorflow_similarity.losses import CircleLoss
from tensorflow_similarity.models import SimilarityModel # TF model with additional features
from tensorflow_similarity.samplers import MultiShotMemorySampler  # sample data 
from tensorflow_similarity.architectures import EfficientNetSim

from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback
import gc

#GLOBAL PARAMETERS
MODEL_NAME = 'EXT28NEW2_ENETB0CIR_512_fullext3_finetune'
SQUARE = 224
RES = (SQUARE,SQUARE)
batch_size = 32
INPUT_SHAPE = (SQUARE,SQUARE,3)
epochs = 1000

#PATH
DATA_PATH = 'D:\Downloads\FaceData\get-dataset-cn340/facedetected_folder_clean'
SAVE_DIR = 'D:\Downloads/facednet/Model'


#Generate seeds by using time
seconds = time.time()
local_time = time.ctime(seconds)
local_time = local_time[11:13]+local_time[14:16]+local_time[17:19]
SEED = int(seconds)


def main():
    fold = 1
    #CLASSES VARIABLE
    CLASSES = get_classes_name(DATA_PATH)
    NUM_CLASSES = len(CLASSES)

    #Create output directory
    out_dir = SAVE_DIR+'/'+MODEL_NAME+'_'+str(SEED)+'_'+local_time
    try :
        os.mkdir(out_dir)
        print("Create OUTPUT folder")
    except OSError :
        print("Fail to create OUTPUT folder or Already Exist")
    os.chdir(out_dir)
    print(out_dir)

    #Data processing
    print('CLASSES :',CLASSES)

    print("Loading train & validate set 90%")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                DATA_PATH,
                validation_split=0.1,
                subset="training",
                shuffle = True,
                labels='inferred',
                label_mode='int',
                seed=SEED,
                image_size=RES,
                class_names = CLASSES,
                color_mode = 'rgb',
                batch_size=batch_size)

    print("Loading test set 10%")
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                DATA_PATH,
                validation_split=0.1,
                subset="validation",
                shuffle = True,
                labels='inferred',
                label_mode='int',
                seed=SEED,
                image_size=RES,
                class_names = CLASSES,
                color_mode = 'rgb',
                batch_size=batch_size)
    
    
    #Sample generator # y_train must be int tensor
    x_train,y_train = split_xy(train_ds)
    del train_ds
    gc.collect()
    #High memory usage
    
    x_train,y_train = augment_concate(x_train,y_train,augment = 'fullext')
    print("Trainning after augmentation :",len(x_train))
    x_val,y_val = split_xy(val_ds)
    del val_ds
    gc.collect()
    print(x_train.shape)
    print(y_train.shape)
    #Batch size = classes * examples
    sampler = MultiShotMemorySampler(x_train,y_train,
                                    classes_per_batch = NUM_CLASSES,
                                    examples_per_class_per_batch = 3,
                                    class_list = range(NUM_CLASSES),
                                    steps_per_epoch = 1000
                                    )
    del x_train
    del y_train
    gc.collect()
    
    #Model
    model = EfficientNetSim(input_shape=INPUT_SHAPE,
                            embedding_size = 512,
                            variant = "B0",
                            augmentation = tf.keras.Sequential([
                                                                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")
                                                                ]),
                            weights = None #'b0_weight/efficientnetv1-b0-noisy_student.h5' #None #"imagenet" #'/b0_weight/noisy.student.notop-b0.h5'
                            ) 

    #model = create_modelB0()
    
    print(model.summary())
    model.compile(optimizer = 'adam', 
                loss = CircleLoss(distance ='cosine') #TripletLoss #CircleLoss #MultiSimilarityLoss
    #,metrics=['accuracy']
    ) 

    checkpoint_dir = "model/f%d" % fold
    checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}.ckpt"
    
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs*0.05,restore_best_weights=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, 
                verbose=0, 
                save_weights_only=False,
                monitor='val_loss',
                save_best_only= True)
    tb_callback = tf.keras.callbacks.TensorBoard(
                log_dir="./logs/f%d" % fold,\
                update_freq="epoch") 

    history = model.fit(
        sampler,
        validation_data=(x_val,y_val),
        epochs=epochs,
        callbacks=[cp_callback,
                tb_callback,
                es_callback
                #ClearMemory()
                ])
        
    
    model.save(checkpoint_dir + "/last")


    print("============= Model Result at last epoch ==================")
    test_loss, test_acc = model.evaluate(x_val,y_val, verbose=2)
    print('\nTest loss:', test_loss)
    print('\nTest accuracy:', test_acc)

    f = open("test%d.txt" %fold, "w")
    f.write('Test loss: {} \nTest accuracy: {}'.format(test_loss,test_acc))
    f.close()
    print("========================================")

class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()

def create_modelB0():
    
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    
    x = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")(inputs)
                                                                
    x = tf.keras.applications.EfficientNetB0(
        include_top=False, weights= "imagenet" #None #"imagenet" #'noisy-student'
        , pooling=None,
        )(x)

    outputs = MetricEmbedding(512)(x)

    return SimilarityModel(inputs, outputs)

def augment_concate(images,labels,augment:str = 'full'):
    if augment == 'fullext' :
        new_images = np.concatenate((images,lighting_augment(images),rotate_augment(images),rotate_augment(lighting_augment(images)),rotate_augment(hue_augment(images)),hue_augment(images)),axis = 0)
        new_labels = np.concatenate((labels,labels,labels,labels,labels,labels),axis = 0)

    elif augment == 'full':
        new_images = np.concatenate((images,lighting_augment(images),rotate_augment(images),hue_augment(images)),axis = 0)
        new_labels = np.concatenate((labels,labels,labels,labels),axis = 0)
        
    elif augment == 'rotate':
        new_images = np.concatenate((images,rotate_augment(images)),axis = 0)
        new_labels = np.concatenate((labels,labels),axis = 0)
        
    elif augment == 'lighting':
        new_images = np.concatenate((images,lighting_augment(images)),axis = 0)
        new_labels = np.concatenate((labels,labels),axis = 0)
    
    elif augment == 'roli' :
        new_images = np.concatenate((images,rotate_augment(lighting_augment(images))),axis = 0)
        new_labels = np.concatenate((labels,labels),axis = 0)
    else :
        print("Invalid Augment only choose ['full','rotate','lighting']")
        return images,labels

    new_ds = tf.data.Dataset.from_tensor_slices((new_images,new_labels)).shuffle(len(new_images), reshuffle_each_iteration=True)
    return tensordata2np(new_ds)

def rotate_augment(data_set):
    augmentrotate = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.05,fill_mode = 'nearest'),
                                        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")
                                        ])
    return augmentrotate(data_set)


def lighting_augment(data_set) :
    augmented = []
    for i in data_set :
        img = tf.image.random_brightness(i, 0.3)
        augmented.append(img)
    
    return np.array(augmented)

def hue_augment(data_set) :
    augmented = []
    for i in data_set :
        img = tf.image.random_hue(i, 0.1)
        augmented.append(img)
    
    return np.array(augmented)

def tensordata2np(dataset) :
    dataset = list(dataset.as_numpy_iterator())
    images = list()
    labels = list()
    for img,label in dataset :
            images.append(img)
            labels.append(label)
    images = np.array(images,dtype=np.uint8)
    labels = np.array(labels,dtype=np.uint8)
    return images,labels

def split_train_val(data_set) :
    total_ds = len(data_set)
    train_size = int(0.9 * total_ds)
    train_ds = data_set.take(train_size)
    val_ds = data_set.skip(train_size)
    print("Split to 90% Trainning set &  10% Validation set")
    print('Trainning set :',len(train_ds))
    print('Validation set :',len(val_ds))
    return train_ds,val_ds


def split_xy(data_set) :
    #loop batch
    images = list()
    labels = list()
    for img_batch,label_batch in data_set :
        for i in range(len(img_batch)) :
            images.append(img_batch[i].numpy().astype("uint8"))
            labels.append(label_batch[i].numpy().astype("uint8"))
    images = np.array(images)
    labels = np.array(labels)
    return images.squeeze(),labels.reshape(-1)


def get_classes_name(path) :
    for i,y in enumerate(os.walk(path)):
        subdirs,dirs,files = y
        if i == 0 :
            return dirs


def show_ds(data_set,classes_name) :
    plt.figure(figsize=(10, 10))
    for images, labels in data_set.take(1):
        for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(classes_name[labels[i]])
                plt.axis("off")
    plt.show()  



if __name__ == "__main__":
    main()
