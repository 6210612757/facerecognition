import tensorflow as tf
import tensorflow_similarity as tfsim
from tensorflow_similarity.samplers import select_examples  # select n example per class
from tensorflow_similarity.visualization import viz_neigbors_imgs  # neigboors vizualisation
from tensorflow_similarity.architectures import EfficientNetSim
from tensorflow_similarity.visualization import confusion_matrix  # matching performance
from tabulate import tabulate
from tensorflow_similarity.utils import tf_cap_memory
tfsim.utils.tf_cap_memory()
from model_trainning import *
import numpy as np
#  205ok 218kindaoverfit 
# 'D:\Downloads/facednet\Model\EXT28NEW_ENETB0CIR_512_fullaugin_1634837475_003115\model/f1\cp-0128.ckpt' 91.01
# 'D:\Downloads/facednet\Model\EXT28_ENETB0CIR_512_fullaugin_1634581815_013015\model/f1\cp-0205.ckpt' 94.01
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_roli_1635526378_235258\model/f1/cp-0119.ckpt' 96.40 std 0.027
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_roli_1635786470_000750\model/f1\cp-0156.ckpt' 94.61

# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_roli_imgnet_1635875046_004406\model/f1\cp-0049.ckpt' 97.01 @ 0.407 std 0.1
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_roli_imgnet_1635976886_050126\model/f1\cp-0032.ckpt' 95.8 @0.3735 
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_roli_imgnet_tune_1636012169_144929\model/f1\cp-0023.ckpt' 97.60 @0.42 std 0.1
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_full_imgnet_tune_1636041156_225236\model/f1\cp-0006.ckpt' 97 @0.33 std 0.1
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext_imgnet_tune_1636081025_095705\model/f1\cp-0048.ckpt' 96 @0.36 std 0.11
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext_imgnet_tune_1636187269_152749\model/f1\cp-0015.ckpt' 97.60 @0.419 std 0.1

# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext_tune_1636129305_232145\model/f1\cp-0160.ckpt' onew 95 @0.12 std 0.035
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext_tune_1636129305_232145\model/f1\cp-0210.ckpt' new 97.01 @0.12 std 0.034 Unknown 1 false
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext_tune_1636213677_224757\model/f1\cp-0225.ckpt' 97.60 @ 0.09282 std 0.0329
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext_tune_1636213677_224757\model/f1\cp-0235.ckpt' 97.60 @0.108 std 0.037
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext_tune1_1636303916_235156\model/f1\cp-0142.ckpt' 95.21
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext_tune1_1636398774_021254\model/f1\cp-0249.ckpt' 94
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext_tune1_1636398774_021254\model/f1\cp-0298.ckpt' 98.20 @0.1383 std 0.0353
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext2_1636560347_230547\model/f1\cp-0209.ckpt' 96.41
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext2_1636644151_222231\model/f1\cp-0050.ckpt' 95
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext3_1636736271_235751\model/f1\cp-0226.ckpt' 95.81 @0.0909
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext3_1636736271_235751\model/f1\cp-0243.ckpt' 98.20 @0.0982 std 0.0296
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext2_finetune_1636911730_004210\model/f1\cp-0171.ckpt' 96.4
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext3_finetune_1637258779_010619\model/f1\cp-0146.ckpt' 95.81
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext3_finetune_1637258779_010619\model/f1\cp-0139.ckpt' 97.01@0.0852
# 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext3_finetune_1637258779_010619\model/f1\cp-0138.ckpt' 97.01@0.0793


MODEL_PATH = 'D:\Downloads/facednet\Model\EXT28NEW2_ENETB0CIR_512_fullext3_finetune_1637258779_010619\model/f1\cp-0138.ckpt'
DATA_PATH = 'D:\Downloads\FaceData\get-dataset-cn340/facedetected_folder_index'
TEST_PATH = 'D:\Downloads\FaceData\get-dataset-cn340/testdataset'
display = 1
RES = (224,224)

def main():
    
    #Generate seeds by using time
    # local_time = time.ctime(time.time())
    # local_time = local_time[11:13]+local_time[14:16]+local_time[17:19]

    #Load data for index
    CLASSES = get_classes_name(DATA_PATH)
    NUM_CLASSES = len(CLASSES)
    print(CLASSES)
    print("Loading train & validate set 90%")
    trainval_ds = tf.keras.preprocessing.image_dataset_from_directory(
                DATA_PATH,
                #FACE_PATH,
                shuffle = True,
                labels='inferred',
                label_mode='int',
                image_size=RES,
                class_names = CLASSES,
                color_mode = 'rgb',
                batch_size=1)

    print("Loading test set 10%")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                TEST_PATH,
                shuffle = True,
                labels='inferred',
                label_mode='int',
                image_size=RES,
                color_mode = 'rgb',
                batch_size=batch_size)
    
    
    
    #Sample generator # y_train must be int tensor
    x_train,y_train = split_xy(trainval_ds)
    x_test,y_test = split_xy(test_ds)

    #Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    #Index the data to the model
    
    model.reset_index()
    model.index(x_train,y_train,data = x_train)

    #Show example of nn
    
    num_neighboors = 5
    MATCHER = 'match_nearest'#'match_majority_vote'

    # lookup nearest neighbors in the index
    nns = model.lookup(x_test, k=num_neighboors)
    #print("NNS :",nns[0][0].distance) #Look up object have attribute distance (Lookup in type.py)
    CLASSES.append('Unknown')
    if display :
        for idx in np.argsort(y_test):
            viz_neigbors_imgs(x_test[idx], y_test[idx], nns[idx], 
                            #class_mapping=CLASSES,
                            fig_size=(16, 2))

    calibration = model.calibrate( #calibrate the threshold from testset
        x_test, 
        y_test, 
        extra_metrics=['precision', 'recall', 'binary_accuracy'], 
        k = num_neighboors,
        matcher = MATCHER,
        verbose=1
    )

    matches = model.match(x_test,
                         cutpoint= 'optimal', 
                         k = num_neighboors,
                        matcher = MATCHER
                        )
    rows = []
    count_match = 0
    for idx, match in enumerate(matches):
        if match == y_test[idx] :
            count_match += 1
        
        rows.append([match, y_test[idx], match == y_test[idx]])
    print(tabulate(rows, headers=['Predicted', 'Expected', 'Correct']))
    print(f'Accuracy : {float(count_match)/len(rows) * 100}%')

    confusion_matrix(matches, y_test, labels=CLASSES, title='Confusin matrix for cutpoint: optimal' )
    #print("Matches :",matches)

    predict = model.lookup(x_test,k=1)

    rows = []
    count_match = 0
    thresholded_labels = []
    known_distances = []
    unknown_distances = []
    #Make lowest distance of unknown be threshold
    for idx, data in enumerate(predict):
        if y_test[idx] == len(CLASSES) -1 :
            unknown_distances.append(data[0].distance)
        else :
            known_distances.append(data[0].distance)

    threshold = min(unknown_distances)

    for idx, data in enumerate(predict):
        # data[0].label or distance
        if data[0].distance >= threshold :
            label = len(CLASSES) - 1
        else :
            label = data[0].label
        thresholded_labels.append(label)
        if label == y_test[idx] :
            count_match += 1
        
        rows.append([label, y_test[idx], label == y_test[idx]])

    print(tabulate(rows, headers=['Predicted', 'Expected', 'Correct']))
    print("")
    print(f'Accuracy after threshold ({threshold}): {float(count_match)/len(rows) * 100}%')
    print("MEAN known distance:",np.mean(known_distances))
    print("MEAN unknown distance:",np.mean(unknown_distances))
    print("STD distance:",np.std(known_distances))
    print("Max of known class distance :", max(known_distances))
    print("Lowest of unknown class distance :",min(unknown_distances))
    confusion_matrix(thresholded_labels, y_test, labels=CLASSES, title=f'Confusin matrix after threshold : {threshold}' )

    threshold = (min(unknown_distances) + max(known_distances))/2
    rows = []
    count_match = 0
    thresholded_labels = []
    for idx, data in enumerate(predict):
        # data[0].label or distance
        if data[0].distance >= threshold :
            label = len(CLASSES) - 1
        else :
            label = data[0].label
        thresholded_labels.append(label)
        if label == y_test[idx] :
            count_match += 1
        
        rows.append([label, y_test[idx], label == y_test[idx]])

    print(tabulate(rows, headers=['Predicted', 'Expected', 'Correct']))
    print(f'Accuracy after threshold ({threshold}): {float(count_match)/len(rows) * 100}%')
    confusion_matrix(thresholded_labels, y_test, labels=CLASSES, title=f'Confusin matrix after threshold : {threshold}' )
    

if __name__ == "__main__":
    main()
    # for model in ALL_MODEL:
    #     MODEL_PATH = model
    #     main()
