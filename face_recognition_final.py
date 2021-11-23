import tensorflow as tf
import tensorflow_similarity as tfsim
import numpy as np
import time
import cv2 as cv
from mtcnn import MTCNN
from imutils.video import FPS
#from mtcnn_tflite.MTCNN import MTCNN
import os
import scipy.spatial

CUR_DIR = os.getcwd()

#Load model
face_detector = MTCNN()
model = tf.keras.models.load_model("face_model")

def main():
    command = list(map(str, input("Command : ").split()))
    
    #Sample
    if command[0] == "capture" :
        name = command[1]
        try:
            os.mkdir(CUR_DIR+"/database")
        except OSError:
            print("database Folder already exists. continue ...")
        capture_sample(name)
        print("Done sampling data.")

        face_recognition()

    #Facever
    if (command[0] == "face_verification") or (command[0] == "facever"):

        face_verification()

    #Facerec
    if ((command[0] == "face_recognition") or (command[0] == "facerec")) and len(command) == 2 :

        path = command[1]
        face_rec_vid(path)

    elif (command[0] == "face_recognition") or (command[0] == "facerec"):
        
        face_recognition()
    

def capture_sample(name:str,conf_t = 0.8,npair_sample:int = 10,down_sampling=2):
    try:
        os.mkdir(CUR_DIR+"/database/"+name)
    except OSError:
        print(name + " Folder already exists. continue...")
    os.chdir(CUR_DIR+"/database/"+name)

    down_sampling=2
    
    vc = cv.VideoCapture(0)
    vc.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    vc.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    vc_width = vc.get(cv.CAP_PROP_FRAME_WIDTH)
    vc_height = vc.get(cv.CAP_PROP_FRAME_HEIGHT)
    n = 0
    fps = FPS().start()
    while vc.isOpened():
        cur_frame = vc.get(1)
        ret, frame = vc.read()
        if not ret:
            print('Error')
            break
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_detector.detect_faces(cv.resize(frame_rgb,(int(vc_width/down_sampling),int(vc_height/down_sampling)),interpolation = cv.INTER_AREA))
        if len(results) == 2:
            
            conf1 = results[0]['confidence']
            conf2 = results[1]['confidence']

            if (conf1 > conf_t) and (conf2 > conf_t) and ((cur_frame % 2) == 0):  # 2 Frame interval
                face1 = crop_face(results[0]['box'], frame,down_sampling)
                face2 = crop_face(results[1]['box'], frame,down_sampling)
                cv.putText(frame,f"Capture a Sample", (30, 30), cv.FONT_ITALIC, 1, (0, 255, 0), 2)
                cv.imwrite(f"sample{n+1}_face.jpg", face1)
                cv.imwrite(f"sample{n+1}_ID.jpg", face2)
                print(f"Save sample{n+1}")
                n += 1

            for res in results:
                confidence = res['confidence']
                if confidence > conf_t:
                    bounding_box(confidence,res['box'], frame,down_sampling = down_sampling)
        elif len(results) > 0 :
            for res in results:
                confidence = res['confidence']
                if confidence > conf_t:
                    bounding_box(confidence,res['box'], frame,down_sampling = down_sampling)
        else:
            print("\rNo face detected", end='')

        fps.update()
        fps.stop()
        text = "FPS: {:.2f}".format(fps.fps())
        cv.putText(frame, text, (15, int(720 * 0.92)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.imshow('camera', put_mask(frame))
        if (cv.waitKey(1) & 0xFF == ord('q')) or (n == npair_sample):
            break
            vc.release()
            cv.destroyAllWindows()

    os.chdir(CUR_DIR)

def face_rec_vid(vid_path,conf_t = 0.95) :

    basename = os.path.basename(vid_path)
    class_names =get_classes_name(CUR_DIR+"/database")
    class_names.append("Unknown")
    print(class_names)
    #Model
    index_ds = tf.keras.preprocessing.image_dataset_from_directory(
                CUR_DIR+"/database",
                shuffle = True,
                labels='inferred',
                label_mode='int',
                image_size=(224,224),
                color_mode = 'rgb',
                batch_size=1)

    x_index,y_index = split_xy(index_ds)

    
    model.reset_index()
    model.index(x_index,y_index,data = x_index)
    

    print("Detect from",vid_path)
    vc = cv.VideoCapture(vid_path)
    frame_width = int(vc.get(3))
    frame_height = int(vc.get(4))
    fps = vc.get(cv.CAP_PROP_FPS)
    out = cv.VideoWriter(basename[:-4]+".avi", cv.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
    while vc.isOpened():
        ret, frame = vc.read()
        #cur_frame = vc.get(1)
        print("=", end="")
        if not ret:
            break
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_detector.detect_faces(frame_rgb)

        if len(results) == 2 :
            conf1 = results[0]['confidence']
            conf2 = results[1]['confidence']

            if (conf1 > conf_t) and (conf2 > conf_t) :

                face1 = cv.resize(crop_face(results[0]['box'], frame_rgb),(224,224),interpolation = cv.INTER_AREA)
                face2 = cv.resize(crop_face(results[1]['box'], frame_rgb),(224,224),interpolation = cv.INTER_AREA)

                label1 = find_face(model,class_names, face1)
                label2 = find_face(model,class_names, face2)

                if (label1 == label2) and ((label1 != class_names[-1]) or (label2 != class_names[-1])):
                    cv.putText(frame,f"SAME person", (30, 30), cv.FONT_ITALIC, 1, (0, 255, 0), 2)
                else :
                    cv.putText(frame,f"NOT same person", (30, 30), cv.FONT_ITALIC, 1, (0, 0, 255), 2)

                bounding_box(conf1,results[0]['box'], frame,label1)
                bounding_box(conf2,results[1]['box'], frame,label2)

        elif len(results) > 0 :
            for res in results:
                confidence = res['confidence']
                if confidence > conf_t:

                    face = cv.resize(crop_face(res['box'], frame_rgb),(224,224),interpolation = cv.INTER_AREA)
                    
                    label = find_face(model,class_names, face)
                    bounding_box(confidence,res['box'], frame,label)


        out.write(frame)
    print("\nDone processing")
    vc.release()
    out.release()

def face_recognition(conf_t = 0.7,down_sampling=2) :
    class_names =get_classes_name(CUR_DIR+"/database")
    class_names.append("Unknown")
    print(class_names)
    #Model
    index_ds = tf.keras.preprocessing.image_dataset_from_directory(
                CUR_DIR+"/database",
                shuffle = True,
                labels='inferred',
                label_mode='int',
                image_size=(224,224),
                color_mode = 'rgb',
                batch_size=1)

    x_index,y_index = split_xy(index_ds)

    
    model.reset_index()
    model.index(x_index,y_index,data = x_index)
    # calibration = model.calibrate( #calibrate the threshold from testset
    #     x_index, 
    #     y_index, 
    #     extra_metrics=['precision', 'recall', 'binary_accuracy'], 
    #     k = 1,
    #     matcher = "match_nearest",
    #     verbose=1
    # )

    #Camera
    
    vc = cv.VideoCapture(0)
    vc.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    vc.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    vc_width = vc.get(cv.CAP_PROP_FRAME_WIDTH )
    vc_height = vc.get(cv.CAP_PROP_FRAME_HEIGHT )
    fps = FPS().start()
    while vc.isOpened():
        #cur_frame = vc.get(1)
        ret, frame = vc.read()
        if not ret:
            print('Error')
            break
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_detector.detect_faces(cv.resize(frame_rgb,(int(vc_width/down_sampling),int(vc_height/down_sampling)),interpolation = cv.INTER_AREA))
        if len(results) == 2 :
            conf1 = results[0]['confidence']
            conf2 = results[1]['confidence']

            if (conf1 > conf_t) and (conf2 > conf_t) :

                face1 = cv.resize(crop_face(results[0]['box'], frame_rgb,down_sampling),(224,224),interpolation = cv.INTER_AREA)
                face2 = cv.resize(crop_face(results[1]['box'], frame_rgb,down_sampling),(224,224),interpolation = cv.INTER_AREA)

                label1 = find_face(model,class_names, face1)
                label2 = find_face(model,class_names, face2)

                if (label1 == label2) and ((label1 != class_names[-1]) or (label2 != class_names[-1])):
                    cv.putText(frame,f"SAME person", (int (vc_width * 0.5), int(vc_height * 0.92)), cv.FONT_ITALIC, 1, (0, 255, 0), 2)
                else :
                    cv.putText(frame,f"NOT same person", (int (vc_width * 0.5), int(vc_height * 0.92)), cv.FONT_ITALIC, 1, (0, 0, 255), 2)

                bounding_box(conf1,results[0]['box'], frame,label1,down_sampling = down_sampling)
                bounding_box(conf2,results[1]['box'], frame,label2,down_sampling = down_sampling)

        elif len(results) > 0 :
            for res in results:
                confidence = res['confidence']
                if confidence > conf_t:

                    face = cv.resize(crop_face(res['box'], frame_rgb,down_sampling),(224,224),interpolation = cv.INTER_AREA)
                    
                    label = find_face(model,class_names, face)
                    bounding_box(confidence,res['box'], frame,label,down_sampling = down_sampling)

                    # label = model.match(np.array([face]),
                    #         cutpoint= 'optimal', 
                    #         k = 1,
                    #         matcher = "match_nearest"
                    #         )
                    # bounding_box(res, frame,class_names[label[0]])
                    

        else : 
            print("\rNo face detected", end='')

        fps.update()
        fps.stop()
        text = "FPS: {:.2f}".format(fps.fps())
        cv.putText(frame, text, (15, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.imshow('camera', put_mask(frame))
        if (cv.waitKey(1) & 0xFF == ord('q')):
            break
            vc.release()
            cv.destroyAllWindows()

def face_verification(conf_t = 0.7,threshold = 0.35,down_sampling=2): #0.35? 0.33
    
    vc = cv.VideoCapture(0)
    vc.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    vc.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    while vc.isOpened():
        
        ret, frame = vc.read()
        if not ret:
            print('Error')
            break
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_detector.detect_faces(cv.resize(frame_rgb,(int(vc_width/down_sampling),int(vc_height/down_sampling)),interpolation = cv.INTER_AREA))
        if len(results) == 2 :
            conf1 = results[0]['confidence']
            conf2 = results[1]['confidence']

            if (conf1 > conf_t) and (conf2 > conf_t) :
                face1 = cv.resize(crop_face(results[0]['box'], frame_rgb,down_sampling),(224,224),interpolation = cv.INTER_AREA)
                face2 = cv.resize(crop_face(results[1]['box'], frame_rgb,down_sampling),(224,224),interpolation = cv.INTER_AREA)
                
                emb = model.predict(np.array([face1,face2]))
                
                distance = scipy.spatial.distance.cosine(emb[0],emb[1])
                
                if distance > threshold :
                    cv.putText(frame,f"NOT same ({distance:.4f})", (30, 30), cv.FONT_ITALIC, 1, (0, 0, 255), 2)
                    
                    bounding_box(conf1,results[0]['box'], frame,down_sampling = down_sampling)
                    bounding_box(conf2,results[1]['box'], frame,down_sampling = down_sampling)
                else :
                    cv.putText(frame,f"SAME ({distance:.4f})", (30, 30), cv.FONT_ITALIC, 1, (0, 255, 0), 2)
                    bounding_box(conf1,results[0]['box'], frame,down_sampling = down_sampling)
                    bounding_box(conf2,results[1]['box'], frame,down_sampling = down_sampling)

        elif len(results) > 0 :
            for res in results:
                confidence = res['confidence']
                if confidence > conf_t:
                    bounding_box(confidence,res['box'], frame,down_sampling = down_sampling)

        else : 
            print("\rNo face detected", end='')

        cv.imshow('camera', put_mask(frame))
        if (cv.waitKey(1) & 0xFF == ord('q')):
            break
            vc.release()
            cv.destroyAllWindows()

def find_face(model,classes,face,th = 0.0982):
    found = model.single_lookup(face,k = 1)
    #Find Nearest with distance threshold
    if found[0].distance < th :
        return classes[found[0].label]
    else :
        return classes[len(classes) - 1]

    
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

def bounding_box(confidence:float,topleftwidthheight:tuple, frame,text:str= None,conf:bool = False,keyp:bool = False ,down_sampling:int = 1):
    x1, y1, width, height = topleftwidthheight
    x1, y1 = abs(x1) * down_sampling, abs(y1) * down_sampling
    x2, y2 = x1 + (width * down_sampling), y1 + (height * down_sampling)
    
    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
    if conf :
        cv.putText(frame, f'conf: {confidence:.3f}',
                (x1, y1), cv.FONT_ITALIC, 1, (0, 0, 255), 1)
    if text :
        cv.putText(frame, text, (x2, y2), cv.FONT_ITALIC, 1, (0, 255, 0), 2)
    if keyp :
        key_points = res['keypoints'].values()
        for point in key_points:
            cv.circle(frame, point, 5, (0, 255, 0), thickness=-1)
    return frame

def crop_face(topleftwidthheight:tuple, frame,down_sampling:int = 1):
    x1, y1, width, height = topleftwidthheight
    x1, y1 = abs(x1) * down_sampling, abs(y1) * down_sampling
    x2, y2 = x1 + (width * down_sampling), y1 + (height* down_sampling)
    crop_img = frame[y1:y2, x1:x2]
    #print("Cropped a Face")
    return crop_img

def put_mask(frame):
    mask = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
    half_y = int(frame.shape[0]/2)
    half_x = int(frame.shape[1]/2)

    half_x_left = int(half_x/2)
    half_x_right = half_x + half_x_left

    cv.circle(mask, (half_x_left, half_y - int(half_y*0.3)),
              int(frame.shape[0]*0.3), (255, 255, 255), -1)
    cv.ellipse(mask, (half_x_left, frame.shape[0]), (int(
        half_x*0.4), int(frame.shape[0]*0.5)), 0, 180, 360, (255, 255, 255), -1)
    cv.rectangle(mask, (half_x, int(half_y*0.3)), (int(half_x +
                 frame.shape[1] * 0.4), half_y + int(half_y*0.3)), (255, 255, 255), -1)
    masked = cv.addWeighted(frame, 1, mask, 0.2, 0)
    return masked

def get_classes_name(path) :
    for i,y in enumerate(os.walk(path)):
        subdirs,dirs,files = y
        if i == 0 :
            return dirs

if __name__ == "__main__":
    main()
