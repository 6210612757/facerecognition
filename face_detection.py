import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import os
from os.path import dirname, join
import scipy.spatial
import time
from imutils.video import FPS


# cv2 and os load img as BGR
DATA_PATH = 'D:\Downloads/facednet/tobedetect'
DES_PATH = 'D:\Downloads/facednet/detected'
SAVE_DIR = os.getcwd() + '/face_capture'




def main():
    #lw_mtcnn()
    #detect_from_cam()
    #ssd_detect_from_cam()
    # os.chdir('D:/Downloads/facednet/face_capture')
    # img = cv.imread("sample1_face.jpg")
    # put_mask(img)
    # vid_path = 'D:\Downloads/facednet/test_data/test_face2.mp4'
    # detect_from_vid(vid_path, "test_face2_60.avi",n_sample=5,fps = 60,capture=True)
    
    detect_img_from_file(DATA_PATH,DES_PATH,conf_t=0.90)
    #detect_face_from_dir('D:\Downloads\FaceData\get-dataset-cn340\dataset','D:\Downloads\FaceData\get-dataset-cn340')
    
def ssd_detect_from_cam (conf_t = 0.5) :
    configFile = "ssd_model/deploy.prototxt.txt"
    modelFile = "ssd_model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    print("Loaded",modelFile)
    net = cv.dnn.readNetFromCaffe(configFile, modelFile)
    vc = cv.VideoCapture(0)
    vc.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    vc.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    fps = FPS().start()
    while vc.isOpened():
        ret, frame = vc.read()
        
        if not ret:
            print('Error')
            break
        origin_h, origin_w = frame.shape[:2]
        blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_t:
                bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
                x_start, y_start, x_end, y_end = bounding_box.astype('int')

                
                label = '{0:.2f}%'.format(confidence * 100)
                
                cv.rectangle(frame, (x_start, y_start), (x_end, y_end),(0, 0, 255), 2)
                
                #cv.rectangle(frame, (x_start, y_start-18), (x_end, y_start), (0, 0, 255), -1)
                
                cv.putText(frame, label, (x_start+2, y_start-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        fps.update()
        fps.stop()
        text = "FPS: {:.2f}".format(fps.fps())
        cv.putText(frame, text, (15, int(origin_h * 0.92)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.imshow('Frame', put_mask(frame))

        if cv.waitKey(1) & 0xFF == ord("q"):
            break


def detect_face_from_dir(directory,DES_path,conf_t = 0.80):
    face_detector = MTCNN()
    print("Directory :",directory)
    os.chdir(DES_path)
    new_path = DES_path +"/facedetected_folder"
    print("Save as :",new_path)
    face_found = 0
    face_notfound = 0
    total_images = 0
    total_vids = 0
    try:
        os.mkdir(new_path)
        pass
    except OSError:
        pass
    os.chdir(new_path)
    #Now in main dir
    all_dirs = []
    for subdir,dirs,files in os.walk(directory) :
        if dirs :
            all_dirs.append(dirs)
    print('all_dirs :',all_dirs)
    #Duplicate Directory
    main_dir = all_dirs[0]
    for dir in main_dir:
        try:
            os.mkdir(new_path+'/'+dir)
            pass
        except OSError:
            pass
        os.chdir(new_path+'/'+dir)
        for subdir in all_dirs[1:] :
            for folder in subdir :
                try:
                    os.mkdir(new_path+'/'+dir+'/'+folder)
                    pass
                except OSError:
                    pass
    os.chdir(new_path)

    #Reading files
    for i in range(len(main_dir)) :
        for n in all_dirs[1:][i] :
            for subdir,dirs,files in os.walk(directory+"/"+main_dir[i]+"/"+n) :
                print(subdir)
                os.chdir(new_path+"/"+main_dir[i]+"/"+n)
                for file in files:
                    print("Processing",file)
                    (base, ext) = os.path.splitext(file)
                    if ext in ('.jpg','.png','.jpeg'):
                        total_images += 1
                        img = cv.imread(os.path.join(subdir, file))
                        results = face_detector.detect_faces(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                        if len(results) > 0 :
                            for res in results :
                                if res['confidence'] > conf_t :
                                    face_found+=1
                                    face = capture_face(res,img)
                                    cv.imwrite(file,face)
                                else :
                                    print("Face not found (threshold).")
                        else :
                            face_notfound+=1
                            print("Face not found.")
                    elif ext in ('.mp4','.avi','.MOV','.mov') :
                        total_vids+=1
                        sample = 0
                        vc = cv.VideoCapture(os.path.join(subdir, file))
                        framerate = vc.get(cv.CAP_PROP_FPS)
                        print(f"Video framerate : {framerate}")
                        framecount = 0
                        while vc.isOpened():
                            ret, frame = vc.read()
                            if not ret:
                                break
                            if (framecount % int(framerate/2)) == 0 :
                                results = face_detector.detect_faces(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                                if len(results) > 0 :
                                    for res in results :
                                        if res['confidence'] > conf_t :
                                            sample+=1
                                            face_found+=1
                                            face = capture_face(res,frame)
                                            cv.imwrite((base+'_sample'+str(sample)+'.jpg'),face)
                                        else :
                                            print("Face not found (threshold).")
                                else :
                                    print("Face not found.")
                            framecount+=1
                        print(f"from vid : {file} found {sample} samples")

    print("Done processing.")
    print(f"From total : {total_images} images & {total_vids} videos")
    print(f"Found : {face_found} faces")
    print(f"Not Found : {face_notfound} images")


def detect_img_from_file(data_path, des_path, conf_t=0.9999):

    count_all = 0
    count_detected = 0
    count_undetected = 0
    face_detector = MTCNN()

    try:
        os.mkdir(des_path)
    except OSError:
        print("Folder already exists. continue ...")
    os.chdir(des_path)

    for i, filename in enumerate(os.listdir(data_path)):
        
        img = cv.imread(os.path.join(data_path, filename))

        if img is None:
            break

        count_all += 1
        print(f"Processing img {i+1} : {filename}", end=" => ")
        results = face_detector.detect_faces(
            cv.cvtColor(img, cv.COLOR_BGR2RGB))
        # print(filenames[i],results)

        if len(results) > 0:
            
            cond = 1  # If found
            for n, res in enumerate(results):
                x1, y1, width, height = res['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height

                confidence = res['confidence']
                if confidence > conf_t:
                    crop_img = img[y1:y2, x1:x2]
                    print("Detected", n+1)
                    temp_filename = filename.split(".")
                    cv.imwrite(temp_filename[0]+'_'+str(n)+"."+temp_filename[-1], crop_img)
                    count_detected += 1
                    cond = 0
            if cond:  # If found but lower than threshold
                print("NOT Detected (Lower than threshold)")
                count_undetected += 1
        else:
            print("NOT Detected")
            count_undetected += 1

    print("From {} Images".format(count_all))
    print("Detection Result : Detected {} Faces \n Undetected {} Images".format(
        count_detected, count_undetected))

def lw_mtcnn(conf_t = 0.7,down_sampling=2):
    face_detector = MTCNN()
    vc = cv.VideoCapture(0)
    vc.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    vc.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    vc_width = vc.get(cv.CAP_PROP_FRAME_WIDTH )
    vc_height = vc.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = FPS().start()

    while vc.isOpened():
        ret, frame = vc.read()
        if not ret:
            print('Error')
            break
        
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        #Down sampling 4 times
        results = face_detector.detect_faces(cv.resize(frame_rgb,(int(vc_width/down_sampling),int(vc_height/down_sampling)),interpolation = cv.INTER_AREA))

        if len(results) > 0:
            for res in results:
                confidence = res['confidence']
                if confidence > conf_t:
                    x1, y1, width, height = res['box']
                    x1, y1 = abs(x1) * down_sampling, abs(y1) * down_sampling
                    x2, y2 = x1 + (width * down_sampling), y1 + (height * down_sampling)
                    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
                    cv.putText(frame, f'conf: {confidence:.3f}',
                            (x1, y1), cv.FONT_ITALIC, 1, (0, 0, 255), 2)
        else :
            print("\rNo face detected", end='')

        fps.update()
        fps.stop()
        text = "FPS: {:.2f}".format(fps.fps())
        cv.putText(frame, text, (15, int(vc_height * 0.92)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.imshow('camera', put_mask(frame))
        if (cv.waitKey(1) & 0xFF == ord('q')) :
            break
            cv.destroyWindow('camera')
            vc.release()
        
        

def detect_from_cam(n_sample:int =5, conf_t=0.9 , capture:bool = False):  # 0.9 detect card at close half face
    face_detector = MTCNN()
    vc = cv.VideoCapture(0)
    vc.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    vc.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    vc_width = vc.get(cv.CAP_PROP_FRAME_WIDTH )
    vc_height = vc.get(cv.CAP_PROP_FRAME_HEIGHT )
    fps = FPS().start()
    
    n = 0
    try:
        os.mkdir(SAVE_DIR)
    except OSError:
        print("Folder already exists. continue ...")
    os.chdir(SAVE_DIR)
    while vc.isOpened():
        cur_frame = vc.get(1)
        ret, frame = vc.read()
        if not ret:
            print('Error')
            break
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_detector.detect_faces(frame_rgb)
        if capture == True :
            if len(results) == 2:

                conf1 = results[0]['confidence']
                conf2 = results[1]['confidence']

                if (conf1 > conf_t) and (conf2 > conf_t) and ((cur_frame % 3) == 0):  # 3 Frame interval
                    face1 = capture_face(results[0], frame)
                    face2 = capture_face(results[1], frame)
                    cv.imwrite(f"sample{n+1}_face.jpg", face1)
                    cv.imwrite(f"sample{n+1}_ID.jpg", face2)
                    print(f"Save sample{n+1}")
                    n += 1
                    # Try calculate distance
                    key1 = [i for i in results[0]['keypoints'].values()]
                    key2 = [i for i in results[1]['keypoints'].values()]
                    d = scipy.spatial.procrustes(key1, key2)[-1]
                    print("Distance :", d)

                for res in results:
                    confidence = res['confidence']
                    if confidence > conf_t:
                        bounding_box(res, frame)

            elif len(results) > 0:
                for res in results:
                    confidence = res['confidence']
                    if confidence > conf_t:
                        bounding_box(res, frame)

            else:
                print("\rNo face detected", end='')
        else :
            if len(results) > 0:
                for res in results:
                    confidence = res['confidence']
                    if confidence > conf_t:
                        bounding_box(res, frame)

            else:
                print("\rNo face detected", end='')
        fps.update()
        fps.stop()
        text = "FPS: {:.2f}".format(fps.fps())
        cv.putText(frame, text, (15, int(vc_height * 0.92)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.imshow('camera', put_mask(frame))
        if (cv.waitKey(1) & 0xFF == ord('q')) or (n == n_sample):
            break
            cv.destroyWindow('camera')
            vc.release()


def detect_from_vid(vid_path, saveas: str, conf_t=0.95,fps:int = 30, n_sample:int = 5,capture:bool = False):
    face_detector = MTCNN()
    vc = cv.VideoCapture(vid_path)
    frame_width = int(vc.get(3))
    frame_height = int(vc.get(4))
    n = 0
    try:
        os.mkdir(SAVE_DIR)
    except OSError:
        print("Folder already exists. continue ...")
    os.chdir(SAVE_DIR)
    print(f"Processing from {vid_path}")
    out = cv.VideoWriter(saveas, cv.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
    while vc.isOpened():
        ret, frame = vc.read()
        cur_frame = vc.get(1)
        print("*", end="")
        if not ret:
            break
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_detector.detect_faces(frame_rgb)
        if n == n_sample :
            capture = False
        if capture :
            if len(results) == 2:

                conf1 = results[0]['confidence']
                conf2 = results[1]['confidence']

                if (conf1 > conf_t) and (conf2 > conf_t) and ((cur_frame % fps) == 0):  # 3 Frame interval
                    face1 = capture_face(results[0], frame)
                    face2 = capture_face(results[1], frame)
                    cv.imwrite(f"sample{n+1}_face.jpg", face1)
                    cv.imwrite(f"sample{n+1}_ID.jpg", face2)
                    print(f"Save sample{n+1}")
                    n += 1

        for res in results:
            confidence = res['confidence']
            if confidence > conf_t:
                bounding_box(res, frame)

        out.write(frame)

    print("\nDone processing")
    vc.release()
    out.release()


def bounding_box(res, frame):
    confidence = res['confidence']
    x1, y1, width, height = res['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    key_points = res['keypoints'].values()
    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
    cv.putText(frame, f'conf: {confidence:.3f}',
               (x1, y1), cv.FONT_ITALIC, 1, (0, 0, 255), 1)
    for point in key_points:
        cv.circle(frame, point, 5, (0, 255, 0), thickness=-1)
    return frame


def capture_face(res, frame):
    x1, y1, width, height = res['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    crop_img = frame[y1:y2, x1:x2]
    print("Cropped a Face")
    return crop_img

# Load with huge memory usage


def load_images_from_folder(folder):
    print("Loading Images data ...")
    filenames = []
    images = []
    for filename in os.listdir(folder):
        print(f"Load {filename}", end='\r')
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            filenames.append(filename)
            images.append(img)
    print("Found {} Images".format(len(images)))

    return filenames, images


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


if __name__ == "__main__":
    main()
