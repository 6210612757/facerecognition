# Face recognition & verification with ID card

## Face detection
We implemented MTCNN (<https://github.com/ipazc/mtcnn>)

The state of art face detection "MTCNN" perform well in detects faces and face in the ID card.

The face detection perform result in average 1.04 FPS for resolution 1280x720.
Sometimes the FPS is dropped below 1 FPS ( Not in realtime detect )

![old fps](/images/oldFPS.png)

We proposed downsampling the input of MTCNN.

from resolution 1280x720 to 640x360 (half resolution). Then map the bounding boxes back to the original resolution by multiply by 2.

Result : increased FPS by 30%

Average 1.34 FPS able to perform in realtime all the time.

![old fps](/images/newFPS.png)

## Face recognition
We used Tensorflow Similarity library. (<https://github.com/tensorflow/similarity>)
Required Python 3.7

We choosed EfficientNet B0 (<https://arxiv.org/abs/1905.11946>) as model architecture.

Because B0 fixed input lowest resolution (224x224) and our job need to deal with low resolution images (face in the ID card).

Augmentation performed : rotate, flip horizontal, brightness, hue
(Mix and match all augmentations)

Embedding size : 512 (Equal with FaceNet512)

Test data with 28 known faces and unknown faces.

167 images consist of 74 known face images and 93 unknown face images.

![confusion](/images/confusion_matrix_lastest.jpg)

Result in : 98.20% Accuracy to classify person with 0.0982 threshold to unknown.

Evaluate model ability to verify faces with FAR, FRR, EER

![EER](/images/FAR_FRR_EER.jpg)

Result in : EER is around (17,10) the lower value in (x,y) the better performance in face verification

Demo :

![demo](/images/6110613129-face_id.gif)
