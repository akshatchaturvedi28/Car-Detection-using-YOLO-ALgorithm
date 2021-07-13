# Car-Detection-using-YOLO-ALgorithm
The first step in building a self-driving car system is to build a car detection system.

In this project, I have used the YOLO algorithm to build a Car Detection system.
YOLO (You only look once) is a popular algorithm because it gives a higher accuracy while being able to run in real-time. "Only looks once" means that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

* The input is a batch of images of shape (m, 608, 608, 3)
* The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers (pc, bx, by, bh, bw, c). Here c can be expanded into further 80 classes then each bounding box will be represented by 85 numbers. (Here pc is the probabilty that there's some object)

The YOLO architecture is like IMAGES (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85)
This means that for every image we get an encoding of shape (19, 19, 5, 85). For each of the pixel in 19x19 image we'll have 5 bounding boxes and respective 85 numbers for each box.

**In total the model will predict 19x19x5 = 1805 boxes just by making one forward pass through the image.**

This is astonishing number and a lot of boxes so to reduce this many boxes we'll follow these steps:
  1. Get rid of boxes with a low score (meaning, the box is not very confident about detecting a class)
  2. Select only one box when several boxes overlap with each other and detect the same object. (Non-max suppression)

To carry out the first step, we'll implement a yolo_filter_boxes() function.
  This function uses 4 parameters:
    1. box_confidence (19, 19, 5, 1) - pc for each box
    2. boxes (19, 19, 5, 4) - bx, by, bh, bw for each box
    3. box_class_probs (19, 19, 5, 80) - probability for each class
    4. threshold (float)
    
Based on the threshold we'll apply the mask on box_confidence x box_class_probs and hence remove the boxes with probability score less than threshold.

We have removed certain number of boxes after filtering by threshold but we're still left with many boxes so we'll now implement step 2.
### NON-MAX SUPPRESION
Non-max suppression uses IoU (Intersection over Union) method to remove overlapping boxes. We calculate the intersection area and the union area between 2 boxes. If IoU is greater than a threshold we remove the box with low probability score.

## Using YOLO pretrained model on Car Detection dataset

- The classes names are stored in coco_classes.txt file
- The 5 anchors are stored in yolo_anchors.txt file
- The shape of images in dataset is (720, 1280) that is pre-processed into (608, 608)

We are using pre-trained YOLO model which converts a batch of (m, 608, 608, 3) to a tensor of shape (m, 19, 19, 5, 85)

At last we'll run the Graph on our test images (available in '/images' folder) to visualize the bouding boxes on the images. We can also provide the images in a loop to get a video output.
