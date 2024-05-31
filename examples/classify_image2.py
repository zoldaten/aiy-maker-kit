import contextlib
import select
import sys
import termios
import tty
from cv2 import imread
from pycoral.utils.dataset import read_label_file
from aiymakerkit import vision
#from aiymakerkit.utils import read_labels_from_metadata
import models
import time


def classify_image(classifier, labels, frame):    
    classes = classifier.get_classes(frame)
    label_id = classes[0].id
    score = classes[0].score
    label = labels.get(label_id)
    print(label, score)
    return classes


def main():    
    #parser.add_argument('-m', '--model', default='models/mobilenet_v2_1.0_224_quant_edgetpu.tflite',
    #parser.add_argument('-m', '--model', default='models/efficientdet_lite3_512_ptq_edgetpu.tflite',
    #parser.add_argument('-m', '--model', default='models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite',
                        #help='File path of .tflite file.')       

    classifier = vision.Classifier('models/mobilenet_v2_1.0_224_quant_edgetpu.tflite')
    labels = read_label_file('models/imagenet_labels.txt')
    
    frame = imread('obama.jpg')
    
    classify_image(classifier, labels, frame)
    start_time = time.time()
    classify_image(classifier, labels, frame)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    #start_time = time.time()    
    main()    
    #print("--- %s seconds ---" % (time.time() - start_time))


