"""
Performs continuous object detection with the camera.

Simply run the script and it will draw boxes around detected objects along
with the predicted labels:

    python3 detect_objects.py

For more instructions, see g.co/aiy/maker
"""

from aiymakerkit import vision
#from aiymakerkit import utils
import models
from pycoral.utils.dataset import read_label_file

#detector = vision.Detector(models.OBJECT_DETECTION_MODEL)
detector = vision.Detector('models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')

#labels = utils.read_labels_from_metadata(models.OBJECT_DETECTION_MODEL)
labels = read_label_file('models/coco_labels.txt')


for frame in vision.get_frames(size=(640,480),capture_device_index=1):
    objects = detector.get_objects(frame, threshold=0.5)
    vision.draw_objects(frame, objects, labels)
