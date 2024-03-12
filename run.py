from Detector import *

modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
##modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz"
classFile="ob.txt"
imagepath="train/3.jpg"
videopath="train/tt.mp4"
threshold=0.5
detector=Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
##detector.predictImage(imagepath,threshold)
detector.predictVideo(videopath,threshold)
