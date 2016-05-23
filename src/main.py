from __future__ import division
from __future__ import print_function
import sys
sys.path.append('neural-networks-and-deep-learning/src')
from sys import argv, stderr
from os import listdir
from os.path import isdir, isfile, join
import cv2
import numpy as np
import network2
import copy
from matplotlib import pyplot as plt
from utils import *
from car import Car


######################################
#			   FUNCTIONS			 #
######################################


def getCarsFromImage(img, carClassifier):
	""" Receives the original image
		Returns a list of Rectangle objects containing the detected cars
	"""
	cars = carClassifier.detectMultiScale(img, 1.3, 2)
	return convertTupleListToRectangleList(cars)

def processImageForNeuralNet(arg1, image=False):
	""" 
	Receives as parameter arg1 the path of the image to be converted or the image already captured with
	cv2 (in that case, pass image=True as a parameter). The return of this function (x) should be passed as
	input to a Network object by network.feedforward(x)
	"""
	SIDE_SIZE = 10
	TOTAL_SIZE = 100
	img = arg1
	if(not image):
		img = cv2.imread(arg1,0)
	img = cv2.resize(img,(SIDE_SIZE,SIDE_SIZE))
	img = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	
	img = np.reshape(img, (TOTAL_SIZE, 1))
	return np.array(img, dtype='f')

def translateNeuralOutput(value):
	if(value < 10):
		return chr(value + 48)
	if(value == 36):
		return 'ESP'
	return chr(value + 55)

def processPlateText(car, net):
	s = ""
	if(len(car.rs) > 0):
		for r in car.rs: #First, we process the characters
			crop_img = car.plateImg[r.y:r.y+r.h, r.x:r.x+r.w]
			#showImage(crop_img)
			p_img = processImageForNeuralNet(crop_img, image=True)
			s = s + "{} ".format(translateNeuralOutput(np.argmax(net.feedforward(p_img))))
			#print(s)
	return s

######################################
#			    MAIN			 	 #
######################################

####### Param specification #######
USE = "Use: <Script Name> <Test Dir>"
if len(argv) < 2:
	printErrorMsg("Param number incorrect\n"+USE)
	exit(1)
vPath = argv[1]
if (not isdir(vPath)):
	printErrorMsg("'"+vPath+"'"+" is not a valid directory\n"+USE)
	exit(1)

vImages = loadImgs(vPath)

#### cv2.CascadeClassifier ####

# http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0

print ("\033[93mLoading CascadeClassifier files..\033[0m")
xml_carClassifier = "resources/coches.xml"
xml_plateClassifier = "resources/matriculas.xml"
carClassifier = cv2.CascadeClassifier(xml_carClassifier)
print ("\033[32mFile '{}' successfully loaded!\033[0m".format(xml_carClassifier))
plateCassifier = cv2.CascadeClassifier(xml_plateClassifier)
print ("\033[32mFile '{}' successfully loaded!\033[0m".format(xml_plateClassifier))
print ("\033[93mLoading Neural Network File..\033[0m")
neural_net_file = "resources/neural_net"
net = network2.load(neural_net_file)
print ("\033[32mFile '{}' successfully loaded!\033[0m".format(neural_net_file))

print ("\033[93mProcessing images..\033[0m")

for img in vImages:
	l_carsR = getCarsFromImage(img.img, carClassifier)
	for carR in l_carsR:
		car = Car(img.img, carR, plateCassifier)
		car.setPlateText(processPlateText(car, net))
		img.addCar(car)

#file = "testing_ocr.txt"
#f = open(file, 'w')

for img in vImages:
	for car in img.cars:
		car.draw()
		if(not car.isPlateEmpty()):
			print(car.plateText)
			#plateRX = car.plateR.x
			#plateRW = car.plateR.w
			#plateRY = car.plateR.y
			#plateRH = car.plateR.h
    		#f.write("{}\t{}\t{}\t{}\t{}\n".format(img.fileName, (plateRX + plateRX +plateRW) // 2, (plateRY + plateRY + plateRH) // 2, car.plateText, plateRW // 2))
		showImage(car.carImg)
	showImage(img.img)

f.close()

exit(0)
