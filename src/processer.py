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


######################################
#			   FUNCTIONS			 #
######################################


def getCarsFromImage(img, carClassifier):
	""" Receives the original image
		Returns a list of Rectangle objects containing the detected cars
	"""
	cars = carClassifier.detectMultiScale(img, 1.3, 2)
	return convertTupleListToRectangleList(cars)
	
def getBestPlate(carImg, plateCassifier):
	""" Receives the car cropped image 
		Returns the best plate and a list of rectangles containing the characters
	"""
	plates = plateCassifier.detectMultiScale(carImg, 1.3, 2) #Returns an array of rectangles (x,y,w,h) delimiting the plates
	l_plates = convertTupleListToRectangleList(plates)
	
	max_rs = []
	max_plateImg = None
	max_plate = None
	for plate in l_plates:
		plateImg = carImg[plate.y:plate.y+plate.h, plate.x:plate.x+plate.w]
		plateImg = cv2.GaussianBlur(plateImg,(5,5),0) # Helps to detect only large borders
		thres = cv2.adaptiveThreshold(plateImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		
		contours, _ = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		rs = []
		for cnt in contours:
			x,y,w,h = cv2.boundingRect(cnt)
			r = Rectangle(x,y,w,h)
			rs.append(r)

		rs = filterBorderRectangles(rs, plate.h)
		if(len(rs) > len(max_rs)):
			max_rs = rs
			max_plateImg = plateImg
			max_plate = plate
	
	return max_plateImg, max_plate, max_rs

def filterBorderRectangles(rs, h, alfa = 1):
	""" Receives a list of Rectangle objects surrounding borders
		Alfa parameter can be specified to adjust the strictness of the filter
		Returns a filtered list of the eight or less most similar Rectangle objects
	"""
	def pythagoras(r):
		return np.sqrt(r.h*r.h + r.w*r.w)
	def inRange(y, r):
		""" Determines wether a rectangle is crossed by a horizontal line 'y' or not 
		"""
		ini = r.y
		fini = r.y + r.h
		return (y > ini and y <= ini + fini)
	def getRatio(rs):
		return rs.h / rs.w
	def diffRatio(r1, r2):
		return np.abs(getRatio(r1) - getRatio(r2))/getRatio(r1)
	def diffHypotenuse(r1, r2):
		py1 = pythagoras(r1)
		py2 = pythagoras(r2)
		return np.abs(py1 - py2)/py1
	def chunkByCriteria(rs, limit, func):
		"""
		Divides rs in chunks of rectangles based on the criteria of the function func and a given limit
			rs --> original list of rectangles
			limit --> % of difference for the values of two consecutive rectangles
			func --> name of the function to be used for calculating the difference %. Must need 2 rectangles (r1, r2)
		"""
		chunk = []
		best_chunk = []
		for i in xrange(0,len(rs)):
			chunk.append(rs[i]) #We append and cut this chunk if the next rectangle is different
			if(i < len(rs) - 1):
				percentage = func(rs[i], rs[i + 1])
				if percentage > limit:
					if (len(best_chunk) < len(chunk)):
						best_chunk = chunk					
					chunk = []
			else:
				if (len(best_chunk) < len(chunk)):
					best_chunk = chunk
		return best_chunk
	
	#print("Alfa: {}".format(alfa))
	mid = h // 2
	MIN_H = 10 #Minimum height for a rectangle containing a digit
	ratioLimit = 1.2 * alfa
	hypotenuseLimit = 0.4 * alfa
	
	# I: Filter values that aren't crossed by the middle line or are too small (< MIN_H)
	rs = filter(lambda x: inRange(mid, x) and (x.h > MIN_H), rs)
	# II: Sort rectangles by ratio and divide in chunks of similar ratios
	rs.sort(key=lambda x: getRatio(x))
	chunk = chunkByCriteria(rs, ratioLimit, diffRatio)
	# III: Sort chunks by hypotenuse (size) and filter
	chunk.sort(key=lambda x: pythagoras(x))
	chunk = chunkByCriteria(chunk, hypotenuseLimit, diffHypotenuse)
	# IV - Experimental: if > 8 rectangles, repeat process with a stricter limit
	if(len(chunk) > 8):
		new_chunk = filterBorderRectangles(chunk, h, alfa = alfa - 0.1)
		if(len(new_chunk) > 0):
			return new_chunk
	chunk.sort(key=lambda r: r.x)
	return chunk

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
	#showImage(img)
	img = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	#print (img)
	
	#img = list(map((lambda x: x/float(1)-0.1), img))
	img = np.reshape(img, (TOTAL_SIZE, 1))
	return np.array(img, dtype='f')

def translateNeuralOutput(value):
	if(value < 10):
		return chr(value + 48)
	if(value == 36):
		return 'ESP'
	return chr(value + 55)

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
#carClassifier = cv2.CascadeClassifier(raw_input("XML Car Classifier:"))
#plateCassifier = cv2.CascadeClassifier(raw_input("XML Plate Classifier:"))
#neural_net_file = raw_input("Neural Net File: ")
xml_carClassifier = "coches.xml"
xml_plateClassifier = "matriculas.xml"
carClassifier = cv2.CascadeClassifier(xml_carClassifier)
print ("\033[32mFile '{}' successfully loaded!\033[0m".format(xml_carClassifier))
plateCassifier = cv2.CascadeClassifier(xml_plateClassifier)
print ("\033[32mFile '{}' successfully loaded!\033[0m".format(xml_plateClassifier))
print ("\033[93mLoading Neural Network File..\033[0m")
#neural_net_file = "trained_neural_net_V6" #In the root dir
neural_net_file = "trained_nets/general_net_V2" #In the root dir
net = network2.load(neural_net_file)
print ("\033[32mFile '{}' successfully loaded!\033[0m".format(neural_net_file))

print ("\033[93mProcessing images..\033[0m")
for img in vImages:	
	l_cars = getCarsFromImage(img.img, carClassifier)
	for car in l_cars:
		carImg = img.img[car.y:car.y+car.h, car.x:car.x+car.w]
		max_plateImg, max_plate, max_rs = getBestPlate(carImg, plateCassifier)
		if(max_plateImg is not None):
			for r in max_rs: #First, we process the characters
				crop_img = max_plateImg[r.y:r.y+r.h, r.x:r.x+r.w]
				showImage(crop_img)
				p_img = processImageForNeuralNet(crop_img, image=True)
				print("{} ".format(translateNeuralOutput(np.argmax(net.feedforward(p_img)))), end='') #Neural Network Output
			print()
			for r in max_rs: #Then, we print the rectangles on the image (this is to avoid overlapping rectangles)
				cv2.rectangle(max_plateImg,(r.x,r.y),(r.x+r.w,r.y+r.h),(255,0,0),1)
			cv2.rectangle(carImg,(max_plate.x,max_plate.y),(max_plate.x+max_plate.w,max_plate.y+max_plate.h),(255,0,0),2)
			showImage(max_plateImg)
		cv2.rectangle(img.img,(car.x,car.y),(car.x+car.w,car.y+car.h),(0,0,255),2) #Print the car rectangle
		
	showImage(img.img) #Visualize the final picture

exit(0)
