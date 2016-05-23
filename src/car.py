from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
from utils import *
import copy

class Car(object):
	"""docstring for Car""" #666
	def __init__(self, img, carR, plateCassifier):
		self.plateCassifier = plateCassifier
		self.img = img #Original image
		self.carR = carR #Rectangle delimiting the car on the Image
		self.carImg = img[carR.y:carR.y+carR.h, carR.x:carR.x+carR.w]
		self.plateImg, self.plateR, self.rs = self.getBestPlate()
		self.plateText = ''
	
	def setPlateText(self, s):
		self.plateText = s

	def isPlateEmpty(self):
		return self.plateText == ''

	def draw(self):
		if(self.carR is not None):
			cv2.rectangle(self.img,(self.carR.x,self.carR.y),(self.carR.x+self.carR.w,self.carR.y+self.carR.h),(0,0,255),2) #Print the car rectangle
			
			if(self.plateR is not None):
				cv2.rectangle(self.carImg,(self.plateR.x,self.plateR.y),(self.plateR.x+self.plateR.w,self.plateR.y+self.plateR.h),(255,0,0),2)
				for r in self.rs:
					cv2.rectangle(self.plateImg,(r.x,r.y),(r.x+r.w,r.y+r.h),(255,0,0),1)

	def getBestPlate(self):
		""" Receives the car cropped image 
			Returns the best plate and a list of rectangles containing the characters
		"""
		plates = self.plateCassifier.detectMultiScale(self.carImg, 1.3, 2) #Returns an array of rectangles (x,y,w,h) delimiting the plates
		l_plates = convertTupleListToRectangleList(plates)
		
		max_rs = []
		max_plateImg = None
		max_plate = None
		for plate in l_plates:
			plateImg = self.carImg[plate.y:plate.y+plate.h, plate.x:plate.x+plate.w]
			plateImgFilter = copy.copy(plateImg)
			plateImgFilter = cv2.GaussianBlur(plateImgFilter,(5,5),0) # Helps to detect only large borders
			thres = cv2.adaptiveThreshold(plateImgFilter,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
			contours, _ = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
			rs = []
			for cnt in contours:
				x,y,w,h = cv2.boundingRect(cnt)
				r = Rectangle(x,y,w,h)
				rs.append(r)

			rs = self.filterBorderRectangles(rs, plate.h)
			if(len(rs) > len(max_rs)):
				max_rs = rs
				max_plateImg = plateImg
				max_plate = plate
		
		return max_plateImg, max_plate, max_rs

	def filterBorderRectangles(self, chunk, h, alfa = 1):
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
		
		MID = h // 2
		MIN_H = h // 3 #Minimum height for a rectangle containing a digit

		while len(chunk) > 8:
			#ratioLimit = 1.2 * alfa
			#hypotenuseLimit = 0.25 #* alfa
			ratioLimit = 2.0 * alfa
			hypotenuseLimit = 0.5 * alfa

			# I: Filter values that aren't crossed by the middle line or are too small (< MIN_H)
			chunk = filter(lambda x: inRange(MID, x) and (x.h > MIN_H), chunk)

			# II: Sort rectangles by ratio and divide in chunks of similar ratios
			chunk.sort(key=lambda x: getRatio(x))
			chunk = chunkByCriteria(chunk, ratioLimit, diffRatio)

			# III: Sort chunks by hypotenuse (size) and filter
			chunk.sort(key=lambda x: pythagoras(x))
			chunk = chunkByCriteria(chunk, hypotenuseLimit, diffHypotenuse)

			alfa = alfa - 0.05

		chunk.sort(key=lambda r: r.x)
		return chunk