from __future__ import division
from sys import argv, stderr
import sys
sys.path.append('neural-networks-and-deep-learning/src')
from os import listdir
from os.path import isdir, isfile, join
import mnist_loader ##
import network
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt

#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#net = network.Network([784, 30, 10]) #784 input neurons, 30 hidden neurons, 10 output neurons
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data) #30 epochs, batch-size: 10, learning-rate: 3.0

def expandTrainingData(training_data):
    #f = gzip.open("../data/mnist.pkl.gz", 'rb')
    #training_data, validation_data, test_data = cPickle.load(f)
    #f.close()
    expanded_training_pairs = []
    j = 0 # counter
    for x, y in training_data:
        expanded_training_pairs.append((x, y))
        image = np.reshape(x, (-1, 10))
        j += 1
        if j % 1000 == 0: print("Expanding image number", j)
        # iterate over data telling us the details of how to
        # do the displacement
        for d, axis, index_position, index in [
                (1,  0, "first", 0),
                (-1, 0, "first", 9),
                (1,  1, "last",  0),
                (-1, 1, "last",  9)]:
            new_img = np.roll(image, d, axis)
            if index_position == "first": 
                new_img[index, :] = np.zeros(10)
            else: 
                new_img[:, index] = np.zeros(10)

            img = np.reshape(new_img, 100)
            print img
            expanded_training_pairs.append((img, y))
    random.shuffle(expanded_training_pairs)
    #expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]
    return expanded_training_pairs

#Returns tuples (x, y) where x = img in one vector of 100px and y = vector of 37 digits to represent the value
def loadImgs(path):
	def processImage(path):
		img = cv2.imread(path,0)
		img = cv2.resize(img,(10,10))
		img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		#cv2.imshow('cuadro',img)
		#cv2.waitKey()
		#cv2.destroyAllWindows()
		img = list(map((lambda x: x/1), img))
		return np.reshape(img, (100, 1))
	def translateASCIIvalue(c):
		v = ord(c)
		v = v - 55 if(v > 57) else v - 48 # ASCII '9' --> 57 (so, it is a character A-Z)
		return v
	def vectorizeValue(c):
		N_ELEMENTS = 37 #[0-9][A-Z]ESP
		result = np.zeros((N_ELEMENTS, 1))
		if(c == 'ESP'):
			result[N_ELEMENTS - 1] = 1.0
		else:
			v = translateASCIIvalue(c)
			#v = v - 55 if(v > 57) else v - 48 # ASCII '9' --> 57 (so, it is a character A-Z)
			result[v] = 1.0
		return result

	tImg = []
	i = 0

	#training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    #training_results = [vectorized_result(y) for y in tr_d[1]]
	imgs = []
	vs_t = [] #Vectorized
	vs_v = [] #Scalar
	for f in listdir(path):
		if (isfile(join(path, f)) and f.endswith('.jpg')):
			i+=1
			if(i > 500):
				break
			print "i: {0}".format(i)
			img = processImage(join(path, f))
			c = f.split('_')[0]
			v = vectorizeValue(c)
			#tImg.append((img, v)) #(V, Correct Value)
			imgs.append(img)
			vs_t.append(v)
			if(c == 'ESP'):
				vs_v.append(36)
			else:
				vs_v.append(translateASCIIvalue(c))
	tImg = zip(imgs,vs_t)
	vImg = zip(imgs, vs_v)
	return tImg, vImg

######################################
#			    MAIN			 	 #
######################################

####### Param specification #######
USE = "Use: <Script Name> <Training Dir> <Test Dir>"
if len(argv) < 3:
	printErrorMsg("Param number incorrect\n"+USE)
	exit(1)
tPath = argv[1]
vPath = argv[2]
if not isdir(tPath):
	printErrorMsg("'"+tPath+"'"+" is not a valid directory\n"+USE)
	exit(1)
if (not isdir(vPath)):
	printErrorMsg("'"+vPath+"'"+" is not a valid directory\n"+USE)
	exit(1)


training_data, test_data = loadImgs(tPath)
training_data = expandTrainingData(training_data)

#validation_data = loadImgs(vPath)
#validation_data = training_data

net = network.Network([100, 50, 30, 15, 37]) #100 input neurons, 14 hidden neurons, 37 output neurons
print "TRAINING BITCH"
net.SGD(training_data, 500, 10, 0.1, test_data=test_data) #30 epochs, batch-size: 10, learning-rate: 3.0

#i = 0
#values = []
#print "New net? -1 for exit, 0 for trying"
#while(True):
#	value = int(raw_input("lvl {0}:".format(i)))
#	if(value == -1):
#		break
#	if(value == 0):
#		epochs = int(raw_input("epochs: "))
#		batch = int(raw_input("batch-size: "))
#		eta = float(raw_input("learning rate: "))
#		net = network.Network(values)
#		net.SGD(training_data, epochs, batch, eta, test_data=test_data) #30 epochs, batch-size: 10, learning-rate: 3.0
#		values = []
#		i = 0
#	values.append(value)
#	i+=1

