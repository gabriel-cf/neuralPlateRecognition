from sys import argv, stderr
import sys
sys.path.append('neural-networks-and-deep-learning/src')
from os import listdir
from os.path import isdir, isfile, join
import network2
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from utils import getInput, showImage

TOTAL_SIZE = 100
SIDE_SIZE = 10
N_ELEMENTS = 37 #[0-9][A-Z]ESP

def getNeuralNetFromUser():
	neural_net_file = "resources/neural_net" #JSON in a text file used to load the neural network
	net = None
	print "Load Neural Network from file?"
	value = getInput("-1 for training a new network, other key to load a trained one: ")
	if (value == '-1'):
		net_layers = [TOTAL_SIZE] #List of neurons, input layer == N pixels
		i =  1
		print "For each layer, insert the number of neurons\nInsert -1 to finish: "
		while(True):
			s_layer = "Layer {}: ".format(i)
			layer = int(getInput(s_layer))
			if(layer == -1):
				break
			net_layers.append(layer)
			i += 1
		net_layers.append(N_ELEMENTS) #Output layer == N possible output values
		net = network2.Network(net_layers, cost=network2.CrossEntropyCost)
		net.large_weight_initializer()
	else:
		value = getInput("-1 for specifying the neural network file. Other to load the default '{}': ".format(neural_net_file))
		if(value == '-1'):
			neural_net_file = getInput("Insert file name of the neural net to be loaded: ")
			while(True):
				if (isfile(neural_net_file)):
					break
				neural_net_file = getInput("Insert file name of the neural net to be loaded: ")
				print "Name invalid, please try again"
		net = network2.load(neural_net_file) #Loads an existing neural network from a file
	return net

def processUserNeuralSettings():
	epochs = int(getInput("epochs: "))
	batch = int(getInput("batch-size: "))
	eta = float(getInput("learning rate (< 1): "))
	lmbda = float(getInput("lambda: "))

	return epochs, batch, eta, lmbda

def expandTrainingData(training_data):
	""" Takes every image on the training_data set and generates 4 additional images 
		by displacing each one pixel up, down, left and right

		Note: the expanded_training_pairs contains the original training_data set
	"""
	expanded_training_pairs = []
	C = 1#/float(1)-0.1
	j = 0 # counter
	for x, y in training_data:
		#expanded_training_pairs.append((np.reshape(x, (TOTAL_SIZE, 1)), y))
		expanded_training_pairs.append((x,y))
		image = np.reshape(x, (-1, SIDE_SIZE))

		j += 1
		if j % 1000 == 0: print("Expanding image number", j)
		# iterate over data telling us the details of how to
		# do the displacement
		for d, axis, index_position, index in [
				(1,  0, "first", 0),
				(-1, 0, "first", SIDE_SIZE - 1),
				(1,  1, "last",  0),
				(-1, 1, "last",  SIDE_SIZE - 1)]:
			new_img = np.roll(image, d, axis)
			p = np.empty(SIDE_SIZE)
			p.fill(C)
			if index_position == "first": 
				p = np.empty(SIDE_SIZE)
				p.fill(C)
				new_img[index, :] = p
			else: 
				new_img[:, index] = p
			img = np.reshape(new_img, TOTAL_SIZE)
			img = np.reshape(new_img, (TOTAL_SIZE, 1))
			expanded_training_pairs.append((img, y))
	random.shuffle(expanded_training_pairs)
	return expanded_training_pairs

def processImage(arg1, image=False):
	""" Takes an image path or a preprocessed image (if image=True) and converts it to binary threshold
		and reshapes it into a TOTAL_SIZE-D x 1 vector
	"""
	img = arg1
	if(not image):
		img = cv2.imread(arg1,0)
	img = cv2.GaussianBlur(img, (5,5), 0)
	img = cv2.resize(img,(SIDE_SIZE,SIDE_SIZE))
	img = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	#img = list(map((lambda x: x/float(1)-0.1), img))
	return np.reshape(img, (TOTAL_SIZE, 1))

def loadImgs(tPath, vPath = None):
	""" Returns a list of tuples representing the image and the value necessary for feeding the neural network
		* If vPath is specified, returns its list of tuples too.
	"""
	def translateASCIIvalue(c):
		""" Returns the int position of a char or the string 'ESP' in a 37-D vector """
		if(c == 'ESP'):
			return 36
		v = ord(c)
		v = v - 55 if(v > 57) else v - 48 #Based on ASCII values
		return v
	def vectorizeValue(c):
		""" Returns a 37-D Vector with a value of 1.0 in the position associated to the char 'c' """
		result = np.zeros((N_ELEMENTS, 1))
		v = translateASCIIvalue(c)
		result[v] = 1.0
		return result
	def loadImgsAux(path, func):
		""" Processes all images and return a list of tuples (img --> TOTAL_SIZE-D x 1, value --> return of 'func') """
		imgs = []
		vs = []
		i = 0
		#Load images
		for f in listdir(path):
			if (isfile(join(path, f)) and f.endswith('.jpg')):
				i+=1
				if i % 1000 == 0: print("Adding image number", i)
				img = processImage(join(path, f))
				c = f.split('_')[0]
				v = func(c)
				imgs.append(img)
				vs.append(v)
		return zip(np.array(imgs, dtype='f'), vs)

	#Load training images
	tImg = loadImgsAux(tPath, vectorizeValue) #Tuples (img --> TOTAL_SIZE-D x 1, value --> N_ELEMENTS-D x 1)
	if(vPath is None):
		return tImg

	#Load testing images
	vImg = loadImgsAux(vPath, translateASCIIvalue) #Tuples (img --> TOTAL_SIZE-D x 1, value --> int)
	return tImg, vImg

######################################
#			    MAIN			 	 #
######################################

####### Param specification #######
USE = "Use: <Script Name> <Training Dir> [Test Dir]"
if len(argv) < 2:
	printErrorMsg("Param number incorrect\n"+USE)
	exit(1)
tPath = argv[1]
vPath = None
if not isdir(tPath):
	printErrorMsg("'"+tPath+"'"+" is not a valid directory\n"+USE)
	exit(1)
if (argv[2]):
	if (not isdir(argv[2])):
		printErrorMsg("'"+argv[2]+"'"+" is not a valid directory\n"+USE)
		exit(1)
	vPath = argv[2]

training_data, test_data = loadImgs(tPath, vPath)
value = getInput("-1 to expand training data, skip otherwise: ")
if (value == '-1'):
	training_data = expandTrainingData(training_data)
	
epochs = 20
batch = 10
eta = 0.1
lmbda = 0.1
print "\nDefinition of the neural network"
print "################################"
net = getNeuralNetFromUser() #Allows to load an existing network or creating a new network
value = getInput("-1 to manually specify parameters, other key for default: ")
if (value == '-1'):
	epochs, batch, eta, lmbda = processUserNeuralSettings()

while(True):
	print "TRAINING NEURAL NET"
	print "###################"
	net.SGD(training_data, epochs, batch, eta,evaluation_data = test_data, lmbda = lmbda, monitor_training_accuracy=True, monitor_evaluation_accuracy=(vPath != None))
	print "Save neural net?"
	value = getInput("-1 for not saving, other key for saving: ")
	if(value != '-1'):
		net.save(getInput("Name: "))
	print "Keep training?"
	value = getInput("-1 for exit, other key for trying: ")
	if(value == '-1'):
		break
	value = getInput("-1 to manually specify parameters, other key for default: ")
	if (value == '-1'):
		epochs, batch, eta, lmbda = processUserNeuralSettings()

exit(0)
