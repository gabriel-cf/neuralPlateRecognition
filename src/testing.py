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
from matplotlib import pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

print training_data[0]
print "###########################"
print training_data[1]


net = network.Network([784, 30, 10]) #784 input neurons, 30 hidden neurons, 10 output neurons
net.SGD(training_data, 30, 10, 3.0, test_data=test_data) #30 epochs, batch-size: 10, learning-rate: 3.0