# neuralPlateRecognition
This project is developed in Python using OpenCV and the free deep learning library `neural-networks-and-deep-learning`. It is capable of recognizing cars and reading the plate number.
It recognizes 37 different values: all characters from `A-Z`, digits from `0-9` and the european symbol of Spain on the plate.

Example:

![Plate recognition](https://i.imgur.com/lt24uc6.png)

## SET UP:
  Decompress the files in `testing_full_system/` and `training_ocr/` and the neural network library located in `src/neural-networks-and-deep-learning.zip`

## GENERAL DESCRIPTION:
  You can test the system executing main.py (See USAGE). The default neural net used is `general_v2` which is the one that gave the best results.
  If you want to train a new neural network, you only have to execute neural.py and follow the steps. 
    You can train it for a few epochs and then change parameters and keep training.
    You can even load an existing net and train keep training it if you wish.
  
  The network `general_v2` was trained with 100 input neurons, 100 first and 100 second level neurons and 37 output neurons.
    Used parameters:
    ```
      Epochs: 100
      Batch-Size: 10
      Learning rate: 0.1
      Lambda: 5.0
    ```
      
## USAGE:
  From `src/`:
  * Test a real-world scenario: 
  `python main.py "../testing_full_system/testing_full_system"`
  * Train a new neural network (new or existing):
  `python neural.py "../training_ocr/training_ocr/" "../training_ocr/testing_ocr/"`
