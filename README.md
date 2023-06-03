# ML-for-blood-glucose
Individual 3rd Year Project at UCL - Custom built neural network in Matlab which trains on photoplethysmography (PPG) data from a patient with Type-1 Diabetes in order to predict glucose levels on unseen PPG signals.

This project was inspired by previous research done on the clinical and medical uses of PPG (for example, measuring blood oxygen levles). The neural network uses stochastic gradient descent and employs a multilayer perceptron 5-3-3 model to train the data.

The PPG sensor used was the Easy Pulse Sensor by Embedded Labs, and more than 200 samples (5-second intervals of PPG data) were collected from a patient with Type-1 Diabetes during low, normal and high blood sugars in order to train the neural network in these categories. 

The data collected is not included in this repository since it contains sensitive medical data, therefore the code cannot be run without it. However, one can still use sample online data to test the accuracy of the custom built neural network.
