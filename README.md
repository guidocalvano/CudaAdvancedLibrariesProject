# MNIST classification with feedforward neural network

## Project Description
To pass the gpu programming specialization course from John Hopkins university on coursera I am doing this capstone 
project. My goal is to understand the correct use of modules in the CuDNN library that is used extensively in
machine learning projects I use on a daily basis.

The modules I intend to understand involve making a prediction given a model, and improving a prediction given training
data. To understand prediction I must first understand model construction. To understand training I must first 
understand prediction.

The simplest way to understand all three topics is to construct a model that can make predictions on the 
simplest data set, and then learn from its mistakes. I chose a model that contains the most important building blocks
of neural networks; A convolution layer, a dense layer and an activation function. 
A commonly used data set fitting this goal contains labeled handwritten numbers, and is called mnist. 

Because the focus is on understanding the use of these modules I will not spend time on good machine learning 
practices; I will not split the data set to improve generalization because it is not relevant to understanding use
of these models. I will merely measure if accuracy increases over during training, and output it to command line
for debugging purposes.

The proof is a set of images that show the predictions of the neural net. Accuracy is low (slightly over
80%), but enough to prove that backpropagation is implemented correctly. Each image file name has the index of the 
image in the trainingset, the predicted label and weather it is correct or wrong.

To install dependencies run:
```shell
make install
```
To download the test dataset run:

```shell
make data
```

To build the executable run:
```shell
make build
```
This produces ./bin/main.exe.

To run this executable using default arguments run:
```shell 
make run
```

To get help on running ./bin/main.exe with different arguments, just run it without extra arguments.

To clean the built executable run:
```shell 
make clean
```

To generate a zip of proof that my code ran correctly run:
```shell 
make proof
```

## Code Organization

```bin/```
This folder contains main.exe after running the following command:
```shell
make build
```

```data/```
After running the following command the data folder will contain textures in the folder ./data/textures.
```shell
make data
```
After running main.exe using the following command ./data/output will contain blurred texture files:
```shell
make run
```

```lib/```
There are no libraries. Everything is installed using sudo apt-get install.

```src/```
All source is in the src folder. It consists of a header file main.cuh and a code file main.cu.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
To install libopencv and libboost on ubuntu do:
```shell
make install
```

```Makefile or CMAkeLists.txt or build.sh```
Makefile contains the make rules.

```run.sh```
There is no run.sh. Just use main.exe in the bin folder.
