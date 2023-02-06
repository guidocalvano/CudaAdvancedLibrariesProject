# IDIR=./
CXX = nvcc

#CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
#LDFLAGS += $(shell pkg-config --libs --static opencv)

.PHONY: all install data clean build run

all: data clean build run

#build:
#	$(CXX) ./src/main.cu --std c++17 `pkg-config opencv --cflags --libs` -o main.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -I./includes -lcuda

install:
	sudo apt-get install libopencv-dev
	sudo apt-get install libboost-all-dev

build:
	nvcc ./src/main.cu -o ./bin/main.exe -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu/ -lcudnn -lcublas -lboost_filesystem -lboost_system -I/usr/local/cuda/include `pkg-config --cflags --libs opencv4` -ccbin g++-7

	# $(CXX) ./src/main.cu s--std c++17 `pkg-config opencv --cflags --libs` -o main.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -I./includes -lcuda


run:
	./bin/main.exe "./data/input/" "./data/output/"

data:
	curl -s http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | gzip -d > data/input/training_image.idx3
	curl -s http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | gzip -d > data/input/training_label.idx1

proof:
	echo Not done yet

clean:
	rm -f ./bin/main.exe

x11ide:  # just handy for accessing my machine over x11
	tmux new-session -s ide "_JAVA_OPTIONS='-Dsun.java2d.xrender=false -Dsun.java2d.pmoffscreen=false' clion ."
