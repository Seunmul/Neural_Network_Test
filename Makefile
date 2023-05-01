# Makefile

RM=rm -f
CC=cc -O -Wall  -pg
CURL=curl
GZIP=gzip

LIBS=-lm

DATADIR=./data
MNIST_FILES= \
	$(DATADIR)/train-images-idx3-ubyte \
	$(DATADIR)/train-labels-idx1-ubyte \
	$(DATADIR)/t10k-images-idx3-ubyte \
	$(DATADIR)/t10k-labels-idx1-ubyte

all: test_bnn

clean:
	-$(RM) ./bnn ./mnist ./rnn *.o

get_mnist:
	-mkdir ./data
	-$(CURL) http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | \
		$(GZIP) -dc > ./data/train-images-idx3-ubyte
	-$(CURL) http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | \
		$(GZIP) -dc > ./data/train-labels-idx1-ubyte
	-$(CURL) http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | \
		$(GZIP) -dc > ./data/t10k-images-idx3-ubyte
	-$(CURL) http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | \
		$(GZIP) -dc > ./data/t10k-labels-idx1-ubyte

test_bnn: ./bnn
	./bnn
	
./bnn: bnn.c
	$(CC) -o $@ $^ $(LIBS)

valgrind_test_mnist:
	valgrind ./mem_test_mnist $(MNIST_FILES)

mem_test_bnn: bnn.o mem_check.o
	gcc -pg -o mem_test_bnn bnn.o mem_check.o -lm
	./mem_test_bnn
	gprof mem_test_bnn gmon.out > result_mem_test_bnn.txt

mem_test_rnn: rnn.o mem_check.o
	gcc -pg -o mem_test_rnn rnn.o mem_check.o -lm
	./mem_test_rnn
	gprof mem_test_rnn gmon.out > result_mem_test_rnn.txt

mem_test_mnist: mem_check.o cnn.o mnist.o
	gcc -pg -o mem_test_mnist mem_check.o cnn.o mnist.o -lm
	./mem_test_mnist $(MNIST_FILES)
	gprof mem_test_mnist gmon.out > result_mem_test_mnist.txt

mnist.o: cnn.h mem_check.h mnist.c
	gcc -pg -c -o mnist.o mnist.c

bnn.o: mem_check.h bnn.c
	gcc -pg -c -o bnn.o bnn.c

rnn.o: mem_check.h rnn.c
	gcc -pg -c -o rnn.o rnn.c

cnn.o: mem_check.h cnn.c
	gcc -pg -c -o cnn.o cnn.c

mem_check.o: mem_check.h mem_check.c
	gcc -pg -c -o mem_check.o mem_check.c
