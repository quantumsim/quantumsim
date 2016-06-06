NUMPY_INCLUDE=$(shell python3 -c 'import numpy; print(numpy.get_include());')
PYTHON_INCLUDE=$(shell python3 -c 'import distutils.sysconfig; print(distutils.sysconfig.get_python_inc());')
CFLAGS=-I$(NUMPY_INCLUDE) -g
CFLAGS+= -I$(PYTHON_INCLUDE)

all: dmcpu.so


dmcpu.c: dmcpu.pyx
	cython dmcpu.pyx

dmcpu.so: dmcpu.c
	gcc -o dmcpu.so -shared -fPIC dmcpu.c $(CFLAGS) 



