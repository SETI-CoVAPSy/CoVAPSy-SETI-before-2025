PYTHON = python3
CXXFLAGS = -O2 -shared -std=c++14 $(shell $(PYTHON) -m pybind11 --includes)

all: KMeans.so

cuda: kmeans.cu clean
	nvcc $(CXXFLAGS) --compiler-options -fPIC -I/usr/include/python3.10 -lpython3.10 -L/usr/lib/python3.10/config-3.10-aarch64-linux-gnu -DUSE_CUDA $< -o KMeans.so

KMeans.so: kmeans.cu
	hipcc $(CXXFLAGS) -fPIC -Wall $< -o $@

clean:
	rm -f KMeans.so KMeans.ptx kmeans.save
