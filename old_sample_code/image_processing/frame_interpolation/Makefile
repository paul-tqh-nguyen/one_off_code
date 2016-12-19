CXX=g++
CXXFLAGS= -fmax-errors=3 -fopenmp

all: CXXFLAGS += -O3 -funroll-loops -DBUILD_DEBUG=0 -std=c++0x 
debug: CXXFLAGS += -ggdb -DBUILD_DEBUG=1 -std=c++0x
all: LDFLAGS = -lpng -fopenmp
debug: LDFLAGS = -lpng -fopenmp

PROGS=main
SRCS=$(PROGS:=.cpp)

all: clean $(PROGS)
debug: clean $(PROGS)

$(PROGS): $(SRCS)
	$(CXX) -o $@ $(CXXFLAGS) $(SRCS) $(LDFLAGS)

clean:
	rm -f $(PROGS)

.PHONY: clean all debug
