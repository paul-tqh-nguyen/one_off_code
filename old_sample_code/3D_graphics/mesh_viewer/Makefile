CXX=g++
CXXFLAGS= -fmax-errors=3 

all: CXXFLAGS += -O3 -funroll-loops -DBUILD_DEBUG=0 -I /usr/include/GL -std=c++11
debug: CXXFLAGS += -g -DBUILD_DEBUG=1 -std=c++11
all: LDFLAGS = -lglut -lGL -lGLU
debug: LDFLAGS = -lglut -lGL -lGLU

PROGS=main
SRCS=$(PROGS:=.cpp)

all: clean $(PROGS)
debug: clean $(PROGS)

$(PROGS): $(SRCS)
	$(CXX) -o $@ $(CXXFLAGS) $(SRCS) $(LDFLAGS)

clean:
	rm -f $(PROGS)

.PHONY: clean all debug
