GLPK_DIR = /users/student/mr111/lywu22/ALS/Final_Project/glpk-4.47/src
LEDA_DIR = /users/student/mr111/lywu22/ALS/Final_Project/LEDA

CXX = g++
CXXFLAGS = -std=c++11 -I$(GLPK_DIR) -I$(LEDA_DIR) -I$(LEDA_DIR)/incl 

TARGET = ../bin/main

SOURCES = $(wildcard *.cpp)

OBJECTS = $(SOURCES:.cpp=.o)

LDFLAGS = -L$(LEDA_DIR) -lG -lL -lm -L$(GLPK_DIR)/.libs -lglpk -L$(GLPK_DIR)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJECTS) $(TARGET)



BENCHMARKS_DIR = ../benchmarks
BENCHMARKS = $(wildcard $(BENCHMARKS_DIR)/*.kiss)

test: $(TARGET)
	@for benchmark in $(BENCHMARKS); do \
		echo "Running $$benchmark"; \
		time ./$(TARGET) $$benchmark; \
	done
