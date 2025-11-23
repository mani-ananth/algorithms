CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2
BOOST_INC ?= /opt/homebrew/opt/boost/include
GTEST_INC ?= /opt/homebrew/opt/googletest/include
GTEST_LIB ?= /opt/homebrew/opt/googletest/lib
TARGET := remove_duplicates
SOURCES := remove_duplicates.cpp

.PHONY: all run clean test

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -I$(BOOST_INC) -I$(GTEST_INC) -L$(GTEST_LIB) -lgtest -lgtest_main -pthread -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

