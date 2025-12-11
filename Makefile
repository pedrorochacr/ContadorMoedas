# Makefile para Contador de Moedas
# Compilação: make
# Execução: ./coin_counter <imagem>

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# OpenCV flags (use pkg-config)
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

# Target
TARGET = coin_counter
SRC = coin_counter.cpp
HEADER = coin_counter.h

# Default target
all: $(TARGET)

$(TARGET): $(SRC) $(HEADER)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $(SRC) $(OPENCV_LIBS)

# Clean
clean:
	rm -f $(TARGET) *.o resultado_*.jpg

# Run with test image
test: $(TARGET)
	./$(TARGET) moedas.jpg

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: clean $(TARGET)

# Help
help:
	@echo "Comandos disponíveis:"
	@echo "  make        - Compila o projeto"
	@echo "  make clean  - Remove arquivos gerados"
	@echo "  make test   - Executa com imagem de teste"
	@echo "  make debug  - Compila com símbolos de debug"

.PHONY: all clean test debug help