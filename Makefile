# Makefile para Contador de Moedas com ML
# Requer OpenCV com módulo DNN

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# OpenCV flags
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

# Target
TARGET = coin_counter_ml
SRC = coin_counter_ml.cpp

# Diretórios
MODEL_DIR = models
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python3
PIP = $(VENV_DIR)/bin/pip

.PHONY: all clean train test help setup-venv

# Build
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(OPENCV_LIBS)
	@echo ""
	@echo "Build concluído: $(TARGET)"
	@echo "Para usar, primeiro treine o modelo com: make train DATASET=<pasta>"

# Cria ambiente virtual Python
setup-venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Criando ambiente virtual Python..."; \
		python3 -m venv $(VENV_DIR); \
		$(PIP) install --upgrade pip; \
		$(PIP) install -r requirements.txt; \
		echo ""; \
		echo "Ambiente virtual criado em $(VENV_DIR)/"; \
	else \
		echo "Ambiente virtual já existe em $(VENV_DIR)/"; \
	fi

# Treina o modelo
train: setup-venv
ifndef DATASET
	@echo "Erro: especifique o dataset"
	@echo "Uso: make train DATASET=<pasta_com_imagens>"
	@echo ""
	@echo "As imagens devem ter nomes como: <valor>_<id>.jpg"
	@echo "Exemplo: 50_123456.jpg (moeda de 50 centavos)"
else
	@mkdir -p $(MODEL_DIR)
	$(PYTHON) train_yolo.py --dataset $(DATASET) --export $(MODEL_DIR)
endif

# Testa com uma imagem
test: $(TARGET)
ifndef IMG
	@echo "Uso: make test IMG=<imagem.jpg>"
else
	./$(TARGET) $(IMG) --model $(MODEL_DIR)/moedas_classifier.onnx --classes $(MODEL_DIR)/classes.txt
endif

# Limpa
clean:
	rm -f $(TARGET) resultado.jpg

# Limpa tudo incluindo venv
clean-all: clean
	rm -rf $(VENV_DIR) $(MODEL_DIR) data runs

# Ajuda
help:
	@echo "Contador de Moedas com Machine Learning"
	@echo ""
	@echo "Comandos disponíveis:"
	@echo "  make              - Compila o programa C++"
	@echo "  make setup-venv   - Cria ambiente virtual Python"
	@echo "  make train DATASET=<pasta>  - Treina o modelo YOLOv8"
	@echo "  make test IMG=<img.jpg>     - Testa com uma imagem"
	@echo "  make clean        - Remove arquivos gerados"
	@echo "  make clean-all    - Remove tudo (incluindo venv)"
	@echo ""
	@echo "Fluxo de uso:"
	@echo "  1. Organize imagens com nomes: <valor>_<id>.jpg"
	@echo "  2. make train DATASET=minhas_moedas/"
	@echo "  3. make test IMG=foto_teste.jpg"