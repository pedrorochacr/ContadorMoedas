# Contador de Moedas - Visão Computacional

Sistema automatizado para detectar, contar e classificar moedas brasileiras em imagens usando OpenCV e C++.

## Funcionalidades

- **Detecção automática** de moedas em imagens
- **Classificação** por tamanho (5, 10, 25, 50 centavos e 1 real)
- **Cálculo do valor total** monetário
- **Tratamento de ruído** com múltiplos filtros
- **Detecção de sobreposições** entre moedas
- **Suporte a diferentes condições de iluminação**

## Requisitos

- C++17 ou superior
- OpenCV 4.x (ou 3.x)
- CMake 3.10+ (opcional)

### Instalação do OpenCV (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install libopencv-dev
```

## Compilação

### Usando Make

```bash
make
```

### Usando CMake

```bash
mkdir build && cd build
cmake ..
make
```

## Uso

```bash
./coin_counter <caminho_da_imagem>
```

### Exemplo

```bash
./coin_counter moedas.jpg
```

## Pipeline de Processamento

O sistema segue o seguinte pipeline baseado nas etapas definidas no projeto:

1. **Conversão para escala de cinza** - Simplifica o processamento
2. **Filtragem** - Reduz ruído (Gaussian, Median ou Bilateral)
3. **Segmentação** - Separa objetos de interesse
4. **Detecção** - Encontra círculos via HoughCircles
5. **Classificação** - Identifica valores pelo tamanho

## Funções Principais

### Pré-processamento

| Função | Descrição |
|--------|-----------|
| `converterParaCinza()` | Converte imagem para escala de cinza |
| `aplicarFiltragem()` | Aplica filtro de redução de ruído |
| `segmentarImagem()` | Segmenta objetos na imagem |
| `equalizarHistograma()` | Melhora contraste |
| `aplicarCLAHE()` | Equalização adaptativa de histograma |

### Detecção

| Função | Descrição |
|--------|-----------|
| `detectarCirculos()` | Detecta círculos com HoughCircles |
| `detectarComParametros()` | Detecção com parâmetros customizados |
| `detectarSobreposicoes()` | Identifica moedas sobrepostas |

### Classificação

| Função | Descrição |
|--------|-----------|
| `classificarMoedaPorRaio()` | Classifica moeda pelo tamanho |
| `obterValorMoeda()` | Retorna valor monetário |
| `processarMoedas()` | Processa todas as moedas |

### Cálculos

| Função | Descrição |
|--------|-----------|
| `calcularValorTotal()` | Soma valores de todas as moedas |
| `contarPorDenominacao()` | Conta por tipo de moeda |

### Visualização

| Função | Descrição |
|--------|-----------|
| `desenharResultados()` | Anota moedas na imagem |
| `exibirResumo()` | Mostra resumo no console |

## Estruturas de Dados

```cpp
struct Moeda {
    Point2f centro;      // Coordenadas do centro
    float raio;          // Raio em pixels
    double valor;        // Valor monetário
    string denominacao;  // Nome (ex: "1 real")
};

struct ParametrosDeteccao {
    int tipoFiltro;      // Tipo de filtro
    double param1;       // Limiar Canny
    double param2;       // Limiar acumulador
    int minRaio;         // Raio mínimo
    int maxRaio;         // Raio máximo
    int minDist;         // Distância mínima
};
```

## Calibração

Para melhor precisão, ajuste os parâmetros em `classificarMoedaPorRaio()` conforme:

- Distância da câmera
- Resolução da imagem
- Moedas de referência conhecidas

### Diâmetros das Moedas Brasileiras

| Moeda | Diâmetro (mm) |
|-------|---------------|
| 10 centavos | 20 |
| 5 centavos | 22 |
| 50 centavos | 23 |
| 25 centavos | 25 |
| 1 real | 27 |

## Saída

O programa gera:

1. **Console**: Resumo com contagem e valor total
2. **resultado_processada.jpg**: Imagem após filtragem
3. **resultado_final.jpg**: Imagem com moedas anotadas

## Exemplo de Saída

```
========================================
      RESUMO DA CONTAGEM DE MOEDAS     
========================================

Quantidade por denominacao:
----------------------------------------
  1 real: 2 moeda(s)
  25 centavos: 1 moeda(s)
  50 centavos: 1 moeda(s)
----------------------------------------
Total de moedas: 4
Valor total: R$ 2.75
========================================
```

## Licença

Projeto acadêmico - Visão Computacional
