#ifndef COIN_COUNTER_H
#define COIN_COUNTER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <string>

using namespace cv;
using namespace std;

// ============================================================================
// ESTRUTURAS DE DADOS
// ============================================================================

/**
 * @brief Estrutura para armazenar informações de uma moeda detectada
 */
struct Moeda {
    Point2f centro;      ///< Coordenadas do centro da moeda
    float raio;          ///< Raio da moeda em pixels
    double valor;        ///< Valor monetário da moeda
    string denominacao;  ///< Nome da denominação (ex: "1 real")
};

/**
 * @brief Enum para tipos de moedas brasileiras
 */
enum class TipoMoeda {
    CINCO_CENTAVOS,
    DEZ_CENTAVOS,
    VINTE_CINCO_CENTAVOS,
    CINQUENTA_CENTAVOS,
    UM_REAL,
    DESCONHECIDA
};

/**
 * @brief Estrutura para parâmetros de detecção configuráveis
 */
struct ParametrosDeteccao {
    int tipoFiltro = 0;     ///< Tipo de filtro (0=Gaussian, 1=Median, 2=Bilateral)
    double param1 = 100;    ///< Limiar superior para Canny
    double param2 = 30;     ///< Limiar do acumulador Hough
    int minRaio = 20;       ///< Raio mínimo em pixels
    int maxRaio = 150;      ///< Raio máximo em pixels
    int minDist = 50;       ///< Distância mínima entre centros
};

/**
 * @brief Estrutura para resultado completo da detecção
 */
struct ResultadoDeteccao {
    vector<Moeda> moedas;       ///< Lista de moedas detectadas
    double valorTotal;          ///< Valor monetário total
    int quantidadeTotal;        ///< Quantidade total de moedas
    Mat imagemProcessada;       ///< Imagem após processamento
    Mat imagemResultado;        ///< Imagem com anotações
};

// ============================================================================
// FUNÇÕES DE PRÉ-PROCESSAMENTO
// ============================================================================

/**
 * @brief Converte imagem para escala de cinza
 * @param imagemOriginal Imagem de entrada (BGR ou BGRA)
 * @return Imagem em escala de cinza
 */
Mat converterParaCinza(const Mat& imagemOriginal);

/**
 * @brief Aplica filtro para redução de ruído
 * @param imagemCinza Imagem em escala de cinza
 * @param tipoFiltro 0=Gaussian, 1=Median, 2=Bilateral
 * @return Imagem filtrada
 */
Mat aplicarFiltragem(const Mat& imagemCinza, int tipoFiltro = 0);

/**
 * @brief Segmenta a imagem para separar objetos
 * @param imagemFiltrada Imagem filtrada
 * @param metodo 0=Adaptativo, 1=Otsu, 2=Canny
 * @return Imagem segmentada
 */
Mat segmentarImagem(const Mat& imagemFiltrada, int metodo = 0);

/**
 * @brief Equaliza histograma para melhorar contraste
 * @param imagemCinza Imagem em escala de cinza
 * @return Imagem equalizada
 */
Mat equalizarHistograma(const Mat& imagemCinza);

/**
 * @brief Aplica CLAHE para equalização adaptativa
 * @param imagemCinza Imagem em escala de cinza
 * @param clipLimit Limite de contraste
 * @param tileSize Tamanho do tile
 * @return Imagem com CLAHE aplicado
 */
Mat aplicarCLAHE(const Mat& imagemCinza, double clipLimit = 2.0, 
                 Size tileSize = Size(8, 8));

// ============================================================================
// FUNÇÕES DE DETECÇÃO
// ============================================================================

/**
 * @brief Detecta círculos usando Transformada de Hough
 * @param imagemFiltrada Imagem pré-processada
 * @param minRaio Raio mínimo
 * @param maxRaio Raio máximo
 * @param param1 Limiar Canny
 * @param param2 Limiar acumulador
 * @param minDist Distância mínima entre centros
 * @return Vetor de círculos (x, y, raio)
 */
vector<Vec3f> detectarCirculos(const Mat& imagemFiltrada, 
                               int minRaio = 20, 
                               int maxRaio = 150,
                               double param1 = 100,
                               double param2 = 30,
                               int minDist = 50);

/**
 * @brief Detecta círculos com parâmetros customizados
 * @param imagemFiltrada Imagem pré-processada
 * @param params Estrutura com parâmetros
 * @return Vetor de círculos
 */
vector<Vec3f> detectarComParametros(const Mat& imagemFiltrada, 
                                     const ParametrosDeteccao& params);

// ============================================================================
// FUNÇÕES DE CLASSIFICAÇÃO
// ============================================================================

/**
 * @brief Classifica tipo de moeda pelo raio
 * @param raio Raio da moeda em pixels
 * @param raioReferencia Raio de referência para calibração
 * @return Tipo da moeda
 */
TipoMoeda classificarMoedaPorRaio(float raio, float raioReferencia = 0);

/**
 * @brief Obtém valor monetário do tipo de moeda
 * @param tipo Tipo da moeda
 * @return Valor em reais
 */
double obterValorMoeda(TipoMoeda tipo);

/**
 * @brief Obtém nome da denominação
 * @param tipo Tipo da moeda
 * @return String com nome da denominação
 */
string obterNomeDenominacao(TipoMoeda tipo);

/**
 * @brief Processa e classifica todas as moedas detectadas
 * @param circulos Vetor de círculos detectados
 * @return Vetor de moedas classificadas
 */
vector<Moeda> processarMoedas(const vector<Vec3f>& circulos);

// ============================================================================
// FUNÇÕES DE CÁLCULO
// ============================================================================

/**
 * @brief Calcula valor total das moedas
 * @param moedas Vetor de moedas
 * @return Valor total em reais
 */
double calcularValorTotal(const vector<Moeda>& moedas);

/**
 * @brief Conta moedas por denominação
 * @param moedas Vetor de moedas
 * @return Map com contagem por denominação
 */
map<string, int> contarPorDenominacao(const vector<Moeda>& moedas);

/**
 * @brief Detecta sobreposições entre moedas
 * @param moedas Vetor de moedas
 * @return Pares de índices de moedas sobrepostas
 */
vector<pair<int, int>> detectarSobreposicoes(const vector<Moeda>& moedas);

// ============================================================================
// FUNÇÕES DE VISUALIZAÇÃO
// ============================================================================

/**
 * @brief Desenha resultados na imagem
 * @param imagemOriginal Imagem original
 * @param moedas Vetor de moedas detectadas
 * @return Imagem com anotações
 */
Mat desenharResultados(const Mat& imagemOriginal, const vector<Moeda>& moedas);

/**
 * @brief Exibe resumo no console
 * @param moedas Vetor de moedas
 */
void exibirResumo(const vector<Moeda>& moedas);

// ============================================================================
// FUNÇÃO PRINCIPAL DO PIPELINE
// ============================================================================

/**
 * @brief Executa pipeline completo de detecção
 * @param imagemOriginal Imagem de entrada
 * @param tipoFiltro Tipo de filtro a usar
 * @param minRaio Raio mínimo
 * @param maxRaio Raio máximo
 * @return Estrutura com todos os resultados
 */
ResultadoDeteccao detectarEContarMoedas(const Mat& imagemOriginal,
                                         int tipoFiltro = 0,
                                         int minRaio = 20,
                                         int maxRaio = 150);

#endif // COIN_COUNTER_H
