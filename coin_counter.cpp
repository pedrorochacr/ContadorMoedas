#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

using namespace cv;
using namespace std;

// Estrutura para armazenar informações de uma moeda detectada
struct Moeda {
    Point2f centro;
    float raio;
    double valor;
    string denominacao;
};

// Enum para tipos de moedas brasileiras (baseado no diâmetro em mm)
// 5 centavos: 22mm, 10 centavos: 20mm, 25 centavos: 25mm
// 50 centavos: 23mm, 1 real: 27mm
enum class TipoMoeda {
    CINCO_CENTAVOS,
    DEZ_CENTAVOS,
    VINTE_CINCO_CENTAVOS,
    CINQUENTA_CENTAVOS,
    UM_REAL,
    DESCONHECIDA
};

// ============================================================================
// FUNÇÃO 1: Conversão para Escala de Cinza
// ============================================================================
Mat converterParaCinza(const Mat& imagemOriginal) {
    Mat imagemCinza;
    
    if (imagemOriginal.channels() == 3) {
        cvtColor(imagemOriginal, imagemCinza, COLOR_BGR2GRAY);
    } else if (imagemOriginal.channels() == 4) {
        cvtColor(imagemOriginal, imagemCinza, COLOR_BGRA2GRAY);
    } else {
        imagemCinza = imagemOriginal.clone();
    }
    
    return imagemCinza;
}

// ============================================================================
// FUNÇÃO 2: Filtragem - Redução de Ruído
// ============================================================================
Mat aplicarFiltragem(const Mat& imagemCinza, int tipoFiltro = 0) {
    Mat imagemFiltrada;
    
    switch (tipoFiltro) {
        case 0:
            // Filtro Gaussiano - bom para ruído geral
            GaussianBlur(imagemCinza, imagemFiltrada, Size(9, 9), 2, 2);
            break;
        case 1:
            // Filtro de Mediana - bom para ruído sal e pimenta
            medianBlur(imagemCinza, imagemFiltrada, 5);
            break;
        case 2:
            // Filtro Bilateral - preserva bordas
            bilateralFilter(imagemCinza, imagemFiltrada, 9, 75, 75);
            break;
        default:
            imagemFiltrada = imagemCinza.clone();
    }
    
    return imagemFiltrada;
}

// ============================================================================
// FUNÇÃO 3: Segmentação - Separar Objetos de Interesse
// ============================================================================
Mat segmentarImagem(const Mat& imagemFiltrada, int metodo = 0) {
    Mat imagemSegmentada;
    
    switch (metodo) {
        case 0: {
            // Threshold adaptativo
            adaptiveThreshold(imagemFiltrada, imagemSegmentada, 255,
                            ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
            break;
        }
        case 1: {
            // Otsu's thresholding
            threshold(imagemFiltrada, imagemSegmentada, 0, 255,
                     THRESH_BINARY_INV | THRESH_OTSU);
            break;
        }
        case 2: {
            // Canny edge detection
            Canny(imagemFiltrada, imagemSegmentada, 50, 150);
            break;
        }
        default:
            imagemSegmentada = imagemFiltrada.clone();
    }
    
    return imagemSegmentada;
}

// ============================================================================
// FUNÇÃO 4: Detecção de Círculos usando Transformada de Hough
// ============================================================================
vector<Vec3f> detectarCirculos(const Mat& imagemFiltrada, 
                               int minRaio = 20, 
                               int maxRaio = 150,
                               double param1 = 100,
                               double param2 = 30,
                               int minDist = 50) {
    vector<Vec3f> circulos;
    
    HoughCircles(imagemFiltrada, circulos, HOUGH_GRADIENT, 1,
                 minDist,      // Distância mínima entre centros
                 param1,       // Limiar superior para Canny
                 param2,       // Limiar do acumulador
                 minRaio,      // Raio mínimo
                 maxRaio);     // Raio máximo
    
    return circulos;
}

// ============================================================================
// FUNÇÃO 5: Classificar Moeda por Tamanho (Raio)
// ============================================================================
TipoMoeda classificarMoedaPorRaio(float raio, float raioReferencia = 0) {
    // Se não tiver referência, usa proporções relativas
    // Assumindo que a maior moeda (1 real) tem ~27mm de diâmetro
    // e a menor (10 centavos) tem ~20mm
    
    // Normaliza o raio se houver referência
    float raioNormalizado = raio;
    
    // Definindo ranges aproximados (podem ser ajustados conforme calibração)
    // Estes valores são proporcionais e devem ser calibrados para cada câmera
    if (raioNormalizado < 35) {
        return TipoMoeda::DEZ_CENTAVOS;       // Menor moeda
    } else if (raioNormalizado < 40) {
        return TipoMoeda::CINCO_CENTAVOS;     // 22mm
    } else if (raioNormalizado < 45) {
        return TipoMoeda::CINQUENTA_CENTAVOS; // 23mm
    } else if (raioNormalizado < 50) {
        return TipoMoeda::VINTE_CINCO_CENTAVOS; // 25mm
    } else {
        return TipoMoeda::UM_REAL;            // 27mm - maior moeda
    }
}

// ============================================================================
// FUNÇÃO 6: Obter Valor da Moeda
// ============================================================================
double obterValorMoeda(TipoMoeda tipo) {
    switch (tipo) {
        case TipoMoeda::CINCO_CENTAVOS:
            return 0.05;
        case TipoMoeda::DEZ_CENTAVOS:
            return 0.10;
        case TipoMoeda::VINTE_CINCO_CENTAVOS:
            return 0.25;
        case TipoMoeda::CINQUENTA_CENTAVOS:
            return 0.50;
        case TipoMoeda::UM_REAL:
            return 1.00;
        default:
            return 0.0;
    }
}

// ============================================================================
// FUNÇÃO 7: Obter Nome da Denominação
// ============================================================================
string obterNomeDenominacao(TipoMoeda tipo) {
    switch (tipo) {
        case TipoMoeda::CINCO_CENTAVOS:
            return "5 centavos";
        case TipoMoeda::DEZ_CENTAVOS:
            return "10 centavos";
        case TipoMoeda::VINTE_CINCO_CENTAVOS:
            return "25 centavos";
        case TipoMoeda::CINQUENTA_CENTAVOS:
            return "50 centavos";
        case TipoMoeda::UM_REAL:
            return "1 real";
        default:
            return "Desconhecida";
    }
}

// ============================================================================
// FUNÇÃO 8: Processar e Classificar Todas as Moedas
// ============================================================================
vector<Moeda> processarMoedas(const vector<Vec3f>& circulos) {
    vector<Moeda> moedas;
    
    // Encontra o maior e menor raio para calibração relativa
    float maxRaio = 0, minRaio = FLT_MAX;
    for (const auto& c : circulos) {
        if (c[2] > maxRaio) maxRaio = c[2];
        if (c[2] < minRaio) minRaio = c[2];
    }
    
    for (const auto& circulo : circulos) {
        Moeda moeda;
        moeda.centro = Point2f(circulo[0], circulo[1]);
        moeda.raio = circulo[2];
        
        TipoMoeda tipo = classificarMoedaPorRaio(moeda.raio);
        moeda.valor = obterValorMoeda(tipo);
        moeda.denominacao = obterNomeDenominacao(tipo);
        
        moedas.push_back(moeda);
    }
    
    return moedas;
}

// ============================================================================
// FUNÇÃO 9: Calcular Valor Total
// ============================================================================
double calcularValorTotal(const vector<Moeda>& moedas) {
    double total = 0.0;
    for (const auto& moeda : moedas) {
        total += moeda.valor;
    }
    return total;
}

// ============================================================================
// FUNÇÃO 10: Contar Moedas por Denominação
// ============================================================================
map<string, int> contarPorDenominacao(const vector<Moeda>& moedas) {
    map<string, int> contagem;
    
    for (const auto& moeda : moedas) {
        contagem[moeda.denominacao]++;
    }
    
    return contagem;
}

// ============================================================================
// FUNÇÃO 11: Desenhar Resultados na Imagem
// ============================================================================
Mat desenharResultados(const Mat& imagemOriginal, const vector<Moeda>& moedas) {
    Mat imagemResultado = imagemOriginal.clone();
    
    for (const auto& moeda : moedas) {
        // Desenha o círculo da moeda
        circle(imagemResultado, moeda.centro, (int)moeda.raio, 
               Scalar(0, 255, 0), 2);
        
        // Desenha o centro
        circle(imagemResultado, moeda.centro, 3, Scalar(0, 0, 255), -1);
        
        // Escreve o valor da moeda
        string texto = moeda.denominacao;
        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 1;
        
        Point textOrg(moeda.centro.x - 30, moeda.centro.y - moeda.raio - 10);
        putText(imagemResultado, texto, textOrg, fontFace, fontScale,
                Scalar(255, 0, 0), thickness);
    }
    
    return imagemResultado;
}

// ============================================================================
// FUNÇÃO 12: Exibir Resumo no Console
// ============================================================================
void exibirResumo(const vector<Moeda>& moedas) {
    cout << "\n========================================" << endl;
    cout << "      RESUMO DA CONTAGEM DE MOEDAS     " << endl;
    cout << "========================================\n" << endl;
    
    map<string, int> contagem = contarPorDenominacao(moedas);
    
    cout << "Quantidade por denominacao:" << endl;
    cout << "----------------------------------------" << endl;
    
    for (const auto& par : contagem) {
        cout << "  " << par.first << ": " << par.second << " moeda(s)" << endl;
    }
    
    cout << "----------------------------------------" << endl;
    cout << "Total de moedas: " << moedas.size() << endl;
    
    double valorTotal = calcularValorTotal(moedas);
    cout << "Valor total: R$ " << fixed << setprecision(2) << valorTotal << endl;
    cout << "========================================\n" << endl;
}

// ============================================================================
// FUNÇÃO 13: Pipeline Completo de Detecção
// ============================================================================
struct ResultadoDeteccao {
    vector<Moeda> moedas;
    double valorTotal;
    int quantidadeTotal;
    Mat imagemProcessada;
    Mat imagemResultado;
};

ResultadoDeteccao detectarEContarMoedas(const Mat& imagemOriginal,
                                         int tipoFiltro = 0,
                                         int minRaio = 20,
                                         int maxRaio = 150) {
    ResultadoDeteccao resultado;
    
    // Etapa 1: Conversão para escala de cinza
    Mat imagemCinza = converterParaCinza(imagemOriginal);
    
    // Etapa 2: Filtragem
    Mat imagemFiltrada = aplicarFiltragem(imagemCinza, tipoFiltro);
    resultado.imagemProcessada = imagemFiltrada.clone();
    
    // Etapa 3: Detecção de círculos (HoughCircles já trabalha com imagem em cinza)
    vector<Vec3f> circulos = detectarCirculos(imagemFiltrada, minRaio, maxRaio);
    
    // Etapa 4: Classificação das moedas
    resultado.moedas = processarMoedas(circulos);
    
    // Etapa 5: Cálculos finais
    resultado.valorTotal = calcularValorTotal(resultado.moedas);
    resultado.quantidadeTotal = resultado.moedas.size();
    
    // Etapa 6: Desenhar resultados
    resultado.imagemResultado = desenharResultados(imagemOriginal, resultado.moedas);
    
    return resultado;
}

// ============================================================================
// FUNÇÃO 14: Ajustar Parâmetros de Detecção (para calibração)
// ============================================================================
struct ParametrosDeteccao {
    int tipoFiltro;
    double param1;
    double param2;
    int minRaio;
    int maxRaio;
    int minDist;
};

vector<Vec3f> detectarComParametros(const Mat& imagemFiltrada, 
                                     const ParametrosDeteccao& params) {
    vector<Vec3f> circulos;
    
    HoughCircles(imagemFiltrada, circulos, HOUGH_GRADIENT, 1,
                 params.minDist,
                 params.param1,
                 params.param2,
                 params.minRaio,
                 params.maxRaio);
    
    return circulos;
}

// ============================================================================
// FUNÇÃO 15: Equalização de Histograma (melhora contraste)
// ============================================================================
Mat equalizarHistograma(const Mat& imagemCinza) {
    Mat imagemEqualizada;
    equalizeHist(imagemCinza, imagemEqualizada);
    return imagemEqualizada;
}

// ============================================================================
// FUNÇÃO 16: Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
// ============================================================================
Mat aplicarCLAHE(const Mat& imagemCinza, double clipLimit = 2.0, Size tileSize = Size(8, 8)) {
    Mat imagemCLAHE;
    Ptr<CLAHE> clahe = createCLAHE(clipLimit, tileSize);
    clahe->apply(imagemCinza, imagemCLAHE);
    return imagemCLAHE;
}

// ============================================================================
// FUNÇÃO 17: Detectar Sobreposições
// ============================================================================
vector<pair<int, int>> detectarSobreposicoes(const vector<Moeda>& moedas) {
    vector<pair<int, int>> sobreposicoes;
    
    for (size_t i = 0; i < moedas.size(); i++) {
        for (size_t j = i + 1; j < moedas.size(); j++) {
            float distancia = norm(moedas[i].centro - moedas[j].centro);
            float somaRaios = moedas[i].raio + moedas[j].raio;
            
            // Se a distância entre centros for menor que a soma dos raios,
            // há sobreposição
            if (distancia < somaRaios * 0.9) { // 0.9 para tolerância
                sobreposicoes.push_back(make_pair(i, j));
            }
        }
    }
    
    return sobreposicoes;
}

// ============================================================================
// FUNÇÃO MAIN - Exemplo de Uso
// ============================================================================
int main(int argc, char** argv) {
    // Verifica argumentos
    string caminhoImagem;
    if (argc > 1) {
        caminhoImagem = argv[1];
    } else {
        cout << "Uso: " << argv[0] << " <caminho_da_imagem>" << endl;
        cout << "\nExecutando com imagem de teste..." << endl;
        caminhoImagem = "moedas.jpg"; // Imagem padrão
    }
    
    // Carrega a imagem
    Mat imagem = imread(caminhoImagem);
    
    if (imagem.empty()) {
        cerr << "Erro: Nao foi possivel carregar a imagem: " << caminhoImagem << endl;
        cerr << "\nCriando imagem de demonstracao..." << endl;
        
        // Cria uma imagem de demonstração com círculos simulando moedas
        imagem = Mat(600, 800, CV_8UC3, Scalar(200, 200, 200));
        
        // Simula moedas de diferentes tamanhos
        circle(imagem, Point(150, 150), 50, Scalar(180, 150, 100), -1); // 1 real
        circle(imagem, Point(350, 200), 45, Scalar(180, 150, 100), -1); // 25 centavos
        circle(imagem, Point(550, 150), 40, Scalar(200, 180, 100), -1); // 50 centavos
        circle(imagem, Point(200, 400), 35, Scalar(200, 180, 100), -1); // 5 centavos
        circle(imagem, Point(400, 400), 30, Scalar(180, 150, 100), -1); // 10 centavos
        circle(imagem, Point(600, 400), 50, Scalar(180, 150, 100), -1); // 1 real
        
        // Adiciona ruído para simular condições reais
        Mat ruido(imagem.size(), imagem.type());
        randn(ruido, 0, 10);
        imagem += ruido;
    }
    
    cout << "\n[INFO] Processando imagem: " << caminhoImagem << endl;
    cout << "[INFO] Dimensoes: " << imagem.cols << "x" << imagem.rows << endl;
    
    // Executa o pipeline de detecção
    ResultadoDeteccao resultado = detectarEContarMoedas(imagem, 0, 25, 100);
    
    // Exibe o resumo
    exibirResumo(resultado.moedas);
    
    // Verifica sobreposições
    auto sobreposicoes = detectarSobreposicoes(resultado.moedas);
    if (!sobreposicoes.empty()) {
        cout << "[AVISO] Detectadas " << sobreposicoes.size() 
             << " possiveis sobreposicoes de moedas." << endl;
    }
    
 
    
    // Salva as imagens de resultado
    imwrite("resultado_processada.jpg", resultado.imagemProcessada);
    imwrite("resultado_final.jpg", resultado.imagemResultado);
    
    cout << "\n[INFO] Imagens salvas:" << endl;
    cout << "  - resultado_processada.jpg" << endl;
    cout << "  - resultado_final.jpg" << endl;
    
    return 0;
}
