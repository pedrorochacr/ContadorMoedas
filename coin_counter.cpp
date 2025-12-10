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
// CONSTANTES DE CALIBRAÇÃO
// ============================================================================
// Raios reais das moedas brasileiras em mm (diâmetro / 2)
const float RAIO_1_REAL_MM = 13.5f;       // 27mm diâmetro - bimetálica
const float RAIO_25_CENTAVOS_MM = 12.5f;  // 25mm diâmetro - prateada
const float RAIO_50_CENTAVOS_MM = 11.5f;  // 23mm diâmetro - prateada
const float RAIO_5_CENTAVOS_MM = 11.0f;   // 22mm diâmetro - dourada
const float RAIO_10_CENTAVOS_MM = 10.0f;  // 20mm diâmetro - dourada

// ============================================================================
// ENUM PARA COR DA MOEDA
// ============================================================================
enum class CorMoeda {
    DOURADA,    // 5 e 10 centavos (bronze/cobre)
    PRATEADA,   // 25 e 50 centavos (aço inox)
    BIMETALICA  // 1 real (centro prata, borda dourada)
};

// ============================================================================
// FUNÇÃO 5: Analisar cor da moeda
// ============================================================================
CorMoeda analisarCorMoeda(const Mat& imagem, Point2f centro, float raio) {
    // Cria máscara circular para o centro da moeda
    Mat mascara = Mat::zeros(imagem.size(), CV_8UC1);
    circle(mascara, centro, (int)(raio * 0.6), Scalar(255), -1); // 60% do raio para centro
    
    // Converte para HSV para análise de cor
    Mat hsv;
    cvtColor(imagem, hsv, COLOR_BGR2HSV);
    
    // Calcula cor média do centro
    Scalar corCentro = mean(hsv, mascara);
    float hCentro = corCentro[0]; // Hue (0-180 no OpenCV)
    float sCentro = corCentro[1]; // Saturation (0-255)
    float vCentro = corCentro[2]; // Value (0-255)
    
    // Cria máscara só da borda
    Mat mascaraBorda = Mat::zeros(imagem.size(), CV_8UC1);
    circle(mascaraBorda, centro, (int)raio, Scalar(255), -1);
    circle(mascaraBorda, centro, (int)(raio * 0.7), Scalar(0), -1);
    
    Scalar corBorda = mean(hsv, mascaraBorda);
    float hBorda = corBorda[0];
    float sBorda = corBorda[1];
    
    cout << "    Cor Centro HSV: H=" << fixed << setprecision(1) << hCentro 
         << " S=" << sCentro << " V=" << vCentro
         << " | Borda H=" << hBorda << " S=" << sBorda;
    
    // Detecta BIMETÁLICA: borda dourada E centro diferente (menos saturado ou Hue diferente)
    bool bordaDourada = (hBorda >= 10 && hBorda <= 40 && sBorda > 100);
    bool centroDiferente = (sCentro < sBorda - 50) || (hCentro > 35);  // Centro menos saturado ou Hue mais alto
    
    if (bordaDourada && centroDiferente) {
        cout << " -> BIMETALICA" << endl;
        return CorMoeda::BIMETALICA;
    }
    
    // DOURADA (bronze/cobre): Hue entre 8-45, Saturação alta (> 100)
    // Moedas douradas tem Hue na faixa laranja/amarelo (8-45)
    if (hCentro >= 8 && hCentro <= 45 && sCentro > 100) {
        cout << " -> DOURADA" << endl;
        return CorMoeda::DOURADA;
    }
    
    // PRATEADA: 
    // 1. Saturação baixa (cinza claro) - moeda limpa
    // 2. Hue fora do range dourado (>45 ou <8) - moeda suja/escura
    if (sCentro < 80) {
        cout << " -> PRATEADA" << endl;
        return CorMoeda::PRATEADA;
    }
    
    // Moeda escura com Hue fora do range dourado = prateada suja
    if (hCentro > 45 || hCentro < 8) {
        cout << " -> PRATEADA (escura/suja)" << endl;
        return CorMoeda::PRATEADA;
    }
    
    // Fallback: se tiver saturação alta e Hue no range, é dourada
    cout << " -> DOURADA (fallback)" << endl;
    return CorMoeda::DOURADA;
}

// ============================================================================
// FUNÇÃO 6: Encontrar maior raio por grupo de cor
// ============================================================================
struct ReferenciaPorCor {
    float maiorRaioDourada = 0;
    float maiorRaioPrateada = 0;
    float maiorRaioBimetalica = 0;
    int countDourada = 0;
    int countPrateada = 0;
    int countBimetalica = 0;
};

ReferenciaPorCor calcularReferenciasPorCor(const Mat& imagem, const vector<Vec3f>& circulos) {
    ReferenciaPorCor ref;
    
    for (const auto& c : circulos) {
        Point2f centro(c[0], c[1]);
        float raio = c[2];
        CorMoeda cor = analisarCorMoeda(imagem, centro, raio);
        
        switch (cor) {
            case CorMoeda::DOURADA:
                if (raio > ref.maiorRaioDourada) ref.maiorRaioDourada = raio;
                ref.countDourada++;
                break;
            case CorMoeda::PRATEADA:
                if (raio > ref.maiorRaioPrateada) ref.maiorRaioPrateada = raio;
                ref.countPrateada++;
                break;
            case CorMoeda::BIMETALICA:
                if (raio > ref.maiorRaioBimetalica) ref.maiorRaioBimetalica = raio;
                ref.countBimetalica++;
                break;
        }
    }
    
    return ref;
}

// ============================================================================
// FUNÇÃO 7: Classificar Moeda por Cor e Tamanho
// ============================================================================
TipoMoeda classificarMoedaPorCorETamanho(float raioPixels, CorMoeda cor, const ReferenciaPorCor& ref) {
    
    // Se é bimetálica, é 1 real (única moeda bimetálica)
    if (cor == CorMoeda::BIMETALICA) {
        return TipoMoeda::UM_REAL;
    }
    
    // DOURADAS: 5 centavos (11mm) e 10 centavos (10mm)
    if (cor == CorMoeda::DOURADA) {
        if (ref.countDourada == 1) {
            // Só uma moeda dourada - precisa adivinhar pelo tamanho absoluto
            if (ref.maiorRaioPrateada > 0) {
                // Usa prateada como referência
                // 50 centavos = 11.5mm, 25 centavos = 12.5mm
                // Assume menor prateada = 50 centavos
                float mmPorPixel = RAIO_50_CENTAVOS_MM / ref.maiorRaioPrateada;
                float raioMM = raioPixels * mmPorPixel;
                return (raioMM < 10.5f) ? TipoMoeda::DEZ_CENTAVOS : TipoMoeda::CINCO_CENTAVOS;
            }
            // Sem referência - assume 5 centavos (mais comum)
            return TipoMoeda::CINCO_CENTAVOS;
        }
        
        // Múltiplas douradas: maior = 5 centavos (11mm), menor = 10 centavos (10mm)
        float mmPorPixel = RAIO_5_CENTAVOS_MM / ref.maiorRaioDourada;
        float raioMM = raioPixels * mmPorPixel;
        
        cout << "  Raio: " << fixed << setprecision(1) << raioPixels 
             << " px -> " << raioMM << " mm (dourada)" << endl;
        
        // Threshold: média entre 10mm e 11mm = 10.5mm
        // Mas na prática a diferença é pequena, então uso 10.3mm para ser mais conservador
        return (raioMM < 10.3f) ? TipoMoeda::DEZ_CENTAVOS : TipoMoeda::CINCO_CENTAVOS;
    }
    
    // PRATEADAS: 25 centavos (12.5mm) e 50 centavos (11.5mm)
    if (cor == CorMoeda::PRATEADA) {
        if (ref.countPrateada == 1) {
            // Só uma moeda prateada - precisa adivinhar
            if (ref.maiorRaioDourada > 0) {
                // 5 centavos (maior dourada) = 11mm
                float mmPorPixel = RAIO_5_CENTAVOS_MM / ref.maiorRaioDourada;
                float raioMM = raioPixels * mmPorPixel;
                return (raioMM < 12.0f) ? TipoMoeda::CINQUENTA_CENTAVOS : TipoMoeda::VINTE_CINCO_CENTAVOS;
            }
            // Sem referência - assume 25 centavos (mais comum)
            return TipoMoeda::VINTE_CINCO_CENTAVOS;
        }
        
        // Múltiplas prateadas: maior = 25 centavos (12.5mm), menor = 50 centavos (11.5mm)
        float mmPorPixel = RAIO_25_CENTAVOS_MM / ref.maiorRaioPrateada;
        float raioMM = raioPixels * mmPorPixel;
        
        cout << "  Raio: " << fixed << setprecision(1) << raioPixels 
             << " px -> " << raioMM << " mm (prateada)" << endl;
        
        return (raioMM < 12.0f) ? TipoMoeda::CINQUENTA_CENTAVOS : TipoMoeda::VINTE_CINCO_CENTAVOS;
    }
    
    return TipoMoeda::UM_REAL; // fallback
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
// FUNÇÃO 9: Processar e Classificar Todas as Moedas
// ============================================================================
vector<Moeda> processarMoedas(const vector<Vec3f>& circulos, const Mat& imagemOriginal) {
    vector<Moeda> moedas;
    
    if (circulos.empty()) {
        return moedas;
    }
    
    cout << "\n[ANALISE DE COR] Detectando cores das moedas...\n" << endl;
    
    // Primeira passada: analisa cores e calcula referências
    ReferenciaPorCor ref = calcularReferenciasPorCor(imagemOriginal, circulos);
    
    cout << "\n[CALIBRACAO]" << endl;
    cout << "  Douradas: " << ref.countDourada << " (maior raio: " << ref.maiorRaioDourada << " px)" << endl;
    cout << "  Prateadas: " << ref.countPrateada << " (maior raio: " << ref.maiorRaioPrateada << " px)" << endl;
    cout << "  Bimetalicas: " << ref.countBimetalica << " (maior raio: " << ref.maiorRaioBimetalica << " px)" << endl;
    cout << endl;
    
    // Segunda passada: classifica cada moeda
    cout << "[CLASSIFICACAO]" << endl;
    for (const auto& circulo : circulos) {
        Moeda moeda;
        moeda.centro = Point2f(circulo[0], circulo[1]);
        moeda.raio = circulo[2];
        
        // Analisa cor novamente (ou poderia cachear)
        CorMoeda cor = analisarCorMoeda(imagemOriginal, moeda.centro, moeda.raio);
        
        // Classifica por cor e tamanho
        TipoMoeda tipo = classificarMoedaPorCorETamanho(moeda.raio, cor, ref);
        moeda.valor = obterValorMoeda(tipo);
        moeda.denominacao = obterNomeDenominacao(tipo);
        
        cout << "  -> " << moeda.denominacao << endl;
        
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
    
    // Etapa 4: Classificação das moedas (agora usa cor + tamanho)
    resultado.moedas = processarMoedas(circulos, imagemOriginal);
    
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
    
    // Exibe as imagens (se houver display disponível)
    #ifdef SHOW_GUI
    const char* display = getenv("DISPLAY");
    if (display != nullptr && strlen(display) > 0) {
        try {
            namedWindow("Imagem Original", WINDOW_NORMAL);
            namedWindow("Imagem Processada", WINDOW_NORMAL);
            namedWindow("Resultado", WINDOW_NORMAL);
            
            imshow("Imagem Original", imagem);
            imshow("Imagem Processada", resultado.imagemProcessada);
            imshow("Resultado", resultado.imagemResultado);
            
            cout << "\nPressione qualquer tecla para sair..." << endl;
            waitKey(0);
            
            destroyAllWindows();
        } catch (...) {
            cout << "[INFO] Display nao disponivel. Salvando imagens..." << endl;
        }
    } else {
        cout << "[INFO] Display nao disponivel. Salvando imagens..." << endl;
    }
    #endif
    
    // Salva as imagens de resultado
    imwrite("resultado_processada.jpg", resultado.imagemProcessada);
    imwrite("resultado_final.jpg", resultado.imagemResultado);
    
    cout << "\n[INFO] Imagens salvas:" << endl;
    cout << "  - resultado_processada.jpg" << endl;
    cout << "  - resultado_final.jpg" << endl;
    
    return 0;
}