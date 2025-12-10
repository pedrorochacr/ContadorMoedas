/**
 * CONTADOR DE MOEDAS BRASILEIRAS COM MACHINE LEARNING
 * 
 * Este programa detecta e classifica moedas brasileiras em imagens usando:
 * - OpenCV para detecção de círculos (HoughCircles)
 * - YOLOv8 (via ONNX) para classificação das moedas
 * 
 * Dependências:
 * - OpenCV 4.x com módulo DNN
 * 
 * Uso:
 *   ./coin_counter_ml <imagem> [--model <modelo.onnx>]
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// ============================================================================
// ESTRUTURAS DE DADOS
// ============================================================================

struct Moeda {
    Point2f centro;
    float raio;
    int classe;           // Índice da classe (0-4)
    float confianca;      // Confiança da classificação
    double valor;         // Valor em reais
    string denominacao;   // Nome (ex: "50 centavos")
};

struct ResultadoDeteccao {
    vector<Moeda> moedas;
    double valorTotal;
    int quantidadeTotal;
    Mat imagemResultado;
};

// ============================================================================
// CLASSE PRINCIPAL: CLASSIFICADOR DE MOEDAS
// ============================================================================

class ClassificadorMoedas {
private:
    Net net;
    vector<string> classNames;
    map<string, double> valoresMoedas;
    bool modeloCarregado;
    int inputSize;
    
public:
    ClassificadorMoedas() : modeloCarregado(false), inputSize(224) {
        // Mapeamento de valores das moedas (suporta vários formatos de nome)
        valoresMoedas["5"] = 0.05;
        valoresMoedas["10"] = 0.10;
        valoresMoedas["25"] = 0.25;
        valoresMoedas["50"] = 0.50;
        valoresMoedas["100"] = 1.00;
        // Formato com sufixo
        valoresMoedas["5_centavos"] = 0.05;
        valoresMoedas["10_centavos"] = 0.10;
        valoresMoedas["25_centavos"] = 0.25;
        valoresMoedas["50_centavos"] = 0.50;
        valoresMoedas["100_real"] = 1.00;
        // Novo formato com zeros à esquerda
        valoresMoedas["005_5centavos"] = 0.05;
        valoresMoedas["010_10centavos"] = 0.10;
        valoresMoedas["025_25centavos"] = 0.25;
        valoresMoedas["050_50centavos"] = 0.50;
        valoresMoedas["100_1real"] = 1.00;
    }
    
    /**
     * Carrega o modelo ONNX e o arquivo de classes
     */
    bool carregarModelo(const string& modeloPath, const string& classesPath) {
        try {
            // Carrega modelo ONNX
            cout << "[INFO] Carregando modelo: " << modeloPath << endl;
            net = readNetFromONNX(modeloPath);
            
            if (net.empty()) {
                cerr << "[ERRO] Falha ao carregar modelo ONNX" << endl;
                return false;
            }
            
            // Configura backend (CUDA se disponível, senão CPU)
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_CPU);
            
            // IMPORTANTE: YOLOv8 ordena as classes ALFABETICAMENTE!
            // Ordem alfabética de "10", "100", "25", "5", "50":
            // 10 < 100 < 25 < 5 < 50
            // Então os índices são:
            //   0 = "10"  (10 centavos)
            //   1 = "100" (1 real)
            //   2 = "25"  (25 centavos)
            //   3 = "5"   (5 centavos)
            //   4 = "50"  (50 centavos)
            
            classNames.clear();
            classNames.push_back("10");   // índice 0
            classNames.push_back("100");  // índice 1
            classNames.push_back("25");   // índice 2
            classNames.push_back("5");    // índice 3
            classNames.push_back("50");   // índice 4
            
            cout << "[INFO] Classes em ordem alfabética (como YOLOv8 usa): ";
            for (size_t i = 0; i < classNames.size(); i++) {
                cout << "[" << i << "]=" << classNames[i] << " ";
            }
            cout << endl;
            
            modeloCarregado = true;
            return true;
            
        } catch (const Exception& e) {
            cerr << "[ERRO] Exceção ao carregar modelo: " << e.what() << endl;
            return false;
        }
    }
    
    /**
     * Classifica uma região da imagem contendo uma moeda
     */
    pair<int, float> classificar(const Mat& imagemMoeda, bool debug = false) {
        if (!modeloCarregado || imagemMoeda.empty()) {
            return {-1, 0.0f};
        }
        
        try {
            // Pré-processa EXATAMENTE como YOLOv8 faz:
            // 1. Resize para 224x224
            // 2. BGR -> RGB
            // 3. Normaliza [0, 255] -> [0, 1]
            // 4. HWC -> CHW
            
            Mat blob;
            // blobFromImage com swapRB=true converte BGR->RGB
            // scalefactor=1/255 normaliza para [0,1]
            blobFromImage(imagemMoeda, blob, 
                          1.0/255.0,           // scale factor
                          Size(inputSize, inputSize),  // size 224x224
                          Scalar(0, 0, 0),     // mean subtraction (none)
                          true,                // swapRB: BGR -> RGB
                          false);              // crop
            
            if (debug) {
                cout << "      [DEBUG] Blob shape: [" << blob.size[0] << ", " << blob.size[1] 
                     << ", " << blob.size[2] << ", " << blob.size[3] << "]" << endl;
                     
                // Verifica alguns valores do blob
                float* data = (float*)blob.data;
                cout << "      [DEBUG] Primeiros pixels (R,G,B): " 
                     << data[0] << ", " << data[224*224] << ", " << data[2*224*224] << endl;
            }
            
            // Executa inferência
            net.setInput(blob);
            Mat output = net.forward();
            
            Mat probs = output.reshape(1, 1);
            int numClasses = probs.cols;
            
            if (debug) {
                cout << "      [DEBUG] Output classes: " << numClasses << endl;
                cout << "      [DEBUG] Raw output: ";
                for (int i = 0; i < numClasses; i++) {
                    cout << classNames[i] << "=" << fixed << setprecision(4) << probs.at<float>(0, i) << " ";
                }
                cout << endl;
            }
            
            // Encontra classe com maior probabilidade
            Point maxLoc;
            double maxVal;
            minMaxLoc(probs, nullptr, &maxVal, nullptr, &maxLoc);
            
            int classId = maxLoc.x;
            float confidence = (float)maxVal;
            
            if (debug) {
                cout << "      [DEBUG] Classe: " << classId << " (" << classNames[classId] 
                     << ") conf: " << (confidence * 100) << "%" << endl;
            }
            
            return {classId, confidence};
            
        } catch (const Exception& e) {
            cerr << "[ERRO] Exceção na classificação: " << e.what() << endl;
            return {-1, 0.0f};
        }
    }
    
    /**
     * Retorna o valor em reais para uma classe
     */
    double getValor(int classId) {
        if (classId < 0 || classId >= (int)classNames.size()) {
            return 0.0;
        }
        string className = classNames[classId];
        auto it = valoresMoedas.find(className);
        return (it != valoresMoedas.end()) ? it->second : 0.0;
    }
    
    /**
     * Retorna o nome da denominação
     */
    string getDenominacao(int classId) {
        if (classId < 0 || classId >= (int)classNames.size()) {
            return "Desconhecida";
        }
        string className = classNames[classId];
        
        // Mapeia para nome legível
        if (className.find("005") != string::npos || className == "5" || className == "5_centavos") 
            return "5 centavos";
        if (className.find("010") != string::npos || className == "10" || className == "10_centavos") 
            return "10 centavos";
        if (className.find("025") != string::npos || className == "25" || className == "25_centavos") 
            return "25 centavos";
        if (className.find("050") != string::npos || className == "50" || className == "50_centavos") 
            return "50 centavos";
        if (className.find("100") != string::npos || className == "100_real") 
            return "1 real";
        
        return className;
    }
    
    bool estaCarregado() const { return modeloCarregado; }
};

// ============================================================================
// FUNÇÕES DE DETECÇÃO DE CÍRCULOS
// ============================================================================

/**
 * Detecta círculos (moedas) na imagem usando HoughCircles
 */
vector<Vec3f> detectarCirculos(const Mat& imagem, int minRaio = 20, int maxRaio = 200) {
    Mat gray, blurred;
    
    // Converte para cinza
    if (imagem.channels() == 3) {
        cvtColor(imagem, gray, COLOR_BGR2GRAY);
    } else {
        gray = imagem.clone();
    }
    
    // Aplica blur para reduzir ruído
    GaussianBlur(gray, blurred, Size(9, 9), 2, 2);
    
    // Detecta círculos
    vector<Vec3f> circles;
    HoughCircles(blurred, circles, HOUGH_GRADIENT, 1,
                 50,      // Distância mínima entre centros
                 100,     // Limiar Canny
                 30,      // Limiar acumulador
                 minRaio, // Raio mínimo
                 maxRaio);// Raio máximo
    
    return circles;
}

/**
 * Extrai a região de uma moeda da imagem, aplicando fundo cinza ao redor
 * para simular as imagens de treino
 */
Mat extrairMoeda(const Mat& imagem, Point2f centro, float raio, int idx = 0) {
    // Cria uma cópia da imagem com fundo cinza
    Mat resultado = Mat(imagem.size(), imagem.type(), Scalar(180, 180, 180));
    
    // Cria máscara circular para a moeda (com margem)
    Mat mask = Mat::zeros(imagem.size(), CV_8UC1);
    circle(mask, centro, (int)(raio * 1.1), Scalar(255), -1);
    
    // Copia apenas a região da moeda para o fundo cinza
    imagem.copyTo(resultado, mask);
    
    // Agora recorta uma região quadrada centrada na moeda
    // com proporções similares às imagens de treino (moeda ocupa ~40% da imagem)
    float fator = 4.0f;
    int tamanho = (int)(raio * fator);
    
    int x = max(0, (int)(centro.x - tamanho / 2));
    int y = max(0, (int)(centro.y - tamanho / 2));
    int w = min(tamanho, resultado.cols - x);
    int h = min(tamanho, resultado.rows - y);
    int lado = min(w, h);
    
    Rect roi(x, y, lado, lado);
    Mat recorte = resultado(roi).clone();
    
    // Salva para debug
    if (idx == 0) {
        imwrite("debug_recorte_cpp.jpg", recorte);
        cout << "      [DEBUG] Recorte salvo em debug_recorte_cpp.jpg" << endl;
    }
    
    return recorte;
}

// ============================================================================
// FUNÇÃO PRINCIPAL DE DETECÇÃO E CLASSIFICAÇÃO
// ============================================================================

ResultadoDeteccao detectarEClassificarMoedas(
    const Mat& imagem,
    ClassificadorMoedas& classificador,
    int minRaio = 20,
    int maxRaio = 200
) {
    ResultadoDeteccao resultado;
    resultado.valorTotal = 0;
    resultado.quantidadeTotal = 0;
    resultado.imagemResultado = imagem.clone();
    
    // Detecta círculos
    cout << "\n[1/2] Detectando moedas (HoughCircles)..." << endl;
    vector<Vec3f> circulos = detectarCirculos(imagem, minRaio, maxRaio);
    cout << "      " << circulos.size() << " círculos detectados" << endl;
    
    if (circulos.empty()) {
        cout << "[AVISO] Nenhuma moeda detectada" << endl;
        return resultado;
    }
    
    // Classifica cada moeda
    cout << "\n[2/2] Classificando moedas (YOLOv8)..." << endl;
    
    int idx = 0;
    for (const auto& c : circulos) {
        Moeda moeda;
        moeda.centro = Point2f(c[0], c[1]);
        moeda.raio = c[2];
        
        // Extrai região da moeda
        Mat moedaImg = extrairMoeda(imagem, moeda.centro, moeda.raio, idx);
        
        // Classifica com ML (debug na primeira moeda)
        bool debug = (resultado.moedas.empty()); // Debug só na primeira
        auto [classId, confianca] = classificador.classificar(moedaImg, debug);
        
        moeda.classe = classId;
        moeda.confianca = confianca;
        moeda.valor = classificador.getValor(classId);
        moeda.denominacao = classificador.getDenominacao(classId);
        
        cout << "      Moeda em (" << (int)moeda.centro.x << ", " << (int)moeda.centro.y 
             << "): " << moeda.denominacao 
             << " (confiança: " << fixed << setprecision(1) << (confianca * 100) << "%)" << endl;
        
        resultado.moedas.push_back(moeda);
        resultado.valorTotal += moeda.valor;
        idx++;
    }
    
    resultado.quantidadeTotal = resultado.moedas.size();
    
    // Desenha resultados
    for (const auto& moeda : resultado.moedas) {
        // Cor baseada na confiança
        Scalar cor;
        if (moeda.confianca > 0.8) {
            cor = Scalar(0, 255, 0);  // Verde: alta confiança
        } else if (moeda.confianca > 0.5) {
            cor = Scalar(0, 255, 255); // Amarelo: média
        } else {
            cor = Scalar(0, 0, 255);   // Vermelho: baixa
        }
        
        // Desenha círculo
        circle(resultado.imagemResultado, moeda.centro, (int)moeda.raio, cor, 2);
        circle(resultado.imagemResultado, moeda.centro, 3, Scalar(0, 0, 255), -1);
        
        // Desenha label
        string label = moeda.denominacao + " (" + 
                      to_string((int)(moeda.confianca * 100)) + "%)";
        Point textPos(moeda.centro.x - 50, moeda.centro.y - moeda.raio - 10);
        putText(resultado.imagemResultado, label, textPos, 
                FONT_HERSHEY_SIMPLEX, 0.5, cor, 2);
    }
    
    return resultado;
}

// ============================================================================
// FUNÇÕES DE RELATÓRIO
// ============================================================================

void exibirResumo(const ResultadoDeteccao& resultado) {
    cout << "\n========================================" << endl;
    cout << "      RESUMO DA CONTAGEM DE MOEDAS     " << endl;
    cout << "========================================\n" << endl;
    
    // Conta por denominação
    map<string, int> contagem;
    for (const auto& moeda : resultado.moedas) {
        contagem[moeda.denominacao]++;
    }
    
    cout << "Quantidade por denominação:" << endl;
    cout << "----------------------------------------" << endl;
    for (const auto& [denom, qtd] : contagem) {
        cout << "  " << denom << ": " << qtd << " moeda(s)" << endl;
    }
    
    cout << "----------------------------------------" << endl;
    cout << "Total de moedas: " << resultado.quantidadeTotal << endl;
    cout << "Valor total: R$ " << fixed << setprecision(2) << resultado.valorTotal << endl;
    cout << "========================================\n" << endl;
}

// ============================================================================
// MAIN
// ============================================================================

void printUsage(const char* programName) {
    cout << "Uso: " << programName << " <imagem> [opções]\n" << endl;
    cout << "Opções:" << endl;
    cout << "  --model <path>    Caminho para modelo ONNX (default: models/moedas_classifier.onnx)" << endl;
    cout << "  --classes <path>  Caminho para arquivo de classes (default: models/classes.txt)" << endl;
    cout << "  --min-raio <int>  Raio mínimo para detecção (default: 20)" << endl;
    cout << "  --max-raio <int>  Raio máximo para detecção (default: 200)" << endl;
    cout << "  --output <path>   Caminho para salvar imagem resultado" << endl;
    cout << "\nExemplo:" << endl;
    cout << "  " << programName << " moedas.jpg --model models/moedas_classifier.onnx" << endl;
}

int main(int argc, char** argv) {
    // Valores padrão
    string imagePath;
    string modelPath = "models/moedas_classifier.onnx";
    string classesPath = "models/classes.txt";
    string outputPath = "resultado.jpg";
    int minRaio = 20;
    int maxRaio = 200;
    
    // Parse argumentos
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    imagePath = argv[1];
    
    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--classes" && i + 1 < argc) {
            classesPath = argv[++i];
        } else if (arg == "--min-raio" && i + 1 < argc) {
            minRaio = stoi(argv[++i]);
        } else if (arg == "--max-raio" && i + 1 < argc) {
            maxRaio = stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    // Carrega imagem
    cout << "\n[INFO] Carregando imagem: " << imagePath << endl;
    Mat imagem = imread(imagePath);
    
    if (imagem.empty()) {
        cerr << "[ERRO] Não foi possível carregar a imagem: " << imagePath << endl;
        return 1;
    }
    
    cout << "[INFO] Dimensões: " << imagem.cols << "x" << imagem.rows << endl;
    
    // Inicializa classificador
    ClassificadorMoedas classificador;
    
    if (!classificador.carregarModelo(modelPath, classesPath)) {
        cerr << "\n[AVISO] Modelo não encontrado. Usando classificação por cor/tamanho." << endl;
        cerr << "        Para usar ML, treine o modelo com: python train_yolo.py --dataset <pasta>" << endl;
        
        // Fallback: usa detecção simples sem ML
        // (Aqui você poderia integrar o código anterior de classificação por cor)
        return 1;
    }
    
    // Detecta e classifica
    ResultadoDeteccao resultado = detectarEClassificarMoedas(
        imagem, classificador, minRaio, maxRaio
    );
    
    // Exibe resumo
    exibirResumo(resultado);
    
    // Salva imagem resultado
    imwrite(outputPath, resultado.imagemResultado);
    cout << "[INFO] Imagem resultado salva em: " << outputPath << endl;
    
    return 0;
}