#!/usr/bin/env python3
"""
Contador de Moedas Brasileiras usando YOLOv8
Detecta moedas com HoughCircles e classifica com YOLOv8
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

def carregar_modelo(modelo_path):
    """Carrega o modelo YOLOv8"""
    try:
        from ultralytics import YOLO
        model = YOLO(modelo_path)
        print(f"[INFO] Modelo carregado: {modelo_path}")
        return model
    except Exception as e:
        print(f"[ERRO] Falha ao carregar modelo: {e}")
        return None

def detectar_circulos(imagem, min_raio=20, max_raio=200):
    """Detecta círculos usando HoughCircles"""
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=min_raio,
        maxRadius=max_raio
    )
    
    if circles is not None:
        return circles[0]
    return []

def preparar_imagem_moeda(imagem, centro, raio):
    """
    Prepara a imagem de uma moeda para classificação.
    Aplica fundo cinza ao redor para simular imagens de treino.
    """
    h, w = imagem.shape[:2]
    cx, cy = int(centro[0]), int(centro[1])
    r = int(raio)
    
    # Cria imagem com fundo cinza
    resultado = np.full_like(imagem, (180, 180, 180))
    
    # Cria máscara circular
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(r * 1.1), 255, -1)
    
    # Copia moeda para fundo cinza
    resultado = np.where(mask[:, :, None] == 255, imagem, resultado)
    
    return resultado

def classificar_moeda(model, imagem):
    """Classifica uma moeda usando YOLOv8"""
    results = model(imagem, verbose=False)
    
    if results and len(results) > 0:
        probs = results[0].probs
        classe_idx = probs.top1
        confianca = probs.top1conf.item()
        nome_classe = results[0].names[classe_idx]
        return nome_classe, confianca
    
    return None, 0.0

def valor_moeda(classe):
    """Retorna o valor em reais de uma classe"""
    valores = {
        '5': 0.05,
        '10': 0.10,
        '25': 0.25,
        '50': 0.50,
        '100': 1.00
    }
    return valores.get(classe, 0.0)

def nome_moeda(classe):
    """Retorna o nome legível da moeda"""
    nomes = {
        '5': '5 centavos',
        '10': '10 centavos',
        '25': '25 centavos',
        '50': '50 centavos',
        '100': '1 real'
    }
    return nomes.get(classe, 'Desconhecida')

def processar_imagem(imagem_path, modelo_path, salvar_resultado=True):
    """Processa uma imagem e conta as moedas"""
    
    # Carrega imagem
    imagem = cv2.imread(str(imagem_path))
    if imagem is None:
        print(f"[ERRO] Não foi possível carregar: {imagem_path}")
        return None
    
    print(f"[INFO] Imagem carregada: {imagem_path}")
    print(f"[INFO] Dimensões: {imagem.shape[1]}x{imagem.shape[0]}")
    
    # Carrega modelo
    model = carregar_modelo(modelo_path)
    if model is None:
        return None
    
    # Detecta círculos
    print("\n[1/2] Detectando moedas (HoughCircles)...")
    circulos = detectar_circulos(imagem)
    print(f"      {len(circulos)} círculos detectados")
    
    if len(circulos) == 0:
        print("[AVISO] Nenhuma moeda detectada")
        return None
    
    # Classifica cada moeda
    print("\n[2/2] Classificando moedas (YOLOv8)...")
    
    resultado_img = imagem.copy()
    moedas = []
    valor_total = 0.0
    
    for circulo in circulos:
        cx, cy, raio = circulo
        
        # Prepara imagem com fundo cinza
        img_preparada = preparar_imagem_moeda(imagem, (cx, cy), raio)
        
        # Classifica
        classe, confianca = classificar_moeda(model, img_preparada)
        
        if classe:
            valor = valor_moeda(classe)
            nome = nome_moeda(classe)
            valor_total += valor
            
            moedas.append({
                'classe': classe,
                'nome': nome,
                'valor': valor,
                'confianca': confianca,
                'centro': (int(cx), int(cy)),
                'raio': int(raio)
            })
            
            print(f"      Moeda em ({int(cx)}, {int(cy)}): {nome} (confiança: {confianca*100:.1f}%)")
            
            # Desenha na imagem
            cor = (0, 255, 0) if confianca > 0.8 else (0, 255, 255) if confianca > 0.5 else (0, 0, 255)
            cv2.circle(resultado_img, (int(cx), int(cy)), int(raio), cor, 2)
            cv2.circle(resultado_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            
            label = f"{nome} ({confianca*100:.0f}%)"
            cv2.putText(resultado_img, label, (int(cx) - 50, int(cy) - int(raio) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
    
    # Exibe resumo
    print("\n" + "=" * 40)
    print("      RESUMO DA CONTAGEM DE MOEDAS")
    print("=" * 40)
    
    # Conta por denominação
    contagem = {}
    for m in moedas:
        nome = m['nome']
        contagem[nome] = contagem.get(nome, 0) + 1
    
    print("Quantidade por denominação:")
    print("-" * 40)
    for nome, qtd in sorted(contagem.items()):
        print(f"  {nome}: {qtd} moeda(s)")
    print("-" * 40)
    print(f"Total de moedas: {len(moedas)}")
    print(f"Valor total: R$ {valor_total:.2f}")
    print("=" * 40)
    
    # Salva resultado
    if salvar_resultado:
        output_path = "resultado.jpg"
        cv2.imwrite(output_path, resultado_img)
        print(f"\n[INFO] Imagem resultado salva em: {output_path}")
    
    return {
        'moedas': moedas,
        'valor_total': valor_total,
        'imagem_resultado': resultado_img
    }

def main():
    parser = argparse.ArgumentParser(description='Contador de Moedas Brasileiras')
    parser.add_argument('imagem', help='Caminho da imagem')
    parser.add_argument('--modelo', '-m', default='runs/moedas_cls/weights/best.pt',
                       help='Caminho do modelo YOLOv8 (default: runs/moedas_cls/weights/best.pt)')
    parser.add_argument('--no-save', action='store_true', help='Não salvar imagem resultado')
    
    args = parser.parse_args()
    
    processar_imagem(args.imagem, args.modelo, salvar_resultado=not args.no_save)

if __name__ == '__main__':
    main()