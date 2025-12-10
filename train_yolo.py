#!/usr/bin/env python3
"""
Treina um modelo YOLOv8 para classificação de moedas brasileiras.

Este script:
- Organiza imagens em pastas por classe (train/val)
- Treina modelo YOLOv8n-cls
- Exporta para ONNX (compatível com OpenCV DNN)

Uso:
  python3 train_yolo.py --dataset <pasta_imagens> --epochs 30

Estrutura esperada das imagens:
  <valor>_<id>.jpg  (ex: 50_1477283178.jpg = moeda de 50 centavos)
  Valores: 5, 10, 25, 50, 100 (100 = 1 real)
"""

import argparse
import random
import shutil
from pathlib import Path
import sys


def find_images(src_dir):
    """Encontra todas as imagens na pasta."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    return [p for p in src_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]


def extract_label_from_name(name: str):
    """Extrai o label do nome do arquivo (ex: '50_123.jpg' -> '50')."""
    if '_' in name:
        return name.split('_', 1)[0]
    return name


def prepare_dataset(src_dir: Path, out_dir: Path, val_fraction=0.2, seed=42):
    """Organiza o dataset em pastas train/val por classe."""
    images = find_images(src_dir)
    if not images:
        raise SystemExit(f'Nenhuma imagem encontrada em {src_dir!s}')

    # Agrupa imagens por label
    classes = {}
    for p in images:
        lbl = extract_label_from_name(p.name)
        classes.setdefault(lbl, []).append(p)

    print(f"\nClasses encontradas:")
    for lbl, items in sorted(classes.items()):
        print(f"  {lbl}: {len(items)} imagens")

    random.seed(seed)
    out_train = out_dir / 'train'
    out_val = out_dir / 'val'
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    names = []
    for lbl, items in sorted(classes.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        names.append(lbl)
        # Cria subpastas por classe
        (out_train / lbl).mkdir(parents=True, exist_ok=True)
        (out_val / lbl).mkdir(parents=True, exist_ok=True)
        
        # Shuffle e split
        random.shuffle(items)
        cut = int(len(items) * (1 - val_fraction))
        train_items = items[:cut]
        val_items = items[cut:]
        
        # Copia arquivos
        for src in train_items:
            dst = out_train / lbl / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
        for src in val_items:
            dst = out_val / lbl / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
        
        print(f"  {lbl}: {len(train_items)} train, {len(val_items)} val")

    print(f"\nDataset organizado em: {out_dir}")
    print(f"  - {out_train}")
    print(f"  - {out_val}")
    
    # Retorna o diretório (não YAML) e lista de classes
    return out_dir, names


def train(data_dir: Path, epochs: int, imgsz: int, batch: int, lr: float, project: str):
    """Treina o modelo YOLOv8."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print('Erro: ultralytics não instalado.')
        print('Execute: pip install ultralytics')
        sys.exit(1)

    print(f'\nIniciando treinamento...')
    print(f'  Dataset: {data_dir}')
    print(f'  Epochs: {epochs}')
    print(f'  Batch size: {batch}')
    print(f'  Image size: {imgsz}')
    
    # Carrega modelo base
    model = YOLO('yolov8n-cls.pt')
    
    # Treina - YOLOv8 classificação espera diretório, não YAML
    results = model.train(
        data=str(data_dir),  # Diretório com train/ e val/
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr,
        project=project,
        name='moedas_cls',
        exist_ok=True
    )
    
    return model, results


def export_model(model, export_dir: Path):
    """Exporta modelo para ONNX (compatível com OpenCV DNN)."""
    print("\nExportando modelo para ONNX...")
    
    # Exporta para ONNX
    onnx_path = model.export(format='onnx', imgsz=224, simplify=True)
    
    # Copia para diretório de saída
    out_onnx = export_dir / 'moedas_classifier.onnx'
    shutil.copy2(onnx_path, out_onnx)
    
    print(f"Modelo ONNX salvo em: {out_onnx}")
    return out_onnx


def parse_args():
    p = argparse.ArgumentParser(description='Treina YOLOv8 para classificação de moedas')
    p.add_argument('--dataset', type=Path, required=True,
                   help='Pasta com imagens (nomeadas como <valor>_<id>.jpg)')
    p.add_argument('--out', type=Path, default=Path('data/cls_dataset'),
                   help='Pasta de saída do dataset organizado')
    p.add_argument('--val', type=float, default=0.2,
                   help='Fração para validação (default: 0.2)')
    p.add_argument('--epochs', type=int, default=30,
                   help='Número de epochs (default: 30)')
    p.add_argument('--imgsz', type=int, default=224,
                   help='Tamanho da imagem (default: 224)')
    p.add_argument('--batch', type=int, default=16,
                   help='Batch size (default: 16)')
    p.add_argument('--lr', type=float, default=0.01,
                   help='Learning rate (default: 0.01)')
    p.add_argument('--project', type=str, default='runs',
                   help='Pasta para salvar resultados do treino')
    p.add_argument('--export', type=Path, default=Path('models'),
                   help='Pasta para exportar modelo ONNX')
    return p.parse_args()


def main():
    args = parse_args()
    
    if not args.dataset.exists():
        print(f'Erro: pasta não encontrada: {args.dataset}')
        sys.exit(1)

    print("=" * 60)
    print("  TREINAMENTO DE CLASSIFICADOR DE MOEDAS - YOLOv8")
    print("=" * 60)
    
    # Prepara dataset
    print(f'\n[1/3] Preparando dataset de {args.dataset}...')
    data_dir, class_names = prepare_dataset(
        args.dataset, args.out, val_fraction=args.val
    )
    
    # Treina modelo
    print(f'\n[2/3] Treinando modelo...')
    model, results = train(
        data_dir,  # Passa diretório, não YAML
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr=args.lr,
        project=args.project
    )
    
    # Exporta modelo
    print(f'\n[3/3] Exportando modelo...')
    args.export.mkdir(parents=True, exist_ok=True)
    onnx_path = export_model(model, args.export)
    
    # Salva mapeamento de classes
    classes_file = args.export / 'classes.txt'
    classes_file.write_text('\n'.join(class_names))
    print(f"Classes salvas em: {classes_file}")
    
    print("\n" + "=" * 60)
    print("  TREINAMENTO CONCLUÍDO!")
    print("=" * 60)
    print(f"\nArquivos gerados:")
    print(f"  - Modelo ONNX: {onnx_path}")
    print(f"  - Classes: {classes_file}")
    print(f"\nPara usar no C++, copie esses arquivos para a pasta do projeto.")


if __name__ == '__main__':
    main()