import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',   type=str, default='data.yaml',   help='YAML dosyası yolu')
    p.add_argument('--model',  type=str, default='yolo11s-obb.pt',  help='Ağırlık dosyası')
    p.add_argument('--epochs', type=int, default=60,            help='Epoch sayısı')
    p.add_argument('--imgsz',  type=int, default=640,           help='Input image size')
    p.add_argument('--degrees', type=float, default=45,         help='Rastgele döndürme açısı')
    p.add_argument('--translate', type=float, default=0.1,      help='Rastgele çevirme')
    p.add_argument('--scale', type=float, default=0.5,          help='Rastgele ölçeklendirme')
    p.add_argument('--shear', type=float, default=0.1,          help='Rastgele kaydırma')
    p.add_argument('--mosaic', type=float, default=1.0,         help='Mozaik veri artırma')
    p.add_argument('--mixup', type=float, default=0.2,          help='Mixup veri artırma')
    return p.parse_args()

def main():
    args = parse_args()
    # Modeli yükle
    model = YOLO(args.model)
    # Eğitime başla
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        degrees=args.degrees,      # Rastgele döndürme açısı
        translate=args.translate,  # Rastgele çevirme
        scale=args.scale,          # Rastgele ölçeklendirme
        shear=args.shear,          # Rastgele kaydırma
        mosaic=args.mosaic,        # Mozaik veri artırmayı etkinleştir
        mixup=args.mixup,          # Mixup veri artırmayı etkinleştir
        device='0',              # GPU seçimi (opsiyonel)
        batch=16,                # Batch size (opsiyonel)
    )

if __name__ == '__main__':
    main()