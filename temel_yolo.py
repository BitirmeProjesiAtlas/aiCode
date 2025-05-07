import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import time
from ultralytics import YOLO

def main():
    # 1) Modeli yükle (OBB eğitilmiş ağırlık)
    model = YOLO(r"runs\obb\train3\weights\best.pt")
    names = model.names

    # 2) Oyun penceresini bul
    window_title = "topdownchar (64-bit Development PCD3D_SM5)"
    wins = gw.getWindowsWithTitle(window_title)
    if not wins:
        raise RuntimeError(f"Pencere bulunamadı: {window_title}")
    win = wins[0]
    win.activate()
    time.sleep(0.5)
    x, y, w, h = win.left, win.top, win.width, win.height

    # 3) Renkler (BGR)
    COLOR_WALL_TRAP = (0, 0, 255)    # kırmızı
    COLOR_FINISH    = (0, 255, 0)    # yeşil
    COLOR_CHAR      = (255, 0, 0)    # mavi
    COLOR_FPS       = (0, 165, 255)  # turuncu

    # 4) Pencereyi oluştur ve en sağa yapıştır
    cv2.namedWindow("YOLO OBB Detect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO OBB Detect", 640, 480)
    sw, sh = pyautogui.size()
    cv2.moveWindow("YOLO OBB Detect", sw - 640, 0)

    # 5) FPS ölçüm zamanı
    prev_time = time.time()

    while True:
        # 6) Ekran görüntüsü
        screenshot = pyautogui.screenshot(region=(x, y, w, h))
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # 7) Model tahmini (OBB)
        res_list = model(frame, verbose=False)
        if not res_list:
            # boşsa sadece FPS yaz
            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_FPS, 2)
            cv2.imshow("YOLO OBB Detect", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        results = res_list[0]

        # 8) OBB kutularını çiz
        for obb in getattr(results, "obb", []):
            cls_id = int(obb.cls.cpu().numpy())
            label = names[cls_id]

            # 8.1) 4 köşe noktasını al
            pts = obb.xyxyxyxy.int().cpu().numpy().reshape(4, 2)

            # 8.2) Renk seç
            if label == "wall" or label in ("floor_trap", "laser", "spinner"):
                color = COLOR_WALL_TRAP
            elif label == "finish_line":
                color = COLOR_FINISH
            elif label == "main_char":
                color = COLOR_CHAR
            else:
                color = (200, 200, 200)

            # 8.3) Poligon çiz ve etiketle
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            # Etiketi birinci köşenin biraz yukarısına yaz
            x0, y0 = pts[0]
            cv2.putText(frame, label, (int(x0), int(y0) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 9) FPS hesapla ve yaz
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_FPS, 2)

        # 10) Göster ve çıkış kontrolü
        cv2.imshow("YOLO OBB Detect", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
