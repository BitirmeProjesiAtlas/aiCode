import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import time
from ultralytics import YOLO

def main():
    model = YOLO(r"runs\obb\train3\weights\best.pt")
    names = model.names

    window_title = "topdownchar (64-bit Development PCD3D_SM5)"
    wins = gw.getWindowsWithTitle(window_title)
    if not wins:
        raise RuntimeError(f"Pencere bulunamadı: {window_title}")
    win = wins[0]
    win.activate()
    time.sleep(0.5)
    x, y, w, h = win.left, win.top, win.width, win.height

    COLOR_WALL_TRAP = (0, 0, 255)    
    COLOR_FINISH    = (0, 255, 0)    
    COLOR_CHAR      = (255, 0, 0)    
    COLOR_FPS       = (0, 165, 255)  

    cv2.namedWindow("YOLO OBB Detect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO OBB Detect", 640, 480)
    sw, sh = pyautogui.size()
    cv2.moveWindow("YOLO OBB Detect", sw - 640, 0)

    prev_time = time.time()

    while True:
        screenshot = pyautogui.screenshot(region=(x, y, w, h))
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        res_list = model(frame, verbose=False)
        if not res_list:
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
        for obb in getattr(results, "obb", []):
            cls_id = int(obb.cls.cpu().numpy())
            label = names[cls_id]
            pts = obb.xyxyxyxy.int().cpu().numpy().reshape(4, 2)

            if label == "wall" or label in ("floor_trap", "laser", "spinner"):
                color = COLOR_WALL_TRAP
            elif label == "finish_line":
                color = COLOR_FINISH
            elif label == "main_char":
                color = COLOR_CHAR
            else:
                color = (200, 200, 200)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            x0, y0 = pts[0]
            cv2.putText(frame, label, (int(x0), int(y0) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_FPS, 2)
        cv2.imshow("YOLO OBB Detect", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
