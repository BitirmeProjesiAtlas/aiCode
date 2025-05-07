import os
import time
import json
import cv2
import numpy as np
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.distributions import Categorical
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
try:
    import win32api, win32con, win32gui
    WIN32_OK = True
except ImportError:
    WIN32_OK = False

pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False
s=0

WINDOW_W, WINDOW_H = 960, 1080
SAFE_MARGIN = 300
FPS_ALPHA = 0.02
GRID_X, GRID_Y = 15, 9

REWARD_SCALE = 1/50 

class Config:
    REWARD = dict(
        movement=0.1,
        progress=0.3,
        wall_avoid=0.2,
        vert_cross=4.0,
        trap_cross=4.0,
        finish_reached=500  
    )
    PENALTY = dict(
        backward_move=10,
        wall_collision=80,      
        PENALTY_STATIONARY=10,
        spinner_collision=100, 
        dangerous_move=10  
    )


def clip(v, low=-200, hi=400):
    return float(np.clip(v, low, hi))

np_pt = lambda p: np.array(p).reshape(-1, 2)

def closest_pts(a, b):
    d = cdist(a, b)
    i = np.unravel_index(np.argmin(d), d.shape)
    return a[i[0]], b[i[1]], d[i]

def pt_line_dist(p, a, b):
    v, ap = b - a, p - a
    t = np.clip(np.dot(ap, v) / (np.linalg.norm(v)**2 + 1e-6), 0, 1)
    return np.linalg.norm(p - (a + t * v))

def closest_distance(obj1_pts, obj2_pts):
    d = cdist(np_pt(obj1_pts), np_pt(obj2_pts))
    return np.min(d)

def rects_intersect(r1, r2, margin=0):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (
        x1 + w1 + margin < x2 or
        x2 + w2 + margin < x1 or
        y1 + h1 + margin < y2 or
        y2 + h2 + margin < y1
    )


class RewardLogger:
    def __init__(self):
        self.hist = []
        self.step_id = 0
        self.total_reward = 0.0
        self.start_time = time.time()

    def log(self, info, action=None, fps=None):
        self.step_id += 1
        self.total_reward += info.get("total", 0)

        entry = {
            "step": self.step_id,
            "timestamp": time.strftime("%F %T"),
            "elapsed_sec": round(time.time() - self.start_time, 3),
            "reward": info.get("total", 0),
            "cumulative_reward": self.total_reward,
            "details": info.get("details", []),
            "action": action,
            "fps": round(fps, 2) if fps is not None else None
        }
        self.hist.append(entry)

    def export(self, fp='reward_logs.json'):
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(self.hist, f, indent=2, ensure_ascii=False)

class RewardSystem:
    def __init__(self, debug=None):
        self.debug = debug
        self.log = RewardLogger()
        self.prev_finish = None
        self.last_stat_t = time.time()
        self.last_char_pos = None
        self.crossed_barriers = set()
        self.last_action = None
        self.last_click_time = 0.0
        self.max_finish_distance = 600  
        self.best_distance = float("inf")  
        self.crossed_traps = set()  
        self.last_trap_warning = 0  

    def calculate(self, st, path_blocked=0.0):  
        r = 0
        details = []

        if st["character_center"] is not None:
            cx, cy = st["character_center"]
            c_h     = max(st.get("character_height", 40), 24)  
            char_bb = [cx - c_h / 2, cy - c_h / 2, c_h, c_h]

            def _check_collisions(objs, penalty_key, label):
                nonlocal r
                for bb in objs:
                    if rects_intersect(char_bb, bb, margin=35):
                        r -= Config.PENALTY[penalty_key]
                        details.append(f"{label} collision -{Config.PENALTY[penalty_key]}")
                        st["done"] = True
                        return True
                return False

            if _check_collisions(st["spinner_detections"], "spinner_collision", "Spinner"):
                time.sleep(0.8) 
            elif _check_collisions(st["laser_detections"],   "spinner_collision", "Laser"):
                time.sleep(0.5)
            elif _check_collisions(st["floor_trap_detections"], "spinner_collision", "Spike‑trap"):
                time.sleep(0.5)

        if st["character_center"] is not None and ((len(st["floor_trap_detections"]) > 0) or (len(st["laser_detections"]) > 0)):
            cx, cy = st["character_center"]
            for trap in st["floor_trap_detections"] + st["laser_detections"]:
                tx, ty, tw, th = trap
                if ty > cy and abs(tx + tw / 2 - cx) < 50:
                    if self.last_action == 'w':
                        r -= Config.PENALTY["dangerous_move"]
                        details.append("Dangerous move -10")
                    elif self.last_action is None:
                        r += 0.1
                        details.append("Caution +0.1")
        if st["character_center"] is not None and len(st["wall_detections"]) > 0:
            char_pts = [st["character_center"]]
            closest_wall_dist = float("inf")

            for wall in st["wall_detections"]:
                wx, wy, ww, wh = wall
                cx, cy = st["character_center"]
                
                left, right = wx, wx + ww
                top, bottom = wy, wy + wh
                
                dx = max(left - cx, cx - right, 0)
                dy = max(top - cy, cy - bottom, 0)
                
                dist = np.sqrt(dx**2 + dy**2)
                closest_wall_dist = min(closest_wall_dist, dist)

            if closest_wall_dist < 50:  
                r -= Config.PENALTY["wall_collision"]
                details.append(f"Wall collision -{Config.PENALTY['wall_collision']}")
                st["done"] = True
                time.sleep(0.1)  
        
        if st["character_center"] is not None and st["finish_corners"] is not None:
            current_dist = closest_distance([st["character_center"]], st["finish_corners"])
            if current_dist <= self.max_finish_distance:
                progress_reward = (1 - (current_dist / self.max_finish_distance)) * 2  
                r += progress_reward
                details.append(f"Finish progress +{progress_reward:.2f}")
        
        if st["character_center"] is not None and st["finish_corners"] is not None:  
            current_dist = closest_distance([st["character_center"]], st["finish_corners"])
            
            if current_dist < 120:  
                r += Config.REWARD["finish_reached"]  
                details.append(f"Finish reached +{Config.REWARD['finish_reached']}")
                st["done"] = True
                time.sleep(2.4)  

        if path_blocked and self.last_action in ('a', 'd'):  
            r += 0.2  
            details.append("Path obstacle avoided +0.2")
        
        
        if st["character_center"] is not None and st["finish_corners"] is not None:
            finish_pts = st["finish_corners"]
            current_dist = closest_distance([st["character_center"]], finish_pts)

            
            if current_dist > self.best_distance:
                 if self.best_distance == float("inf"):
                    self.best_distance = current_dist
                    penalty = (current_dist - self.best_distance) * 0.0001  
                    r -= penalty
                    details.append(f"Distance penalty -{penalty:.2f}")
            else:
                 self.best_distance = current_dist  


            if current_dist < 120:
                r += 200
                details.append(f"Finish proximity +200")
                st["done"] = True
                time.sleep(2.4)

        if st["character_center"] is not None:
            cy = st["character_center"][1]
            for i, trap in enumerate(st["wall_detections"]):
                tx, ty, tw, th = trap
                trap_id = (tx, ty)  
                if cy < ty and trap_id not in self.crossed_traps:
                    r += 4
                    details.append(f"wall crossed +4")
                    self.crossed_traps.add(trap_id)


        r = clip(r)

        if self.last_action is not None:
            self.log.log(dict(total=r, details=details), action=self.last_action)

        return r, details

class GameMonitor:
    def __init__(self, title, model_path, debug=None):
        self.title = title
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.reward = RewardSystem(debug)
        self.fps = self.total = 0.0
        self.prev_time = time.time()
        self.latest_state = None
        self.closed = False
        self._setup_window()

    def _setup_window(self):
        wins = pyautogui.getWindowsWithTitle(self.title)
        if not wins:
            raise RuntimeError(f"Pencere bulunamadı: {self.title}")
        win = wins[0]
        win.activate(); time.sleep(0.05)

        self.w, self.h = WINDOW_W, WINDOW_H
        left, top = win.left, win.top
        self.x, self.y = left, top

        cv2.namedWindow("AI-Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI-Monitor", 980, 700)
        sw, _ = pyautogui.size()
        cv2.moveWindow("AI-Monitor", sw-980, 50)

    def _real_click(self, sx, sy):
        min_x = self.x + SAFE_MARGIN
        max_x = self.x + self.w - SAFE_MARGIN - 1
        min_y = self.y + SAFE_MARGIN
        max_y = self.y + self.h - SAFE_MARGIN - 1
        sx = int(np.clip(sx, min_x, max_x))
        sy = int(np.clip(sy, min_y, max_y))

        self.win = pyautogui.getWindowsWithTitle(self.title)[0]
        self.win.activate()
        if WIN32_OK:
            win32gui.SetForegroundWindow(self.win._hWnd)
        time.sleep(0.02)
        pyautogui.moveTo(sx, sy)
        if WIN32_OK:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0,0,0,0)
            time.sleep(0.02)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0,0,0,0)
            cx, cy = sx - self.x, sy - self.y
            lparam = cy<<16 | cx
            for msg in (win32con.WM_LBUTTONDOWN, win32con.WM_LBUTTONUP):
                win32gui.PostMessage(self.win._hWnd, msg, win32con.MK_LBUTTON, lparam)
        else:
            pyautogui.click()

    def _parse_state(self, res):
        s = dict(
            character_center=None, character_height=0,
            finish_center=None, finish_corners=None,   
            wall_detections=[],
            floor_trap_detections=[],
            laser_detections=[],
            spinner_detections=[],
            done=False
        )
        det = res[0].obb if res[0].obb else res[0].boxes
        if det is None:
            return s

        for b in det:
            if res[0].obb:
                pts  = b.xyxyxyxy.int().cpu().numpy().reshape(4, 2)
                rect = cv2.boundingRect(pts)
            else:
                x1, y1, x2, y2 = b.xyxy.int().cpu().numpy().flatten()
                pts  = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                rect = (x1, y1, x2 - x1, y2 - y1)

            lbl = self.names[int(b.cls)]

            if lbl == "main_char":
                s["character_center"]  = np.mean(pts, 0)
                s["character_height"]  = rect[3]

            elif lbl == "finish_line":                        # ⬅️ ➕
                s["finish_corners"] = pts.tolist()
                s["finish_center"]  = np.mean(pts, 0)

            elif lbl == "floor_trap":
                s["floor_trap_detections"].append(rect)
            elif lbl == "laser":
                s["laser_detections"].append(rect)
            elif lbl == "spinner":
                s["spinner_detections"].append(rect)
            elif lbl == "wall":
                s["wall_detections"].append(rect)

        return s


    def _draw_overlay(self, frame, res, info):
        det = res[0].obb if res[0].obb else res[0].boxes
        if det:
            for b in det:
                if res[0].obb:
                    pts = b.xyxyxyxy.int().cpu().numpy().reshape(4,2)
                else:
                    x1,y1,x2,y2 = b.xyxy.int().cpu().numpy().flatten()
                    pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
                
                class_name = self.names[int(b.cls)]
                color = (0,255,0) 
                
                if class_name == "main_char":
                    color = (0, 255, 0)  
                elif class_name == "finish_line":
                    color = (255, 0, 0)  
                elif class_name in ("floor_trap","laser","spinner"):
                    color = (0, 255, 255)  
                elif class_name == "wall":
                    color = (0, 0, 255)  

                cv2.polylines(frame,[pts],True,color,2)

        cv2.putText(frame, f"FPS: {self.fps:4.1f}", (10,26), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
        cv2.putText(frame, f"Total: {self.total:+.1f}", (10,52), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        y0 = 80
        for idx,t in enumerate(info[:12]):
            col = (0,255,0) if "+" in t else (0,0,255)
            cv2.putText(frame,t,(10,y0+idx*20)  ,cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)


class RLGameEnv(GameMonitor):
    ACTION_SPACE = ['w', 'a', 'd', 'wa', 'wd','none']  
    OBS_DIM = 37

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        frame = np.array(pyautogui.screenshot(region=(self.x, self.y, self.w, self.h)))
        res = self.model.predict(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), verbose=False, conf=1)
        self.latest_state = self._parse_state(res)
        self.last_action_time = 0.0  

    def step(self, action):
        
        pyautogui.keyUp('w')
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')

        
        if 'w' in action:
            pyautogui.keyDown('w')
        if 'a' in action:
            pyautogui.keyDown('a')
        if 'd' in action:
            pyautogui.keyDown('d')

        
        frame = np.array(pyautogui.screenshot(region=(self.x, self.y, self.w, self.h)))
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        res = self.model.predict(bgr, verbose=False, conf=0.4)

        self.latest_state = self._parse_state(res)
        self.reward.last_action = action  

       
        obs = self._get_obs()
        path_blocked = obs[9]  

        
        r, info = self.reward.calculate(self.latest_state, path_blocked=path_blocked)
        self.total += r

        done = (self.closed or self.latest_state.get("done", False) or self.total <= -1000)

        self.fps = self.fps * (1 - FPS_ALPHA) + (1.0 / (time.time() - self.prev_time + 1e-6)) * FPS_ALPHA
        self.prev_time = time.time()
        self._draw_overlay(bgr, res, info)
        cv2.imshow("AI-Monitor", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[MON] 'q' ile çıkış")
            self.closed = True
            os._exit(0)

        return obs, r, done, {"reason": "|".join(info)}

    def reset(self):
        if self.closed:
            raise RuntimeError("Pencere kapalı")
        self.total = 0
        self.reward.prev_finish = None
        time.sleep(0.3)
        return self._get_obs()

    def _get_obs(self):
        st = self.latest_state or dict(
            character_center=None, finish_center=None,
            wall_detections=[], trap_detections=[], done=False,
            floor_trap_detections=[], laser_detections=[], spinner_detections=[]
        )
        safe = lambda p: np.array(p) if p is not None else np.zeros(2)
        char = safe(st["character_center"])
        fin  = safe(st["finish_center"]) 
        
       
        target_dir = fin - char if char.any() and fin.any() else np.zeros(2)
        target_dir_normalized = target_dir / (np.linalg.norm(target_dir) if np.linalg.norm(target_dir) > 0 else 1)

       
        wall_features = []
        for rx, ry, rw, rh in st["wall_detections"]:
            wall_center = np.array([rx + rw / 2, ry + rh / 2])
            rel_pos = wall_center - char
            wall_left = (rx + rw) < char[0]   
            wall_right = rx > char[0]         
            wall_above = (ry + rh) < char[1]  
            wall_below = ry > char[1]         
            wall_features.extend([
                rel_pos[0] / self.w, rel_pos[1] / self.h,
                float(wall_left), float(wall_right),
                float(wall_above), float(wall_below)
            ])
        
        
        max_walls = 3
        wall_features = wall_features[:max_walls * 6] if len(wall_features) > max_walls * 6 else wall_features + [0.0] * (max_walls * 6 - len(wall_features))

       
        path_blocked = 0.0
        if char.any() and fin.any():
            path_vector = fin - char
            for wall in st["wall_detections"]:
                wx, wy, ww, wh = wall
                if pt_line_dist(np.array([wx + ww / 2, wy + wh / 2]), char, fin) < 50:
                    path_blocked = 1.0
                    break

        
        danger_features = []
        for trap_type in [st["floor_trap_detections"], st["laser_detections"], st["spinner_detections"]]:
            closest = None
            min_dist = float('inf')
            if char.any():
                for t in trap_type[:1]:
                    tx, ty, tw, th = t
                    t_center = np.array([tx + tw / 2, ty + th / 2])
                    dist = np.linalg.norm(t_center - char)
                    if dist < min_dist:
                        closest = t_center
                        min_dist = dist
            if closest is not None:
                danger_features.extend([closest[0] / self.w, closest[1] / self.h, min_dist / 1000.0])
            else:
                danger_features.extend([0.0, 0.0, 1.0])

        return np.concatenate([
            np.array([
                char[0] / self.w, char[1] / self.h,
                fin[0] / self.w, fin[1] / self.h,
                target_dir_normalized[0], target_dir_normalized[1],
                len(st["wall_detections"]) / 10.0,
                (len(st["floor_trap_detections"]) + len(st["laser_detections"]) + len(st["spinner_detections"])) / 10.0,
                self.total / 1000.0,
                path_blocked
            ]),
            np.array(wall_features),
            np.array(danger_features)
        ], dtype=np.float32).flatten()


class CustomGameEnv(Env):
    def __init__(self):
        self.game_env = RLGameEnv(  
            title="AI_3rd (64-bit Development PCD3D_SM5)",
            model_path=r"runs\obb\train3\weights\best.pt"
        )
        self.action_space = Discrete(len(RLGameEnv.ACTION_SPACE))  
        self.observation_space = Box(  
            low=-np.inf, 
            high=np.inf, 
            shape=(self.game_env.OBS_DIM,),  
            dtype=np.float32
        )

    def step(self, action):
        obs, r, done, info = self.game_env.step(RLGameEnv.ACTION_SPACE[action])
        return obs, r, done, False, info  

    def reset(self, seed=None, options=None):
        obs = self.game_env.reset()
        return obs, {}  

    def render(self, mode='human'):
        pass  

    def close(self):
        self.game_env.closed = True


class StepLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_counter = 0

    def _on_step(self) -> bool:
        self.step_counter += 1
        print(f"Current step: {self.step_counter}", end='\r')
        return True 
    


class FullCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, config, seed, yolov5_path=None, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.config = config
        self.seed = seed
        self.yolov5_path = yolov5_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            env = self.training_env
            if hasattr(env, 'venv'):
                env = env.venv

            if hasattr(env, 'envs'):
                env = env.envs[0]
            while hasattr(env, 'env'):
                env = env.env
            monitor = getattr(env, 'game_env', env)

           
            reward_logger = monitor.reward.log

          
            raw_state = monitor.latest_state

            serializable_state = {}
            for key, val in raw_state.items():
                if isinstance(val, np.ndarray):
                    serializable_state[key] = val.tolist()
                else:
                    serializable_state[key] = val

            state_snapshot = {
                'latest_state': serializable_state,
                'total_reward': monitor.total,
                'crossed_traps': list(monitor.reward.crossed_traps),
                'best_distance': monitor.reward.best_distance
            }
            save_full_checkpoint(
                model=self.model,
                step=self.num_timesteps,
                config=self.config,
                seed=self.seed,
                reward_logger=reward_logger,
                state_snapshot=state_snapshot,
                yolov5_path=self.yolov5_path
            )

        return True



def save_full_checkpoint(
    model,
    step,
    config,
    seed,
    reward_logger=None,
    state_snapshot=None,
    yolov5_path=None
):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"logs/checkpoint_step_spinner_{step}_{ts}"
    os.makedirs(base, exist_ok=True)

    
    model.save(f"{base}/ppo_model2.zip")

   
    with open(f"{base}/parameters.pkl", "wb") as f:
        torch.save(model.get_parameters(), f)

  
    with open(f"{base}/config.json", "w") as f:
        json.dump(config, f, indent=2)

   
    with open(f"{base}/seeds.json", "w") as f:
        json.dump(seed, f, indent=2)

   
    if reward_logger is not None:
        reward_logger.export(f"{base}/reward_logs.json")

    
    if state_snapshot is not None:
        with open(f"{base}/env_state.json", "w") as f:
            json.dump(
                state_snapshot,
                f,
                indent=2,
                default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o
            )


   
    if yolov5_path and os.path.exists(yolov5_path):
        shutil.copy2(yolov5_path, f"{base}/yolo_model.pt")

    
    if reward_logger is not None:
        save_reward_plot(
            log_path=f"{base}/reward_logs.json",
            save_path=f"{base}/reward_plot.png"
        )

    print(f"✅ Tam yedekleme kaydedildi: {base}")


def save_reward_plot(log_path='reward_logs.json', save_path='reward_plot.png'):
    with open(log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    steps = [entry["step"] for entry in data]
    rewards = [entry["reward"] for entry in data]
    cumulative = [entry["cumulative_reward"] for entry in data]

    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.plot(steps, rewards, label="Reward per Step")
    plt.legend()
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(steps, cumulative, label="Cumulative Reward", color='orange')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Reward grafiği kaydedildi: {save_path}")


def train_sb3(checkpoint_path="ppo_sb3_final.zip"):
   
    SEED = dict(torch_seed=42, numpy_seed=123, random_seed=777)
    torch.manual_seed(SEED["torch_seed"])
    np.random.seed(SEED["numpy_seed"])

    CONFIG = dict(
        learning_rate=0.0003,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.001,      
        n_epochs=10,
        batch_size=64
    )

    
    env = CustomGameEnv()
    if os.path.exists(checkpoint_path):
        print(f"♻️ Checkpoint bulundu → {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        print("🆕 Yeni model başlatılıyor.")
        model = PPO(
            "MlpPolicy", env,
            learning_rate=CONFIG["learning_rate"],
            n_steps=2048,
            batch_size=CONFIG["batch_size"],
            n_epochs=CONFIG["n_epochs"],
            gamma=CONFIG["gamma"],
            gae_lambda=0.95,
            clip_range=CONFIG["clip_range"],
            ent_coef=CONFIG["ent_coef"],
            verbose=1,
            device="cpu",
            tensorboard_log="./ppo_tensorboard/"  
        )
   
    yolov5_model_path = r"runs\obb\train3\weights\best.pt"
    full_ckpt_callback = FullCheckpointCallback(
        save_freq=10000,
        config=CONFIG,
        seed=SEED,
        yolov5_path=yolov5_model_path
    )
    step_logger = StepLoggerCallback()

   
    model.learn(
        total_timesteps=32768,
        callback=[full_ckpt_callback, step_logger],
        progress_bar=True
    )

    model.save("ppo_sb3_finalv4_spinner.zip")
    print("\n✅ Eğitim tamamlandı → ppo_sb3_final.zip")

if __name__ == "__main__":
    train_sb3() 
