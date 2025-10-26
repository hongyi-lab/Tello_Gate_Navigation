# -*- coding: utf-8 -*-
"""
Tello Gate Navigation — YOLO + RED + Candidate Planner (anchors) + NMS + TTC trigger
- 在原脚本上增量加入：
  1) 候选轨迹锚点网格（Mφ×Mθ）生成、评分选优、NMS 留存 #1/#2，并允许人工切换“备选方案”。
  2) 视频叠加展示全部候选（颜色=代价，#1=绿色粗线，#2=黄色粗线、虚线），并在 HUD 显示当前选用方案。
  3) 引入 TTC 触发：TTC = A / dA/dt，当 TTC < 阈值时触发穿门承诺态（更稳）；仍保留面积阈值+dA>0 判据。
  4) 在不引入姿态/推力接口与 ESDF 的情况下，用图像域启发式代价近似 YOPOv2 的“安全/平滑/目标对齐”思想。
参照 YOPO/YOPOv2 的锚点与一阶段规划理念（图3/4/5/6、式(5)、NMS 选优等）。详见论文 PDF。  :contentReference[oaicite:2]{index=2}
保持原有的 GateCrossSM、ForwardBurstOnce、Controller 等工程骨架与飞控接口不变。                  :contentReference[oaicite:3]{index=3}
"""
#toggle_overlay / toggle_backup：按键开关显示与备选使用
#c：开/关候选轨迹叠加
#b：备选方案切换（Top2）开/关（HUD 会显示 USE=ALT#2）

'''
先在室内测试：调小 FWD_NEAR_BOOST 与 TARGET_FWD；看 HUD 的 TTC 与 Cands: TOP1/TOP2 是否稳定。

若抖动：增大 EMA_ALPHA_ERR 的系数（更慢更稳），或提升 CAND_WSM（加强平滑代价），或略放宽 CAND_NMS_PX。

换 RGB‑D：把安全代价换成 ESDF 距离梯度，代价/选优机制不用改，直接升级到“可微可训练”的版式。

10.3proversion

'''
import os, time, cv2, numpy as np, torch, unicodedata, math
from dataclasses import dataclass
from ultralytics import YOLO
from djitellopy import Tello
from djitellopy.tello import TelloException

# =========================
# Config（集中所有可调参数）
# =========================
@dataclass
class Config:
    # Model/runtime
    MODEL_PATH: str = "bestBy11s.pt"
    IMGSZ: int = 640
    CONF: float = 0.64
    IOU: float = 0.45
    MAX_DET: int = 10

    # Frame & normalization
    FRAME_W: int = 960
    FRAME_H: int = 720

    # Error processing
    ERR_DEAD_BAND: float = 0.04
    EMA_ALPHA_ERR: float = 0.35
    EMA_ALPHA_AREA: float = 0.30

    # Control gains (相机坐标 -> RC)
    K_YAW: float  = 55.0
    K_ROLL_NEAR: float = 28.0
    K_ROLL_FAR: float  = 14.0
    K_THR: float   = 52.0
    TARGET_FWD: int = 20
    FWD_NEAR_BOOST: int = 28
    NEAR_GATE_AREA_FRAC: float = 0.15

    # Candidate planner (anchors) —— 新增
    CAND_M_THETA: int = 5        # 水平网格列数（Mθ）
    CAND_M_PHI: int   = 3        # 垂直网格行数（Mφ）
    CAND_WG: float    = 1.00     # 目标对齐代价权重 Jg
    CAND_WEDGE: float = 0.30     # 贴边惩罚权重（避免靠近画面边界丢失）
    CAND_WSM: float   = 0.20     # 平滑代价（与上次选中的锚点差异）
    CAND_WBOX: float  = -0.25    # 在门框 bbox 内的奖励（负号=减小代价）
    CAND_NMS_PX: int  = 120      # NMS 的最小像素间距（#2 需与 #1 保持一定差异）
    CAND_DRAW: bool   = True     # 是否叠加显示候选
    CAND_MAX_DRAW: int = 64      # 控制绘制数量（节省开销）

    # TTC 触发 —— 新增
    TTC_TRIGGER_SEC: float = 0.70  # 当 TTC < 此值触发穿门承诺态
    TTC_MIN_DAREA: float = 1e-5    # 计算 TTC 时的最小面积增量，避免除零

    # Search 策略
    SEARCH_YAW_DPS: int = 16
    SEARCH_FWD: int = 6

    # RC & safety
    MAX_CMD: int = 100
    RC_RATE_HZ: float = 20.0
    STICKY_MAX_FRAMES: int = 20
    STICKY_DECAY: float = 0.90

    # Gate-cross（穿门）策略
    Z_OFFSET_CM: int = 30
    Z_OFFSET_SPEED: int = 28
    FORWARD_BURST_CM: int = 75
    FORWARD_BURST_MIN_SPEED: int = 28
    CROSS_HOLD_SEC: float = 1.0
    CROSS_YAW_LIMIT: int = 15


        # =========================
    # Top-down Map (Bird's-eye)
    # =========================
    MAP_ON: bool = True           # 开关：启动就弹出地图窗口
    MAP_SIZE: int = 700           # 地图画布大小（像素，方形）
    MAP_PX_PER_M: int = 80        # 比例尺：1 m = 80 px（可按实测改）
    CAM_FOV_H_DEG: float = 82.6   # Tello 水平视场角（近似）
    GATE_W_M: float = 0.8         # gate 实际宽度（米，按你的框尺寸改）
    TRAIL_MAX_N: int = 2000       # 轨迹最多保存的点数


CFG = Config()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF = (torch.cuda.is_available() and DEVICE.startswith("cuda"))
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# RC sender with rate-limit
# =========================
_last_rc_sent = (0,0,0,0)
_last_rc_time = 0.0
def clamp(x, lo, hi): return max(lo, min(hi, int(x)))

def safe_send_rc(tello: Tello, r:int, p:int, z:int, y:int, tag:str=""):
    """保持与原版一致：限速率 + 去重发送 + 统一打印。"""
    global _last_rc_sent, _last_rc_time
    r, p, z, y = (clamp(r,-CFG.MAX_CMD,CFG.MAX_CMD),
                  clamp(p,-CFG.MAX_CMD,CFG.MAX_CMD),
                  clamp(z,-CFG.MAX_CMD,CFG.MAX_CMD),
                  clamp(y,-CFG.MAX_CMD,CFG.MAX_CMD))
    now = time.time()
    min_interval = 1.0/CFG.RC_RATE_HZ
    if (now - _last_rc_time) < min_interval and (r,p,z,y)==_last_rc_sent:
        return
    try:
        tello.send_rc_control(r,p,z,y)
        _last_rc_sent = (r,p,z,y)
        _last_rc_time = now
        print(f"[RC]{'['+tag+']' if tag else ''} r={r:+d} p={p:+d} z={z:+d} y={y:+d}")
    except Exception as e:
        print("[WARN] rc_control error:", e)

# =========================
# YOLO / Gate pick
# =========================
def auto_gate_ids_by_name(model) -> set[int]:
    return {i for i, name in model.names.items() if 'gate' in str(name).lower()}

def pick_gate_bbox(results, allow_ids: set[int] | None):
    boxes = results[0].boxes if results else None
    if boxes is None or len(boxes) == 0: return None
    best, best_area = None, 0.0
    for b in boxes:
        xyxy = b.xyxy[0].detach().cpu().numpy()
        cls = int(b.cls[0].item()) if b.cls is not None else -1
        if allow_ids is not None and cls not in allow_ids: continue
        x1,y1,x2,y2 = xyxy
        area = max(0.0, x2-x1) * max(0.0, y2-y1)
        if area > best_area:
            best_area, best = area, xyxy
    return best

# =========================
# RED helper（与原版一致）
# =========================
LOWER_RED1 = np.array([0, 80, 80], dtype=np.uint8)
UPPER_RED1 = np.array([10, 255,255], dtype=np.uint8)
LOWER_RED2 = np.array([170, 80, 80], dtype=np.uint8)
UPPER_RED2 = np.array([179,255,255], dtype=np.uint8)
KERNEL = np.ones((3,3), np.uint8)
MIN_GATE_AREA_FRAC = 0.01
AR_MIN, AR_MAX = 0.4, 2.5

def detect_red_bbox_rgb(frame_rgb):
    h, w = frame_rgb.shape[:2]
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    m2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    img_area = float(w*h)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        area = cv2.contourArea(c)
        if area / max(1.0, img_area) < MIN_GATE_AREA_FRAC: continue
        x,y,bw,bh = cv2.boundingRect(c)
        ar = bh / max(1.0, bw)
        if not (AR_MIN <= ar <= AR_MAX): continue
        cx, cy = int(x + bw/2), int(y + bh/2)
        return {"ok": True, "bbox": (x, y, x+bw, y+bh), "center": (cx, cy), "area_frac": area / img_area}
    return None

# =========================
# Filters / helpers
# =========================
class EMA:
    def __init__(self, alpha:float, init:float=0.0):
        self.a = float(alpha); self.v = float(init); self.initd=False
    def update(self, x:float):
        if not self.initd:
            self.v = float(x); self.initd=True
        else:
            self.v = self.a*float(x) + (1.0-self.a)*self.v
        return self.v

def norm_errors(center_xy, frame_w, frame_h):
    cx_img, cy_img = frame_w//2, frame_h//2
    cx, cy = center_xy
    ex = (cx - cx_img) / (frame_w * 0.5)
    ey = (cy - cy_img) / (frame_h * 0.5)
    return float(ex), float(ey)

# =========================
# Sticky rc（原版）
# =========================
class StickyRC:
    def __init__(self):
        self.rc = (0,0,0,0,(0.0,0.0,0.0)); self.frames = 0; self.valid = False
    def set(self, rc_tuple):
        self.rc = rc_tuple; self.frames = 0; self.valid = True
    def lost_tick_and_get(self):
        if not self.valid: return None
        self.frames += 1
        if self.frames > CFG.STICKY_MAX_FRAMES:
            self.valid = False; return None
        r,p,t,y,pos = self.rc
        dec = (CFG.STICKY_DECAY ** self.frames)
        r = clamp(int(r * dec), -CFG.MAX_CMD, CFG.MAX_CMD)
        p = clamp(int(p * dec), -CFG.MAX_CMD, CFG.MAX_CMD)
        t = clamp(int(t * dec), -CFG.MAX_CMD, CFG.MAX_CMD)
        y = clamp(int(y * dec), -CFG.MAX_CMD, CFG.MAX_CMD)
        return (r,p,t,y,pos)

# =========================
# One-shot Z offset & Forward burst（原版）
# =========================
class ZOffsetOnce:
    def __init__(self):
        self.active = False; self.end_time = 0.0
    def trigger(self):
        dur = CFG.Z_OFFSET_CM / float(max(1, CFG.Z_OFFSET_SPEED))
        self.active = True; self.end_time = time.time() + dur
        print(f"[Z-OFFSET] start: -{CFG.Z_OFFSET_CM}cm at {CFG.Z_OFFSET_SPEED}cm/s for {dur:.2f}s")
    def bias(self):
        if not self.active: return 0
        if time.time() >= self.end_time:
            self.active = False; print("[Z-OFFSET] done"); return 0
        return -int(CFG.Z_OFFSET_SPEED)  # down negative

class ForwardBurstOnce:
    def __init__(self):
        self.active=False; self.end_time=0.0; self.speed=0
    def trigger(self, current_pitch_cm_s:int):
        v = max(abs(int(current_pitch_cm_s)), CFG.FORWARD_BURST_MIN_SPEED, 10)
        dur = CFG.FORWARD_BURST_CM / float(v)
        self.speed = v; self.end_time = time.time() + dur; self.active=True
        print(f"[FWD-BURST] start: +{CFG.FORWARD_BURST_CM}cm at {v}cm/s for {dur:.2f}s")
    def bias(self):
        if not self.active: return 0
        if time.time() >= self.end_time:
            self.active=False; print("[FWD-BURST] done"); return 0
        return int(self.speed)  # forward positive

# =========================
# Gate crossing state machine（原版）
# =========================
class GateCrossSM:
    """进入后在短时内“坚持穿门”：限小yaw + 强前进 + 可叠加下压"""
    def __init__(self): self.active=False; self.until=0.0
    def start(self, hold_sec:float):
        self.active=True; self.until = time.time() + float(hold_sec)
        print(f"[CROSS] commit for {hold_sec:.2f}s")
    def step(self):
        if self.active and time.time() >= self.until:
            self.active=False; print("[CROSS] done")
        return self.active

# =========================
# Controller（原版，增加一个 compute_to_point）
# =========================
class Controller:
    def __init__(self, w:int, h:int):
        self.w=w; self.h=h
        self.ex_ema=EMA(CFG.EMA_ALPHA_ERR, 0.0)
        self.ey_ema=EMA(CFG.EMA_ALPHA_ERR, 0.0)
        self.af_ema=EMA(CFG.EMA_ALPHA_AREA, 0.0)
        self.ex_prev=0.0; self.ey_prev=0.0; self.t_prev=time.time()

    def _core(self, ex, ey, area_frac, near_gate:bool, has_gate:bool):
        if not has_gate:
            return 0, CFG.SEARCH_FWD, 0, CFG.SEARCH_YAW_DPS, (0.0,0.0,0.0)

        # deadband + EMA
        if abs(ex) < CFG.ERR_DEAD_BAND: ex = 0.0
        if abs(ey) < CFG.ERR_DEAD_BAND: ey = 0.0
        ex_f = self.ex_ema.update(ex); ey_f = self.ey_ema.update(ey)
        af_f = self.af_ema.update(max(0.0, float(area_frac)))

        # 简单 PD（微分取 EMA 后的有限差分）
        now = time.time(); dt = max(1e-3, now - self.t_prev)
        dex = (ex_f - self.ex_prev) / dt
        dey = (ey_f - self.ey_prev) / dt
        self.ex_prev, self.ey_prev, self.t_prev = ex_f, ey_f, now

        yaw = clamp(int(CFG.K_YAW * ex_f + 0.0*dex), -CFG.MAX_CMD, CFG.MAX_CMD)
        kroll = CFG.K_ROLL_NEAR if near_gate else CFG.K_ROLL_FAR
        roll = clamp(int(kroll * ex_f), -CFG.MAX_CMD, CFG.MAX_CMD)
        thr  = clamp(int(-CFG.K_THR * ey_f + 0.0*dey), -CFG.MAX_CMD, CFG.MAX_CMD)
        pitch_ff = CFG.TARGET_FWD + (CFG.FWD_NEAR_BOOST if near_gate else 0)
        pitch = clamp(int(pitch_ff), -CFG.MAX_CMD, CFG.MAX_CMD)
        return roll, pitch, thr, yaw, (ex_f, ey_f, af_f)

    def compute(self, center_xy, area_frac, near_gate:bool, has_gate:bool):
        # 仍保留：以“门中心”为控制目标
        ex, ey = norm_errors(center_xy, self.w, self.h)
        return self._core(ex, ey, area_frac, near_gate, has_gate)

    def compute_to_point(self, target_xy, area_frac, near_gate:bool, has_gate:bool):
        # 新增：以“候选锚点”像素为控制目标（替代直接瞄准门中心）
        ex, ey = norm_errors(target_xy, self.w, self.h)
        return self._core(ex, ey, area_frac, near_gate, has_gate)

# =========================
# Candidate planner（锚点网格 + 代价评分 + NMS + 可视化）—— 新增
# =========================
class CandidatePlanner:
    """
    在图像上均匀生成 Mφ×Mθ 个锚点（anchors），基于：
    - 目标对齐 Jg：锚点与门中心像素误差（越小越好）
    - 贴边惩罚 Jedge：离画面边缘越近惩罚越大
    - 平滑 Jsm：与上次选中锚点的像素差（希望连续帧变化小）
    - BBox 奖励 Jbox：若锚点落在门框 bbox 内，减小代价
    综合代价：J = w_g*Jg + w_edge*Jedge + w_sm*Jsm + w_box*Jbox
    选出 #1/#2，#2 需与 #1 像素距离 >= NMS_PX（保持备选多样性）。
    叠加到画面上显示所有候选与 Top1/Top2（颜色=代价大小）。
    """
    def __init__(self, w:int, h:int):
        self.w, self.h = w, h
        self.grid_uv = self._make_grid(w, h, CFG.CAND_M_THETA, CFG.CAND_M_PHI)
        self.prev_sel_uv = None  # 上一帧选中锚点，用于平滑代价
        self.use_backup = False  # 是否启用备选方案
        self.overlay_on = CFG.CAND_DRAW
        self.last_top = None     # (u,v,J) of best
        self.last_top2 = None

    @staticmethod
    def _make_grid(w, h, m_theta, m_phi):
        us = [int((j+0.5)/m_theta * w) for j in range(m_theta)]
        vs = [int((i+0.5)/m_phi   * h) for i in range(m_phi)]
        pts = [(u,v) for v in vs for u in us]  # 行优先/列优先均可
        return pts

    def _edge_penalty(self, u, v):
        # 距边缘越近，惩罚越大。margin 取 12% 画面宽高
        margin_x = 0.12 * self.w
        margin_y = 0.12 * self.h
        dx = min(u, self.w - u); dy = min(v, self.h - v)
        px = max(0.0, (margin_x - dx) / margin_x)
        py = max(0.0, (margin_y - dy) / margin_y)
        return (px*px + py*py)

    def _smooth_penalty(self, u, v):
        if self.prev_sel_uv is None: return 0.0
        du = (u - self.prev_sel_uv[0]) / float(self.w)
        dv = (v - self.prev_sel_uv[1]) / float(self.h)
        return (du*du + dv*dv)

    def _goal_cost(self, u, v, gate_cx, gate_cy):
        ex = (u - gate_cx) / (0.5*self.w)
        ey = (v - gate_cy) / (0.5*self.h)
        return (ex*ex + ey*ey)

    def _bbox_bonus(self, u, v, bbox):
        if bbox is None: return 0.0
        x1,y1,x2,y2 = bbox
        # 给 bbox 内的点一个奖励（减小代价），边界收缩一点避免抖动
        shrink = 8
        x1+=shrink; y1+=shrink; x2-=shrink; y2-=shrink
        if x1 < u < x2 and y1 < v < y2:
            return 1.0  # 由权重 CAND_WBOX 乘该项
        return 0.0

    def _score_all(self, gate_center, bbox):
        # 返回 [(u,v,J), ...] 按代价升序
        if gate_center is None: return []
        gate_cx, gate_cy = gate_center
        scored = []
        for (u,v) in self.grid_uv:
            Jg = self._goal_cost(u,v, gate_cx,gate_cy)
            Je = self._edge_penalty(u,v)
            Js = self._smooth_penalty(u,v)
            Jb = self._bbox_bonus(u,v, bbox)
            J = (CFG.CAND_WG*Jg + CFG.CAND_WEDGE*Je + CFG.CAND_WSM*Js + CFG.CAND_WBOX*Jb)
            scored.append((u,v,J))
        scored.sort(key=lambda x: x[2])
        return scored

    @staticmethod
    def _far_enough(p, q, thr):
        return (abs(p[0]-q[0])>thr) or (abs(p[1]-q[1])>thr)

    def select(self, gate_center, bbox):
        ranked = self._score_all(gate_center, bbox)
        if not ranked:
            self.last_top, self.last_top2 = None, None
            return None, None, []

        top1 = ranked[0]
        top2 = None
        for cand in ranked[1:]:
            if self._far_enough((top1[0],top1[1]), (cand[0],cand[1]), CFG.CAND_NMS_PX):
                top2 = cand; break

        self.last_top, self.last_top2 = top1, top2
        return top1, top2, ranked[:CFG.CAND_MAX_DRAW]

    def toggle_overlay(self):
        self.overlay_on = not self.overlay_on
        print(f"[CAND] overlay {'ON' if self.overlay_on else 'OFF'}")

    def toggle_backup(self):
        self.use_backup = not self.use_backup
        print(f"[CAND] use backup {'ON' if self.use_backup else 'OFF'}")

    def commit_prev(self, uv):
        self.prev_sel_uv = uv

    def draw(self, img_rgb, ranked, top1, top2):
        if not self.overlay_on or ranked is None: return
        h, w = img_rgb.shape[:2]
        cx, cy = w//2, h//2

        # 归一化代价 -> 颜色映射（代价低=绿色，高=红色）
        if len(ranked)>0:
            Js = [r[2] for r in ranked]
            jmin, jmax = float(min(Js)), float(max(Js))
            jspan = max(1e-6, jmax - jmin)
        else:
            jmin, jspan = 0.0, 1.0

        def color_for(J):
            t = (J - jmin)/jspan
            # 绿色->黄色->红色渐变
            r = int(255 * t)
            g = int(255 * (1.0 - 0.5*t))
            b = 0
            return (r,g,b)

        # 画所有候选
        for (u,v,J) in ranked:
            col = color_for(J)
            cv2.line(img_rgb, (cx,cy), (int(u),int(v)), col, 1, lineType=cv2.LINE_AA)

        # 强调 Top1 / Top2
        if top1 is not None:
            u,v,J = top1
            cv2.line(img_rgb, (cx,cy), (int(u),int(v)), (0,255,0), 3, lineType=cv2.LINE_AA)
            cv2.circle(img_rgb, (int(u),int(v)), 6, (0,255,0), -1)
            cv2.putText(img_rgb, "BEST#1", (int(u)+6, int(v)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if top2 is not None:
            u,v,J = top2
            # 虚线实现（多段小线）
            segs = 16
            for i in range(segs):
                if i%2==0:
                    p0 = (int(cx + (u-cx)*i/segs),   int(cy + (v-cy)*i/segs))
                    p1 = (int(cx + (u-cx)*(i+1)/segs), int(cy + (v-cy)*(i+1)/segs))
                    cv2.line(img_rgb, p0, p1, (0,255,255), 3, lineType=cv2.LINE_AA)
            cv2.circle(img_rgb, (int(u),int(v)), 6, (0,255,255), -1)
            cv2.putText(img_rgb, "ALT#2", (int(u)+6, int(v)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            

class TopDownMap:
    """
    极简 2D 顶视图：
    - 以 RC 命令积分得到 (x,y,psi)，单位 m、deg（p≈前进速度cm/s，y≈yaw速deg/s）。
    - 用 bbox 宽度 + 相机水平FOV 估距（针孔模型），把 gate 投到世界坐标。
    - 叠加候选锚点方向射线，用于对比“控制朝向 vs 目标方位”。
    """
    def __init__(self, frame_w:int, frame_h:int):
        self.sz = CFG.MAP_SIZE
        self.ppm = CFG.MAP_PX_PER_M
        self.show = CFG.MAP_ON

        self.x_m = 0.0   # 世界坐标（起点为(0,0)）
        self.y_m = 0.0
        self.psi_deg = 0.0
        self.last_t = time.time()

        self.frame_w = frame_w
        self.frame_h = frame_h
        self.fx_px = 0.5 * frame_w / math.tan(math.radians(CFG.CAM_FOV_H_DEG)*0.5)

        self.trail = []          # [(x_m,y_m), ...]
        self.gates_world = []    # [(x_m,y_m,t), ...] 仅保留最近若干个
        self.gates_keep = 12

        # 视图中心（像素），用于平移/重置
        self.cx = self.sz//2
        self.cy = self.sz//2

        # 最近一次的候选方向（弧度），画射线
        self.cand_bearings = []  # [(bearing_rad, color), ...]
        self.cand_len_m = 2.5

    def toggle(self):
        self.show = not self.show
        print(f"[MAP] {'ON' if self.show else 'OFF'}")

    def recenter(self):
        self.cx = self.sz//2
        self.cy = self.sz//2
        print("[MAP] recentered")

    def clear_trail(self):
        self.trail.clear()
        print("[MAP] trail cleared")

    # ========== 简单里程计：用 RC 命令近似积分 ==========
    def step_odometry(self, rc_tuple, dt):
        # rc: (r, p, z, y) 其中 p≈cm/s 前向速度，y≈deg/s 偏航角速度
        r, p, z, y = rc_tuple
        v_ms = float(p) * 0.01      # cm/s -> m/s
        self.psi_deg += float(y) * dt
        psi = math.radians(self.psi_deg)
        # 只用前向速度积分（可选把横滚 r 当侧向速度，这里先省略）
        self.x_m += v_ms * math.cos(psi) * dt
        self.y_m += v_ms * math.sin(psi) * dt

        self.trail.append((self.x_m, self.y_m))
        if len(self.trail) > CFG.TRAIL_MAX_N:
            self.trail = self.trail[-CFG.TRAIL_MAX_N:]

    # ========== 视觉量测 -> 世界坐标 ==========
    def observe_gate_from_bbox(self, bbox, center_xy):
        """
        bbox: (x1,y1,x2,y2) in px；center_xy: (cx,cy) in px（图像坐标）
        用 bbox 宽度估距： Z ≈ fx * W / wpx
        bearing（相对机头）≈ atan((u - u0)/fx)
        """
        if bbox is None or center_xy is None: return
        x1,y1,x2,y2 = bbox
        u0 = self.frame_w/2.0
        bw = max(1.0, (x2 - x1))
        u, _ = center_xy

        Z_m = (self.fx_px * CFG.GATE_W_M) / bw
        yaw_off = math.atan((u - u0) / self.fx_px)      # 相机/机体坐标内的水平偏角（弧度）
        bearing = math.radians(self.psi_deg) + yaw_off  # 世界坐标系中的方位角

        gx = self.x_m + Z_m * math.cos(bearing)
        gy = self.y_m + Z_m * math.sin(bearing)
        self.gates_world.append((gx, gy, time.time()))
        if len(self.gates_world) > self.gates_keep:
            self.gates_world = self.gates_world[-self.gates_keep:]

    def observe_candidates(self, top1, top2):
        """把 Top1/Top2 锚点像素转成方位角（仅方向，用固定长度画射线）"""
        self.cand_bearings.clear()
        for top, col in [(top1,(0,255,0)), (top2,(0,255,255))]:
            if top is None: continue
            u,v,J = top
            u0 = self.frame_w/2.0
            yaw_off = math.atan((u - u0) / self.fx_px)
            bearing = math.radians(self.psi_deg) + yaw_off
            self.cand_bearings.append((bearing, col))

    # ========== 绘制 ==========
    def _w2p(self, x_m, y_m):
        """世界坐标 -> 画布像素（y 轴向上为正）"""
        px = int(self.cx + x_m * self.ppm)
        py = int(self.cy - y_m * self.ppm)
        return (px, py)

    def _draw_grid(self, img):
        step_px = self.ppm  # 每 1 m 一格
        for k in range(0, self.sz, step_px):
            cv2.line(img, (k,0), (k,self.sz), (40,40,40), 1)
            cv2.line(img, (0,k), (self.sz,k), (40,40,40), 1)
        # 中心十字
        cv2.line(img, (self.cx,0), (self.cx,self.sz), (80,80,80), 1)
        cv2.line(img, (0,self.cy), (self.sz,self.cy), (80,80,80), 1)

    def render(self):
        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        self._draw_grid(img)

        # 画轨迹
        if len(self.trail) >= 2:
            for i in range(1, len(self.trail)):
                p0 = self._w2p(*self.trail[i-1])
                p1 = self._w2p(*self.trail[i])
                cv2.line(img, p0, p1, (90,90,200), 2)

        # 画 gate 估计点
        for (gx, gy, t) in self.gates_world:
            gp = self._w2p(gx, gy)
            cv2.circle(img, gp, 6, (255,255,255), -1)
            cv2.circle(img, gp, 12, (100,100,100), 1)

        # 画候选方向射线（以机体位置为起点）
        origin = self._w2p(self.x_m, self.y_m)
        for bearing, col in self.cand_bearings:
            end = (self.x_m + self.cand_len_m * math.cos(bearing),
                   self.y_m + self.cand_len_m * math.sin(bearing))
            endp = self._w2p(*end)
            cv2.arrowedLine(img, origin, endp, col, 3, tipLength=0.12)

        # 画无人机：三角机头
        psi = math.radians(self.psi_deg)
        nose = (self.x_m + 0.25*math.cos(psi), self.y_m + 0.25*math.sin(psi))
        left = (self.x_m + 0.12*math.cos(psi+2.6), self.y_m + 0.12*math.sin(psi+2.6))
        right= (self.x_m + 0.12*math.cos(psi-2.6), self.y_m + 0.12*math.sin(psi-2.6))
        pts = np.array([self._w2p(*nose), self._w2p(*left), self._w2p(*right)], np.int32)
        cv2.fillConvexPoly(img, pts, (0,180,255))
        cv2.polylines(img, [pts], True, (0,0,0), 2)

        # HUD
        cv2.putText(img, f"(x,y)=({self.x_m:+.1f}m,{self.y_m:+.1f}m) psi={self.psi_deg:+.1f}deg",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        cv2.putText(img, "Keys: g map on/off | o recenter | u clear trail",
                    (10, self.sz-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
        return img

# =========================
# Drawing HUD（原版+少量信息增加）
# =========================
def bgr_to_rgb(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def rgb_to_bgr(img): return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
def col_rgb(r,g,b): return (r,g,b)

def draw_hud_rgb(frame_rgb, red_result, yolo_result, rc_tuple, pos_err, sm_active, af_raw, cand_info_text):
    h, w = frame_rgb.shape[:2]
    cx_img, cy_img = w//2, h//2
    cv2.circle(frame_rgb, (cx_img, cy_img), 6, col_rgb(255,255,255), -1)

    if red_result and red_result.get("ok", False):
        x1,y1,x2,y2 = map(int, red_result["bbox"])
        cx, cy = red_result["center"]; af = red_result["area_frac"]
        cv2.rectangle(frame_rgb, (x1,y1), (x2,y2), col_rgb(0,255,0), 2)
        cv2.circle(frame_rgb, (cx,cy), 7, col_rgb(0,0,255), -1)
        cv2.putText(frame_rgb, f"RED area={af:.3f}", (x1, max(0,y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_rgb(255,255,0), 2)

    if yolo_result and yolo_result.get("ok", False):
        x1,y1,x2,y2 = map(int, yolo_result["bbox"])
        cx, cy = yolo_result["center"]; af = yolo_result["area_frac"]
        cv2.rectangle(frame_rgb, (x1,y1), (x2,y2), col_rgb(255,0,255), 2)
        cv2.circle(frame_rgb, (cx,cy), 7, col_rgb(255,0,255), -1)
        cv2.putText(frame_rgb, f"YOLO area={af:.3f}", (x1, min(h-5,y2+18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_rgb(255,255,255), 2)

    r, p, t, y = rc_tuple
    ex, ey, ez = pos_err
    cv2.putText(frame_rgb, f"rc(r,p,t,y)=({r:+3d},{p:+3d},{t:+3d},{y:+3d})", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_rgb(0,255,255), 2)
    cv2.putText(frame_rgb, f"filt_err(ex,ey,af)=({ex:+.2f},{ey:+.2f},{ez:.3f})  raw_af={af_raw:.3f}", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_rgb(255,255,255), 2)
    cv2.putText(frame_rgb, f"SM_CROSS={'ON' if sm_active else 'off'}  NEAR_THR={CFG.NEAR_GATE_AREA_FRAC:.2f}",
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_rgb(255,255,0), 2)
    if cand_info_text:
        cv2.putText(frame_rgb, cand_info_text, (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_rgb(0,255,0), 2)
    cv2.putText(frame_rgb, "Keys: t takeoff | l land | m auto/manual | q/e yaw | w/s fwd/back | a/d left/right | r/f up/down | x stop |"
                           " c cand overlay | b backup toggle | ESC quit",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_rgb(255,255,255), 1)

# =========================
# Main
# =========================
def main():
    # Load YOLO
    if not os.path.exists(CFG.MODEL_PATH):
        print(f"[WARN] YOLO weights not found: {CFG.MODEL_PATH} — YOLO branch disabled.")
        model = None; gate_ids = None
    else:
        print("[INFO] Loading YOLO model...")
        model = YOLO(CFG.MODEL_PATH)
        try:
            model.to(DEVICE)
            if HALF:
                try: model.model.half()
                except Exception: pass
            print(f"[INFO] Model on {DEVICE} | half={HALF}")
            if DEVICE.startswith("cuda"):
                print(f"[INFO] CUDA: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"[WARN] Failed to move model to {DEVICE}, using CPU. Reason: {e}")
        try:
            _ = model.predict(np.zeros((CFG.IMGSZ, CFG.IMGSZ, 3), np.uint8),
                              imgsz=CFG.IMGSZ, conf=0.01, device=DEVICE,
                              half=HALF, verbose=False)
        except Exception:
            pass
        gate_ids = auto_gate_ids_by_name(model) if model else None

    # Tello
    t = Tello(); print("[INFO] Connecting to Tello...")
    t.connect()
    try:
        batt = t.get_battery()
        print(f"[INFO] Tello connected. Battery: {batt}%")
    except Exception:
        print("[INFO] Tello connected.")
    t.streamon()
    reader = t.get_frame_read()
    time.sleep(0.25)

    cv2.namedWindow("Tello (YOLO+RED+Candidates)", cv2.WINDOW_NORMAL)
    if CFG.MAP_ON:
        cv2.namedWindow("Top-Down Map", cv2.WINDOW_NORMAL)
    topo = TopDownMap(CFG.FRAME_W, CFG.FRAME_H)
    flying=False; auto_mode=True
    yaw_dir = 1; last_yaw_flip = time.time()
    last_rc = (0,0,0,0)
    sticky = StickyRC()
    zoffset = ZOffsetOnce()
    forwardburst = ForwardBurstOnce()
    cross_sm = GateCrossSM()
    manual_hold = (0,0,0,0)
    prev_area = 0.0; prev_area_t = time.time()
    controller = Controller(CFG.FRAME_W, CFG.FRAME_H)
    planner = CandidatePlanner(CFG.FRAME_W, CFG.FRAME_H)
    

    def poll_key():
        try:
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                ch = unicodedata.normalize('NFKC', ch)
                if ch: return ch
        except Exception:
            pass
        k = cv2.waitKey(1)
        if k == -1: return None
        if k == 27: return 'ESC'
        try:
            ch = chr(k & 0xFF)
        except: return None
        ch = unicodedata.normalize('NFKC', ch)
        return ch

    try:
        while True:
            frame_bgr = reader.frame
            if frame_bgr is None:
                time.sleep(0.005); continue
            frame_bgr = cv2.resize(frame_bgr, (CFG.FRAME_W, CFG.FRAME_H))
            frame_rgb = bgr_to_rgb(frame_bgr)
            h,w = frame_rgb.shape[:2]

            # 检测
            red_res = detect_red_bbox_rgb(frame_rgb)
            red_ok = red_res is not None and red_res.get("ok", False)

            yolo_res = None
            if model is not None:
                results = model.predict(frame_rgb, imgsz=CFG.IMGSZ, conf=CFG.CONF, iou=CFG.IOU,
                                        max_det=CFG.MAX_DET, device=DEVICE, half=HALF, verbose=False)
                yolo_bbox = pick_gate_bbox(results, gate_ids)
                if yolo_bbox is not None:
                    x1,y1,x2,y2 = map(int, yolo_bbox)
                    cx, cy = int((x1+x2)//2), int((y1+y2)//2)
                    area_frac = ((x2-x1)*(y2-y1)) / float(w*h)
                    yolo_res = {"ok": True, "bbox": (x1,y1,x2,y2),
                                "center": (cx,cy), "area_frac": area_frac}
                vis = results[0].plot() if results else frame_rgb
                if vis is not frame_rgb: vis = bgr_to_rgb(vis)
            else:
                vis = frame_rgb

            key = poll_key()
            # 额外键位（新增）
            if key == 'c': planner.toggle_overlay()
            if key == 'b': planner.toggle_backup()
            if key == 'g': topo.toggle()
            if key == 'o': topo.recenter()
            if key == 'u': topo.clear_trail()

            # 起降/模式（与原版一致）
            if key == 't' and not flying:
                try:
                    b = t.get_battery()
                    if b is not None and b < 15:
                        print("[WARN] Battery low (<15%).")
                    t.takeoff(); flying=True; auto_mode=False
                    print("[MODE] MANUAL after takeoff (press 'm' for AUTO).")
                except TelloException as e:
                    print(f"[WARN] takeoff rejected: {e}")
                except Exception as e:
                    print(f"[WARN] takeoff error: {e}")

            elif key == 'l' and flying:
                try:
                    safe_send_rc(t,0,0,0,0,"land-stop")
                    t.land(); flying=False
                except Exception as e:
                    print(f"[WARN] land error: {e}")

            elif key == 'm':
                auto_mode = not auto_mode
                print(f"[MODE] {'AUTO' if auto_mode else 'MANUAL'}")

            elif key == 'ESC':
                print("[INFO] Quit"); break

            # ===== MANUAL（与原版一致）=====
            if flying and not auto_mode:
                updated = False
                r,p,z,y = manual_hold
                vel = 25
                if key == 'w': p = +vel; updated=True
                elif key == 's': p = -vel; updated=True
                elif key == 'a': r = -vel; updated=True
                elif key == 'd': r = +vel; updated=True
                elif key == 'r': z = +vel; updated=True
                elif key == 'f': z = -vel; updated=True
                elif key == 'q': y = -vel; updated=True
                elif key == 'e': y = +vel; updated=True
                elif key == 'x': r=p=z=y=0; updated=True

                if updated:
                    manual_hold = (r,p,z,y)
                    last_rc = (r,p,z,y)
                    safe_send_rc(t, r,p,z,y, tag="MANUAL")

            # ===== AUTO =====
            # 用哪个检测源作为“门中心/面积”
            area_for_trigger = 0.0
            center_for_ctrl = None
            bbox_for_bonus = None
            if red_ok:
                center_for_ctrl = red_res["center"]; area_for_trigger = red_res["area_frac"]
                bbox_for_bonus = red_res["bbox"]
            elif yolo_res and yolo_res.get("ok", False):
                center_for_ctrl = yolo_res["center"]; area_for_trigger = yolo_res["area_frac"]
                bbox_for_bonus = yolo_res["bbox"]

            # 近门判断
            near_gate = (area_for_trigger >= CFG.NEAR_GATE_AREA_FRAC)

            # 面积导数与 TTC
            now_t = time.time()
            dt_area = max(1e-3, now_t - prev_area_t)
            d_area = (area_for_trigger - prev_area) / dt_area
            ttc = None
            if d_area > CFG.TTC_MIN_DAREA:
                ttc = area_for_trigger / d_area  # 近似 TTC
            prev_area = area_for_trigger
            prev_area_t = now_t

            # 候选轨迹：只有有目标时才生成/选优
            top1 = top2 = None
            ranked = None
            if center_for_ctrl is not None:
                top1, top2, ranked = planner.select(center_for_ctrl, bbox_for_bonus)
                # 叠加候选线条到可视图像
                planner.draw(vis, ranked, top1, top2)

            # 触发穿门承诺：① 近门且 dA>0（原逻辑），② TTC < 阈值（新增）
            if flying and auto_mode:
                expected_pitch = CFG.TARGET_FWD + (CFG.FWD_NEAR_BOOST if near_gate else 0)
                if ((near_gate and d_area > 0) or (ttc is not None and ttc < CFG.TTC_TRIGGER_SEC)):
                    if not cross_sm.active:
                        zoffset.trigger()
                        forwardburst.trigger(expected_pitch)
                        cross_sm.start(CFG.CROSS_HOLD_SEC)

                if center_for_ctrl is not None:
                    # 选择控制目标像素：Top1 或 Top2（备选开关）
                    if planner.use_backup and top2 is not None:
                        target_px = (int(top2[0]), int(top2[1]))
                    elif top1 is not None:
                        target_px = (int(top1[0]), int(top1[1]))
                    else:
                        target_px = center_for_ctrl  # 回退

                    # 计算控制量（以目标像素为 setpoint）
                    r,p,z,y,pos = controller.compute_to_point(
                        target_px, area_for_trigger, near_gate, has_gate=True
                    )

                    # 叠加偏置/承诺期限制
                    def apply_biases(r, p, z, y, expect_pitch):
                        z += zoffset.bias()
                        p = max(p, forwardburst.bias())
                        if cross_sm.step():
                            y = clamp(y, -CFG.CROSS_YAW_LIMIT, CFG.CROSS_YAW_LIMIT)
                            p = max(p, max(CFG.TARGET_FWD, expect_pitch))
                        return r,p,z,y

                    r,p,z,y = apply_biases(r,p,z,y, expect_pitch=p)
                    last_rc = (int(r),int(p),int(z),int(y))
                    planner.commit_prev(target_px)   # 用于平滑代价
                    sticky.set((int(r),int(p),int(z),int(y),pos))
                    safe_send_rc(t, last_rc[0], last_rc[1], last_rc[2], last_rc[3],
                                 tag="AUTO-CAND-ALT" if planner.use_backup else "AUTO-CAND")

                else:
                    # 丢失：粘滞 or 搜索
                    rc = sticky.lost_tick_and_get()
                    def apply_biases(r, p, z, y, expect_pitch):
                        z += zoffset.bias()
                        p = max(p, forwardburst.bias())
                        if cross_sm.step():
                            y = clamp(y, -CFG.CROSS_YAW_LIMIT, CFG.CROSS_YAW_LIMIT)
                            p = max(p, max(CFG.TARGET_FWD, expect_pitch))
                        return r,p,z,y

                    if rc is not None:
                        r,p,z,y,pos = rc
                        r,p,z,y = apply_biases(r,p,z,y, expect_pitch=p)
                        last_rc = (int(r),int(p),int(z),int(y))
                        safe_send_rc(t, last_rc[0], last_rc[1], last_rc[2], last_rc[3], tag="AUTO-STICKY")
                    else:
                        now = time.time()
                        if now - last_yaw_flip > 2.5:
                            yaw_dir = -1 if (np.random.rand()>0.5) else 1
                            last_yaw_flip = now
                        yaw = yaw_dir * CFG.SEARCH_YAW_DPS
                        r,p,z,y = 0, CFG.SEARCH_FWD, 0, int(yaw)
                        r,p,z,y = apply_biases(r,p,z,y, expect_pitch=p)
                        last_rc = (int(r),int(p),int(z),int(y))
                        safe_send_rc(t, last_rc[0], last_rc[1], last_rc[2], last_rc[3], tag="AUTO-SEARCH")

            # HUD
            ex,ey,af = controller.ex_prev, controller.ey_prev, controller.af_ema.v
            cand_text = ""
            if top1 is not None:
                cand_text += f"Cands: TOP1 J={top1[2]:.3f}"
            if top2 is not None:
                cand_text += f" | TOP2 J={top2[2]:.3f}"
            if planner.use_backup:
                cand_text += " | USE=ALT#2"
            if ttc is not None:
                cand_text += f" | TTC={ttc:.2f}s"
            draw_hud_rgb(vis, red_res if red_ok else None,
                         yolo_res, last_rc, (ex,ey,af), cross_sm.active, area_for_trigger, cand_text)

            # ===== Map（里程计 + 视觉投影 + 候选方向）=====
            now_t_map = time.time()
            dt_map = max(1e-3, now_t_map - topo.last_t)
            topo.last_t = now_t_map
            topo.step_odometry(last_rc, dt_map)

            # gate 量测：优先用 YOLO，其次用 RED（都可以画）
            if yolo_res and yolo_res.get("ok", False):
                topo.observe_gate_from_bbox(yolo_res["bbox"], yolo_res["center"])
            if red_ok:
                topo.observe_gate_from_bbox(red_res["bbox"], red_res["center"])

            # 候选方向 (Top1/Top2)
            topo.observe_candidates(top1, top2)

            if topo.show:
                cv2.imshow("Top-Down Map", topo.render())

            cv2.imshow("Tello (YOLO+RED+Candidates)", rgb_to_bgr(vis))

    finally:
        try: safe_send_rc(t,0,0,0,0,"finally")
        except: pass
        try: t.streamoff()
        except: pass
        try: t.end()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass

if __name__ == "__main__":
    main()
