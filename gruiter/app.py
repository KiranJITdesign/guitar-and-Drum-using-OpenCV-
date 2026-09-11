import math
import queue
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp
import sounddevice as sd

try:
    from scipy import signal as scipy_signal
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

SR = 44100

def midi_to_freq(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))

# -------------------- Synth Helpers --------------------
def ks_pluck(freq: float, velocity: float = 1.0, sr: int = SR,
             duration: float = 2.2, brightness: float = 0.5) -> np.ndarray:
    freq = float(max(40.0, min(1200.0, freq)))
    velocity = float(np.clip(velocity, 0.1, 1.0))
    N = int(sr * duration)
    L = max(2, int(sr / freq))

    t = (freq - 82.0) / (330.0 - 82.0)
    t = float(np.clip(t, 0.0, 1.0))
    loss = 0.999 - 0.005 * t

    if _HAS_SCIPY:
        exc = np.random.uniform(-1.0, 1.0, L).astype(np.float64)
        b_blend = 0.3 + 0.7 * float(np.clip(brightness, 0.0, 1.0))
        exc_smooth = np.empty_like(exc)
        exc_smooth[0] = exc[0]
        for i in range(1, L):
            exc_smooth[i] = b_blend * exc[i] + (1.0 - b_blend) * exc_smooth[i - 1]
        x = np.zeros(N, dtype=np.float64)
        x[:L] = exc_smooth
        
        b = np.array([1.0])
        a = np.zeros(L + 2)
        a[0] = 1.0
        a[L] = -0.5 * loss
        a[L + 1] = -0.5 * loss
        out = scipy_signal.lfilter(b, a, x).astype(np.float32)
    else:
        tt = np.arange(N, dtype=np.float32) / sr
        out = np.zeros(N, dtype=np.float32)
        for n_harm in range(1, 9):
            amp = 1.0 / (n_harm ** 1.25)
            decay_rate = 1.5 + 0.9 * n_harm + freq / 400.0
            out += (amp * np.sin(2.0 * np.pi * freq * n_harm * tt)
                    * np.exp(-tt * decay_rate)).astype(np.float32)
        snap_len = min(int(sr * 0.008), N)
        out[:snap_len] += (np.random.uniform(-1, 1, snap_len).astype(np.float32)
                           * np.linspace(1.0, 0.0, snap_len, dtype=np.float32) * 0.35)
        out *= np.exp(-tt * 1.8).astype(np.float32)

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-6:
        out = out / peak
    out = out * (0.9 * velocity)
    fade = int(sr * 0.02)
    if fade < len(out):
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out.astype(np.float32)

def generate_drum(drum_type: str, velocity: float = 1.0, sr: int = SR) -> np.ndarray:
    velocity = float(np.clip(velocity, 0.1, 1.0))
    N = int(sr * 0.5)
    t = np.arange(N, dtype=np.float32) / sr
    
    if drum_type == "kick":
        freq_env = 150.0 * np.exp(-t * 20.0) + 40.0
        phase = np.cumsum(2.0 * np.pi * freq_env / sr)
        out = np.sin(phase) * np.exp(-t * 10.0)
    elif drum_type == "snare":
        tone = np.sin(2.0 * np.pi * 180.0 * t) * np.exp(-t * 15.0)
        noise = np.random.uniform(-1.0, 1.0, N) * np.exp(-t * 25.0)
        out = 0.5 * tone + 0.5 * noise
    elif drum_type == "hihat":
        noise = np.random.uniform(-1.0, 1.0, N) * np.exp(-t * 40.0)
        out = noise
        for i in range(1, len(out)):
            out[i] = out[i] - 0.5 * out[i-1]
    elif drum_type == "tom":
        freq_env = 120.0 * np.exp(-t * 10.0) + 80.0
        phase = np.cumsum(2.0 * np.pi * freq_env / sr)
        out = np.sin(phase) * np.exp(-t * 8.0)
    else:
        out = np.sin(2.0 * np.pi * 100.0 * t) * np.exp(-t * 10.0)
        
    out = out.astype(np.float32)
    peak = float(np.max(np.abs(out)))
    if peak > 1e-6:
        out = out / peak
    return out * velocity


# -------------------- AudioEngine --------------------
class AudioEngine:
    def __init__(self, sr: int = SR, master: float = 0.8, blocksize: int = 256):
        self.sr = sr
        self.master = float(master)
        self.blocksize = blocksize
        self.stream = None
        self._pending = queue.Queue()
        self._voices = []
        self._lock = threading.Lock()
        self._cache = {} 
        self._drum_cache = {}

    def _callback(self, outdata, frames, time_info, status):
        try:
            while True:
                buf = self._pending.get_nowait()
                with self._lock:
                    self._voices.append([buf, 0])
                    if len(self._voices) > 16:
                        self._voices.pop(0)
        except queue.Empty:
            pass

        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            alive = []
            for buf, pos in self._voices:
                n = min(len(buf) - pos, frames)
                if n > 0:
                    out[:n] += buf[pos:pos + n]
                    pos += n
                if pos < len(buf):
                    alive.append([buf, pos])
            self._voices = alive

        out = np.tanh(out * self.master).astype(np.float32)
        outdata[:] = out.reshape(-1, 1)

    def start(self):
        if self.stream is not None:
            return
        self.stream = sd.OutputStream(
            channels=1,
            callback=self._callback,
            samplerate=self.sr,
            blocksize=self.blocksize,
            latency="low",
            dtype="float32",
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def silence(self):
        with self._lock:
            self._voices = []
            
    def pluck_guitar(self, freq: float, velocity: float = 1.0, brightness: float = 0.55):
        if self.stream is None:
            self.start()
        midi_key = round(float(freq))
        key = (midi_key, round(brightness, 2))
        buf = self._cache.get(key)
        if buf is None:
            buf = ks_pluck(freq, velocity=1.0, sr=self.sr, brightness=brightness)
            self._cache[key] = buf
            
        v = float(np.clip(velocity, 0.1, 1.0))
        if v < 0.999:
            buf = (buf * v).astype(np.float32)
        self._pending.put(buf)
        
    def pluck_drum(self, drum_type: str, velocity: float = 1.0):
        if self.stream is None:
            self.start()
        buf = self._drum_cache.get(drum_type)
        if buf is None:
            buf = generate_drum(drum_type, velocity=1.0, sr=self.sr)
            self._drum_cache[drum_type] = buf
            
        v = float(np.clip(velocity, 0.1, 1.0))
        if v < 0.999:
            buf = (buf * v).astype(np.float32)
        self._pending.put(buf)


# -------------------- Hand detection --------------------
@dataclass
class ControlConfig:
    string_base_midi: list = field(default_factory=lambda: [40, 45, 50, 55, 59, 64])
    # Easy play params
    pluck_vel_threshold: float = 1.1
    cross_vel_threshold: float = 0.5
    per_string_cooldown: float = 0.09
    motion_smooth: float = 0.85


class HandController:
    def __init__(self, config: ControlConfig):
        self.config = config
        mp_hands = mp.solutions.hands
        self.mp_hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.prev_tip = None
        self.prev_time = None
        self.prev_lane = None
        self.last_pluck = [0.0] * 6
        self.smooth_vel = 0.0
        self.sound_enabled = False

    @staticmethod
    def _dist(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def _detect_palm_open(self, lm):
        thumb_tip = lm[4]
        index_mcp = lm[5]
        index_tip = lm[8]
        middle_tip = lm[12]
        ring_tip = lm[16]
        pinky_tip = lm[20]

        d_thumb_index = self._dist(thumb_tip, index_mcp)
        wrist = lm[0]
        tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        wrist_d = [self._dist(t, wrist) for t in tips]
        avg_tip = float(np.mean(wrist_d))

        return d_thumb_index > 0.10 and avg_tip > 0.20

    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_hands.process(frame_rgb)

        if not res.multi_hand_landmarks:
            self.prev_tip = None
            self.prev_time = None
            self.prev_lane = None
            self.smooth_vel *= 0.8
            return None

        hand_lms = res.multi_hand_landmarks[0]
        lm = np.array([[p.x, p.y, p.z] for p in hand_lms.landmark], dtype=np.float32)

        tip = lm[8][:2]
        palm_y = float(lm[0][1])
        now = time.time()
        
        raw_vel = 0.0
        if self.prev_tip is not None and self.prev_time is not None and now > self.prev_time:
            dt = now - self.prev_time
            if dt > 1e-4:
                raw_vel = self._dist(tip, self.prev_tip) / dt

        self.smooth_vel = self.config.motion_smooth * self.smooth_vel + (1.0 - self.config.motion_smooth) * raw_vel

        x = float(np.clip(tip[0], 0.0, 1.0))
        lane = min(int(x * 6), 5)
        crossed = (self.prev_lane is not None and lane != self.prev_lane)

        self.prev_tip = tip.copy()
        self.prev_time = now
        self.prev_lane = lane

        freq = midi_to_freq(float(self.config.string_base_midi[lane]))
        
        # Easy swipe calculation for guitar
        guitar_velocity = float(np.clip(0.35 + raw_vel / 5.0, 0.3, 1.0))
        
        # Original drum volume based on palm height
        drum_volume = float(np.clip((0.65 - palm_y) / 0.65, 0.0, 1.0))

        should_pluck_guitar = False
        if crossed and raw_vel > self.config.cross_vel_threshold:
            should_pluck_guitar = True
        elif raw_vel > self.config.pluck_vel_threshold:
            should_pluck_guitar = True

        if should_pluck_guitar and (now - self.last_pluck[lane]) < self.config.per_string_cooldown:
            should_pluck_guitar = False
            
        if should_pluck_guitar:
            self.last_pluck[lane] = now
            
        palm_open = self._detect_palm_open(lm)

        return {
            "string_idx": lane,
            "freq": freq,
            "guitar_velocity": guitar_velocity,
            "drum_volume": drum_volume,
            "raw_vel": float(raw_vel),
            "smooth_vel": float(self.smooth_vel),
            "crossed": bool(crossed),
            "trigger_guitar": bool(should_pluck_guitar),
            "palm_open": palm_open,
            "landmarks": lm,
        }

    def draw(self, frame_bgr, data):
        if data is None:
            return frame_bgr

        lm = data["landmarks"]
        h, w = frame_bgr.shape[:2]

        x_px = int(np.clip(lm[8][0], 0, 1) * w)
        y_px = int(np.clip(lm[8][1], 0, 1) * h)
        cv2.circle(frame_bgr, (x_px, y_px), 8, (0, 255, 0), -1)

        y_base = int(lm[0][1] * h)
        top = max(10, y_base - 80)
        bottom = min(h - 10, y_base + 80)
        
        for i in range(6):
            x0 = int((i / 6) * w)
            x1 = int(((i + 1) / 6) * w)
            cv2.line(frame_bgr, (x0, top), (x0, bottom), (255, 255, 255), 1)
            if i == data["string_idx"]:
                cv2.rectangle(frame_bgr, (x0, top), (x1, bottom), (0, 255, 255), 2)

        return frame_bgr

def main():
    print("[startup] App with realistic KS guitar + drums")
    config = ControlConfig()
    controller = HandController(config)
    audio = AudioEngine()
    
    # Pre-render caches
    print("[startup] Pre-rendering strings and drums...")
    t0 = time.time()
    for midi in config.string_base_midi:
        audio.pluck_guitar(midi_to_freq(float(midi)), velocity=0.0)
    for drum in ["kick", "snare", "hihat", "tom"]:
        audio.pluck_drum(drum, velocity=0.0)
    
    # clear queue
    try:
        while True:
            audio._pending.get_nowait()
    except queue.Empty:
        pass
    print(f"[startup] Audio ready in {time.time()-t0:.2f}s.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    last_drum_time = 0.0
    drum_cooldown = 0.12

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            data = controller.process(frame)

            if data is not None:
                now = time.time()
                
                if data["palm_open"]:
                    if not controller.sound_enabled:
                        controller.sound_enabled = True
                        audio.start()
                else:
                    if controller.sound_enabled and data["smooth_vel"] < 0.10:
                        audio.silence()

                if controller.sound_enabled:
                    if data["palm_open"]:
                        if data["trigger_guitar"]:
                            audio.pluck_guitar(freq=data["freq"], velocity=data["guitar_velocity"])
                    else:
                        if (data["smooth_vel"] > 0.35 and (now - last_drum_time) > drum_cooldown):
                            v = data["drum_volume"]
                            if v > 0.72: drum = "hihat"
                            elif v > 0.48: drum = "snare"
                            elif v > 0.25: drum = "tom"
                            else: drum = "kick"

                            audio.pluck_drum(drum, velocity=float(np.clip(v, 0.05, 1.0)))
                            last_drum_time = now

                frame = controller.draw(frame, data)
                
                mode = "GUITAR" if data["palm_open"] else "DRUM"
                cv2.putText(
                    frame,
                    f"Mode: {mode} (open palm=Guitar, closed palm=Drums)",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                "Hand Guitar+Drums: Swipe index to strum; closed palm punch for drums; ESC quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Gruitre", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord(' '), ord('p')):
                controller.sound_enabled = not controller.sound_enabled
                if controller.sound_enabled:
                    audio.start()
                else:
                    audio.silence()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        audio.stop()

if __name__ == "__main__":
    main()
