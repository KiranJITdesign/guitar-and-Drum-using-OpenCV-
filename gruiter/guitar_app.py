"""
Gruitre - Guitar (realistic, easy, low-latency)
- Realistic plucked-string tone via Karplus-Strong physical modelling
- Polyphonic (all 6 strings can ring together, like a real guitar)
- Low-latency audio: small blocksize, vectorized callback, cached notes
- Easy to play: just swipe index finger across lanes (string-crossing + strum)
"""
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
NOTE_NAMES = ["E2", "A2", "D3", "G3", "B3", "E4"]


def midi_to_freq(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


# -------------------- Realistic guitar tone: Karplus-Strong --------------------
def ks_pluck(freq: float, velocity: float = 1.0, sr: int = SR,
             duration: float = 2.2, brightness: float = 0.5) -> np.ndarray:
    """Render one plucked-string note. Fast (<5ms with scipy), called from main thread."""
    freq = float(max(40.0, min(1200.0, freq)))
    velocity = float(np.clip(velocity, 0.1, 1.0))
    N = int(sr * duration)
    L = max(2, int(sr / freq))

    # Higher strings decay a bit faster, like a real guitar.
    # loss ~0.999 for low E, ~0.994 for high E.
    t = (freq - 82.0) / (330.0 - 82.0)
    t = float(np.clip(t, 0.0, 1.0))
    loss = 0.999 - 0.005 * t

    if _HAS_SCIPY:
        # Excitation: noise burst, lowpassed slightly for pick softness.
        exc = np.random.uniform(-1.0, 1.0, L).astype(np.float64)
        # Simple one-pole lowpass on excitation: controls brightness.
        # brightness 0..1 -> less/more high freq
        b_blend = 0.3 + 0.7 * float(np.clip(brightness, 0.0, 1.0))
        exc_smooth = np.empty_like(exc)
        exc_smooth[0] = exc[0]
        for i in range(1, L):
            exc_smooth[i] = b_blend * exc[i] + (1.0 - b_blend) * exc_smooth[i - 1]
        x = np.zeros(N, dtype=np.float64)
        x[:L] = exc_smooth
        # KS loop as IIR filter: y[n] = x[n] + loss*0.5*(y[n-L]+y[n-L-1])
        b = np.array([1.0])
        a = np.zeros(L + 2)
        a[0] = 1.0
        a[L] = -0.5 * loss
        a[L + 1] = -0.5 * loss
        out = scipy_signal.lfilter(b, a, x).astype(np.float32)
    else:
        # Fallback (no scipy): vectorized decaying harmonics + pick snap.
        # Fully numpy, no per-sample Python loop -> still fast.
        tt = np.arange(N, dtype=np.float32) / sr
        # harmonic amplitudes for plucked string ~ 1/n^1.2
        out = np.zeros(N, dtype=np.float32)
        for n_harm in range(1, 9):
            amp = 1.0 / (n_harm ** 1.25)
            # higher harmonics decay faster
            dec = math.exp(-1.2 * n_harm * freq / 4000.0) if False else None
            decay_rate = 1.5 + 0.9 * n_harm + freq / 400.0
            out += (amp * np.sin(2.0 * np.pi * freq * n_harm * tt)
                    * np.exp(-tt * decay_rate)).astype(np.float32)
        # pick transient: short noise burst
        snap_len = min(int(sr * 0.008), N)
        out[:snap_len] += (np.random.uniform(-1, 1, snap_len).astype(np.float32)
                           * np.linspace(1.0, 0.0, snap_len, dtype=np.float32) * 0.35)
        # overall body decay
        out *= np.exp(-tt * 1.8).astype(np.float32)

    # Normalize then apply velocity + short fade-out to avoid clicks
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-6:
        out = out / peak
    out = out * (0.9 * velocity)
    fade = int(sr * 0.02)
    if fade < len(out):
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out.astype(np.float32)


# -------------------- Low-latency polyphonic audio engine --------------------
class AudioEngine:
    """Callback does ONLY vectorized mixing. Note rendering happens in main thread.
    Main thread -> queue -> callback mixes. This is what removes the delay/stutter."""

    def __init__(self, sr: int = SR, master: float = 0.8, blocksize: int = 256):
        self.sr = sr
        self.master = float(master)
        self.blocksize = blocksize
        self.stream = None
        self._pending: "queue.Queue[np.ndarray]" = queue.Queue()
        self._voices: list = []  # list of [buffer, pos]
        self._lock = threading.Lock()
        self._cache: dict = {}  # (rounded_midi, brightness) -> buffer at vel=1.0

    def _callback(self, outdata, frames, time_info, status):
        # Drain newly plucked notes (non-blocking, fast)
        try:
            while True:
                buf = self._pending.get_nowait()
                with self._lock:
                    self._voices.append([buf, 0])
                    # cap polyphony: drop oldest quiet/finished first
                    if len(self._voices) > 12:
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

        # soft clip to avoid harsh digital clipping on strums
        out = np.tanh(out * self.master).astype(np.float32)
        outdata[:] = out.reshape(-1, 1)

    def start(self):
        if self.stream is not None:
            return
        self.stream = sd.OutputStream(
            channels=1,
            callback=self._callback,
            samplerate=self.sr,
            blocksize=self.blocksize,  # small = low latency (~6ms @256)
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

    def pluck(self, freq: float, velocity: float = 1.0, brightness: float = 0.55):
        """Render (cached) + enqueue. Returns immediately -> no play delay."""
        if self.stream is None:
            self.start()
        midi_key = round(float(freq))  # cache by Hz is fine; freqs are fixed notes
        key = (midi_key, round(brightness, 2))
        buf = self._cache.get(key)
        if buf is None:
            buf = ks_pluck(freq, velocity=1.0, sr=self.sr, brightness=brightness)
            self._cache[key] = buf
        # scale cached full-velocity buffer to requested velocity (cheap vector op)
        v = float(np.clip(velocity, 0.1, 1.0))
        if v < 0.999:
            buf = (buf * v).astype(np.float32)
        self._pending.put(buf)


# -------------------- Hand detection: easy strum mapping --------------------
@dataclass
class ControlConfig:
    string_base_midi: list = field(default_factory=lambda: [40, 45, 50, 55, 59, 64])
    # Easy-play thresholds on RAW (unsmoothed) velocity, normalized units/sec.
    # A casual swipe is ~2-6. Old code used smoothed 0.45 which felt laggy/hard.
    pluck_vel_threshold: float = 1.1
    cross_vel_threshold: float = 0.5   # crossing a lane line needs only gentle motion
    per_string_cooldown: float = 0.09


class HandGuitarController:
    def __init__(self, config: ControlConfig):
        self.config = config
        mp_hands = mp.solutions.hands
        self.mp_hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,  # lite model = much lower camera->sound latency
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.prev_tip = None
        self.prev_time = None
        self.prev_lane = None
        self.last_pluck = [0.0] * 6

    @staticmethod
    def _dist(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_hands.process(frame_rgb)

        if not res.multi_hand_landmarks:
            self.prev_tip = None
            self.prev_time = None
            self.prev_lane = None
            return None

        hand_lms = res.multi_hand_landmarks[0]
        lm = np.array([[p.x, p.y, p.z] for p in hand_lms.landmark], dtype=np.float32)

        tip = lm[8][:2]  # index fingertip
        now = time.time()
        raw_vel = 0.0
        if self.prev_tip is not None and self.prev_time is not None and now > self.prev_time:
            dt = now - self.prev_time
            if dt > 1e-4:
                raw_vel = self._dist(tip, self.prev_tip) / dt

        x = float(np.clip(tip[0], 0.0, 1.0))
        lane = min(int(x * 6), 5)
        crossed = (self.prev_lane is not None and lane != self.prev_lane)

        self.prev_tip = tip.copy()
        self.prev_time = now
        self.prev_lane = lane

        freq = midi_to_freq(float(self.config.string_base_midi[lane]))
        # Louder when you swipe faster; always audible floor so light moves still sing.
        velocity = float(np.clip(0.35 + raw_vel / 5.0, 0.3, 1.0))

        # Trigger logic (easy): lane crossing with gentle motion, OR fast flick anywhere.
        should = False
        if crossed and raw_vel > self.config.cross_vel_threshold:
            should = True
        elif raw_vel > self.config.pluck_vel_threshold:
            should = True

        if should and (now - self.last_pluck[lane]) < self.config.per_string_cooldown:
            should = False
        if should:
            self.last_pluck[lane] = now

        return {
            "string_idx": lane,
            "freq": freq,
            "velocity": velocity,
            "raw_vel": float(raw_vel),
            "crossed": bool(crossed),
            "trigger": bool(should),
            "landmarks": lm,
        }

    def draw(self, frame_bgr, data):
        if data is None:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        lm = data["landmarks"]
        # Fixed full-height lanes: stable targets, easy to aim.
        for i in range(6):
            x0 = int((i / 6) * w)
            color = (0, 255, 255) if i == data["string_idx"] else (90, 90, 90)
            thick = 2 if i == data["string_idx"] else 1
            cv2.line(frame_bgr, (x0, 0), (x0, h), color, thick)
            cv2.putText(frame_bgr, NOTE_NAMES[i], (x0 + 8, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        x_px = int(np.clip(lm[8][0], 0, 1) * w)
        y_px = int(np.clip(lm[8][1], 0, 1) * h)
        dot_color = (0, 200, 0) if data["trigger"] else (0, 255, 0)
        cv2.circle(frame_bgr, (x_px, y_px), 10, dot_color, -1)
        cv2.putText(
            frame_bgr,
            f"{NOTE_NAMES[data['string_idx']]} {data['freq']:.0f}Hz  swipe={data['raw_vel']:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2, cv2.LINE_AA,
        )
        return frame_bgr


def main():
    print("[startup] realistic guitar: KS strings + low-latency polyphonic audio")
    print(f"[startup] scipy={'yes' if _HAS_SCIPY else 'no (using fallback synth)'}")
    config = ControlConfig()
    controller = HandGuitarController(config)
    audio = AudioEngine()
    audio.start()  # start once; keep hot so first pluck has zero delay

    # Pre-render all 6 open strings now -> subsequent plucks are instant cache hits.
    print("[startup] pre-rendering 6 strings...")
    t0 = time.time()
    for midi in config.string_base_midi:
        audio.pluck(midi_to_freq(float(midi)), velocity=0.0 if False else 0.9)
        # drain the silent-ish warmup? keep them: they are quiet warmups; clear queue instead
    # Remove warmup notes from queue so app starts silent, but cache stays hot.
    try:
        while True:
            audio._pending.get_nowait()
    except queue.Empty:
        pass
    print(f"[startup] strings ready in {time.time()-t0:.2f}s. Swipe to strum!")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # smaller frames = faster hand tracking
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            data = controller.process(frame)

            if data is not None:
                if data["trigger"]:
                    audio.pluck(freq=data["freq"], velocity=data["velocity"])
                frame = controller.draw(frame, data)

            cv2.putText(frame, "Swipe index across lanes to strum  |  ESC quit",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Gruitre - Guitar", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        audio.stop()


if __name__ == "__main__":
    main()
