import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
import sounddevice as sd


# -------------------- Audio (simple real-time synth; used for drum tones) --------------------
@dataclass
class SynthState:
    sample_rate: int = 44100
    active: bool = False
    master_volume: float = 0.5
    current_freq: float = 0.0
    env_level: float = 0.0
    env_attack: float = 0.002
    env_decay: float = 0.09
    env_release: float = 0.02
    env_phase: str = "idle"


def midi_to_freq(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


class Voice:
    def __init__(self, state: SynthState):
        self.state = state
        self.phase = 0.0

    def note_on(self, freq: float, velocity: float = 1.0):
        self.state.current_freq = float(freq)
        self.state.master_volume = float(np.clip(velocity, 0.0, 1.0))
        self.state.env_phase = "attack"
        self.state.env_level = 0.0

    def note_off(self):
        self.state.env_phase = "release"

    def render(self, frames: int):
        sr = self.state.sample_rate
        out = np.zeros(frames, dtype=np.float32)

        phase_inc = 2.0 * math.pi * self.state.current_freq / sr if self.state.current_freq > 0 else 0.0

        for i in range(frames):
            if not self.state.active and self.state.env_phase == "idle":
                out[i] = 0.0
                continue

            if self.state.env_phase == "attack":
                a = math.exp(-1.0 / (sr * max(self.state.env_attack, 1e-4)))
                self.state.env_level = 1.0 + (self.state.env_level - 1.0) * a
                if self.state.env_level >= 0.98:
                    self.state.env_phase = "decay"

            elif self.state.env_phase == "decay":
                d = math.exp(-1.0 / (sr * max(self.state.env_decay, 1e-3)))
                self.state.env_level = self.state.env_level * d
                if self.state.env_level <= 0.01:
                    self.state.env_phase = "idle"

            elif self.state.env_phase == "release":
                r = math.exp(-1.0 / (sr * max(self.state.env_release, 1e-3)))
                self.state.env_level = self.state.env_level * r
                if self.state.env_level <= 0.001:
                    self.state.env_level = 0.0
                    self.state.env_phase = "idle"

            elif self.state.env_phase == "idle":
                self.state.env_level = 0.0

            if self.state.current_freq > 0 and self.state.env_phase != "idle":
                self.phase += phase_inc
                if self.phase > 2.0 * math.pi:
                    self.phase -= 2.0 * math.pi

                s = math.sin(self.phase)
                square = 1.0 if s >= 0 else -1.0
                # drum-ish: a bit more square
                out[i] = (0.55 * s + 0.45 * square) * self.state.env_level * self.state.master_volume
            else:
                out[i] = 0.0

        return out


class AudioEngine:
    def __init__(self):
        self.state = SynthState()
        self.voice = Voice(self.state)
        self.stream = None

    def _callback(self, outdata, frames, time_info, status):
        audio = self.voice.render(frames)
        outdata[:] = audio.reshape(-1, 1)

    def start(self):
        if self.stream is not None:
            return
        self.state.active = True
        self.stream = sd.OutputStream(
            channels=1,
            callback=self._callback,
            samplerate=self.state.sample_rate,
            blocksize=0,
        )
        self.stream.start()

    def stop(self):
        self.state.active = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def hit(self, freq: float, velocity: float = 1.0):
        if not self.state.active:
            self.start()
        self.voice.note_on(freq=freq, velocity=velocity)


# -------------------- Hand detection + mapping (drums) --------------------
@dataclass
class DrumConfig:
    pluck_vel_threshold: float
    motion_smooth: float
    drum_cooldown: float


mp_hands = mp.solutions.hands


class HandDrumController:
    def __init__(self, config: DrumConfig):
        self.config = config
        self.mp_hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.prev_index_tip = None
        self.prev_time = None
        self.smooth_vel = 0.0

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
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_hands.process(frame_rgb)

        if not res.multi_hand_landmarks:
            self.prev_index_tip = None
            self.prev_time = None
            self.smooth_vel *= 0.8
            return None

        hand_lms = res.multi_hand_landmarks[0]
        lm = np.array([[p.x, p.y, p.z] for p in hand_lms.landmark], dtype=np.float32)

        index_tip = lm[8][:2]
        palm_y = float(lm[0][1])

        now = time.time()
        vel = 0.0
        if self.prev_index_tip is not None and self.prev_time is not None and self.prev_time < now:
            dt = now - self.prev_time
            if dt > 1e-4:
                vel = self._dist(index_tip, self.prev_index_tip) / dt

        self.smooth_vel = self.config.motion_smooth * self.smooth_vel + (1.0 - self.config.motion_smooth) * vel
        self.prev_index_tip = index_tip
        self.prev_time = now

        # drum velocity/volume metric: lower wrist => bigger value
        v = float(np.clip((0.65 - palm_y) / 0.65, 0.0, 1.0))
        palm_open = self._detect_palm_open(lm)

        return {
            "palm_open": palm_open,
            "smooth_vel": float(self.smooth_vel),
            "volume": v,
            "landmarks": lm,
        }

    def draw(self, frame_bgr, data, drum_name: str | None = None):
        if data is None:
            return frame_bgr

        lm = data["landmarks"]
        h, w = frame_bgr.shape[:2]
        x_px = int(lm[8][0] * w)
        y_px = int(lm[8][1] * h)
        cv2.circle(frame_bgr, (x_px, y_px), 8, (0, 255, 0), -1)

        mode = "GUITAR" if data["palm_open"] else "DRUM"
        cv2.putText(
            frame_bgr,
            f"Mode: {mode}  vel={data['smooth_vel']:.2f} vol={data['volume']:.2f}" + (f"  hit={drum_name}" if drum_name else ""),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 255, 50),
            2,
            cv2.LINE_AA,
        )
        return frame_bgr


def main():
    print("[startup] loading drum config...")

    config = DrumConfig(
        pluck_vel_threshold=0.35,
        motion_smooth=0.85,
        drum_cooldown=0.12,
    )

    print("[startup] initializing controller + audio...")
    controller = HandDrumController(config)
    audio = AudioEngine()

    # Kick/snare/tom/hihat tones (simple synth frequencies)
    drum_freqs = {
        "kick": 60.0,
        "snare": 180.0,
        "tom": 120.0,
        "hihat": 400.0,
    }

    print("[startup] opening webcam (index 0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    print("[startup] entering loop. Press ESC to quit.")

    last_drum_time = 0.0
    last_drum_name = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            data = controller.process(frame)

            if data is not None:
                drum_name = None

                # Drum app only triggers when palm is closed.
                if not data["palm_open"]:
                    now = time.time()
                    if data["smooth_vel"] > config.pluck_vel_threshold and (now - last_drum_time) > config.drum_cooldown:
                        v = data["volume"]
                        if v > 0.72:
                            drum_name = "hihat"
                        elif v > 0.48:
                            drum_name = "snare"
                        elif v > 0.25:
                            drum_name = "tom"
                        else:
                            drum_name = "kick"

                        audio.hit(drum_freqs[drum_name], velocity=float(np.clip(v, 0.05, 1.0)))
                        last_drum_time = now
                        last_drum_name = drum_name

                frame = controller.draw(frame, data, drum_name=drum_name)

            cv2.putText(
                frame,
                "Drum mode: keep palm CLOSED; move index fast for hits; ESC quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Gruitre - Drums", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        audio.stop()


if __name__ == "__main__":
    main()

