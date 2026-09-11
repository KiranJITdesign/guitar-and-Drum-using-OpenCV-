# guitar-and-Drum-using-OpenCV-
 description: A webcam-controlled virtual drum app that lets users play kick, snare, tom, and hi-hat sounds through hand gestures.    How it works (one line): MediaPipe tracks your hand through the webcam, detects fast index-finger movements with a closed palm, and maps hand position to a drum sound and volume.          


# Gruitre (OpenCV Hand Guitar)

A Python project that uses your **camera** to detect your hand and turn motion into **6-string guitar-like tones**.

## Features
- Webcam capture (OpenCV)
- Hand landmark detection (MediaPipe)
- Maps finger positions to **6 strings**
- Different tones are produced as you “play” (pluck = fast movement)
- Controls:
  - **Open palm / stop**: toggles sound on/off
  - **Pitch (octave)**: using thumb openness
  - **Volume**: by hand height

## Setup
### 1) Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install opencv-python mediapipe numpy sounddevice
```

> If `sounddevice` fails, install PortAudio support or try:
> `pip install sounddevice --user`

## Run
```bash
python app.py
```

## Hand Guitar + Drums controls
- **Open palm (GUITAR mode):** move your index finger fast → **pluck** a 6-string note.
- **Closed palm (DRUM mode):** fast motion → play a **drum hit** (kick/snare/tom/hihat) based on your hand height.
- **Volume:** hand height (lower wrist/larger “volume” value = louder).
- **Octave:** thumb openness.

## How it works (high level)
- MediaPipe returns 21 hand landmarks.
- We compute a simple “fingertip motion” (velocity) to decide when an event happens.
- We map fingertip X-position into **6 string lanes**.
- We map thumb/palm geometry into pitch/octave and hand height into volume.


## Notes
- Works best with good lighting and a stable camera.
- You can tune parameters in `app.py`.

