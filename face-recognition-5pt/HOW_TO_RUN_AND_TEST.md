# How to Run and Test Face Locking

Follow these steps to run the project and test the Face Locking feature.

---

## 1. Setup (first time only)

**Windows (Command Prompt or PowerShell):**
```cmd
cd face-recognition-5pt
setup.bat
```

**macOS / Linux:**
```bash
cd face-recognition-5pt
bash setup.sh
```

This creates a virtual environment (`.venv`) and installs dependencies.

---

## 2. Activate the virtual environment

**Windows:**
```cmd
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

---

## 3. Download the ArcFace model (first time only)

```cmd
python download_model.py
```

You should see the model downloaded to `models/embedder_arcface.onnx`.

---

## 4. Verify project files

```cmd
python verify.py
```

You should see all required files listed with checkmarks. If you get a Unicode error on Windows, run:
```cmd
set PYTHONIOENCODING=utf-8
python verify.py
```

---

## 5. Test the camera (optional)

```cmd
python -m src.camera
```

A window should open with your webcam. Press **q** to quit.

---

## 6. Enroll at least one person (required for Face Locking)

```cmd
python -m src.enroll
```

1. Enter a name (e.g. `Gabi` or your name).
2. Look at the camera and press **SPACE** to capture samples (or **A** for auto-capture).
3. Capture at least 15 samples for better recognition.
4. Press **S** to save, then **Q** to quit.

---

## 7. Run Face Locking

```cmd
python -m src.lock
```

1. You will see the list of enrolled identities.
2. Enter the **exact name** of the person to lock onto (e.g. `Gabi`).
3. Look at the camera. When you are recognized with confidence, the system will **lock** (green box + "LOCKED: &lt;name&gt;").
4. While locked:
   - Move your face **left** or **right** → "face moved left/right" is recorded.
   - **Blink** → "eye blink" is recorded.
   - **Smile** → "smile or laugh" is recorded.
5. Press **q** to quit.

---

## 8. Check the action history file

After running Face Locking, open the history file:

- **Location:** `face-recognition-5pt/data/history/`
- **Filename format:** `&lt;name&gt;_history_&lt;timestamp&gt;.txt`  
  Example: `gabi_history_20260129120530.txt`

Each line is one action: `timestamp  action_type  description`

---

## Quick test without camera (sanity check)

To only check that the lock module loads and the database path works (no camera):

```cmd
cd face-recognition-5pt
.venv\Scripts\activate
python -c "from src.config import ensure_dirs; ensure_dirs(); from src.lock import load_database; db=load_database(); print('OK - Lock module loads. Enrolled:', list(db.keys()) if db else 'None')"
```

- If you have **not** enrolled anyone: you should see `Enrolled: None`.
- If you **have** enrolled: you should see the list of names.

---

## Troubleshooting

| Problem | Solution |
|--------|----------|
| "No enrolled identities" | Run `python -m src.enroll` first and save with **S**. |
| "Cannot open camera" | Close other apps using the camera; check system privacy settings for camera access. |
| "Model not found" | Run `python download_model.py`. |
| "mediapipe not installed" | Activate `.venv` and run `pip install -r requirements.txt`. |
| verify.py Unicode error on Windows | Run `set PYTHONIOENCODING=utf-8` then `python verify.py`. |
