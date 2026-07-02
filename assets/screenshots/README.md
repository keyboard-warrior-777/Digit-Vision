# Screenshots

This directory contains screenshots of the DigitVision Streamlit dashboard.

---

## Adding Screenshots

1. Train all models: `make train && make evaluate`
2. Launch the app: `make run`
3. Navigate to each page and capture a screenshot
4. Save files here with descriptive names:

| Filename | Page |
|---|---|
| `01_home.png` | Home page — project overview and model summary cards |
| `02_recognize_draw.png` | Digit Recognition — Draw tab with prediction result |
| `02_recognize_upload.png` | Digit Recognition — Upload tab |
| `03_playground.png` | Model Playground — three models side-by-side |
| `04_analytics_confusion.png` | Analytics — confusion matrix |
| `04_analytics_roc.png` | Analytics — ROC curves |
| `04_analytics_f1.png` | Analytics — per-class F1 chart |
| `05_gradcam.png` | CNN Explainer — Grad-CAM heatmap overlay |
| `06_dataset.png` | Dataset Explorer |
| `07_about.png` | About page |

## Screenshot Guidelines

- Use a 1280px wide browser window for consistent sizing
- Use the dark theme (default)
- Capture after all three models have been trained and evaluated
- Draw a recognizable digit (e.g., `7`) for the prediction screenshots

## Demo GIF

A screen recording GIF can be placed here as `demo.gif`.

**Recommended recording flow:**
1. Open Draw tab → draw a digit `7` → click Predict
2. Navigate to Playground → draw the same digit → click Compare
3. Navigate to CNN Explainer → show the Grad-CAM heatmap

**Tools:**
- Windows: Xbox Game Bar (`Win+G`) or ShareX
- macOS: QuickTime Player → File → New Screen Recording
- Linux: OBS Studio or `simplescreenrecorder`

Convert to GIF: [ezgif.com](https://ezgif.com/video-to-gif) or `ffmpeg`:

```bash
ffmpeg -i demo.mp4 -vf "fps=10,scale=800:-1" -loop 0 demo.gif
```
