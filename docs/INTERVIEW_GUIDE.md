# DigitVision — Interview Guide

This guide prepares you to discuss every technical decision in this repository
confidently at an internship or junior engineering interview. It is organised
by the questions interviewers most commonly ask about ML portfolio projects.

Each answer is grounded in the actual code — not generic theory.

---

## 30-Second Explanation

> *"DigitVision is a handwritten digit recognition system that trains and compares
> three neural network architectures — a Dense baseline, LeNet-5 from 1998, and a
> modern Custom CNN — on the MNIST dataset. It wraps them in an interactive
> Streamlit dashboard with Grad-CAM explainability so you can actually see what the
> CNN is looking at when it makes a prediction. I built it to demonstrate that I can
> write production-quality ML code: the project has 145 automated tests, a CI/CD
> pipeline, Docker support, and every component is independently testable."*

---

## 2-Minute Explanation

> *"DigitVision lets users draw digits on a canvas or upload images, and it predicts
> the digit using one of three models they can choose between.*
>
> *The three models tell a story: the Dense Neural Network treats every pixel
> independently — it achieves 97.5% accuracy but uses 530,000 parameters. LeNet-5,
> which Yann LeCun designed in 1998, introduces convolutions and reaches 98.5% with
> only 61,000 parameters. My Custom CNN uses modern practices — BatchNormalization,
> GlobalAveragePooling instead of Flatten, and MaxPooling — and reaches 99.3% with
> just 75,000 parameters. That's 7× fewer parameters than the Dense NN for better
> accuracy.*
>
> *Beyond the models, the system has Grad-CAM explainability, which generates
> heatmaps showing which pixels drove each prediction. The backend is fully decoupled
> from the Streamlit frontend — every function can be tested without launching a browser.
> The 145-test suite covers preprocessing, model architectures, Grad-CAM, prediction
> caching, artifact generation, and the UI components themselves.*
>
> *The project runs locally in a Python venv, in Docker for reproducible deployment,
> and has GitHub Actions for automated lint and test checks on every commit."*

---

## 5-Minute Deep Technical Explanation

### Data Pipeline

> *"MNIST is downloaded via `tf.keras.datasets` and cached locally. I normalize
> pixel values to `[0.0, 1.0]` and reshape from `(N, 28, 28)` to `(N, 28, 28, 1)` —
> the channel dimension that Conv2D requires. Labels are one-hot encoded for
> categorical cross-entropy loss.*
>
> *For the validation split, I use `numpy.random.default_rng` with a fixed seed
> instead of `train_test_split`. This means the split is reproducible across runs
> without depending on scikit-learn's random state, which can behave differently
> across versions.*
>
> *The dataset is stored in an `@dataclass(frozen=True)`, which prevents any
> downstream function from accidentally mutating the data in memory. It also makes
> the access named — `data.X_train` is clearer than `data[0]`."*

### Model Architectures

> *"The Dense NN is intentionally bad at the spatial task. Flattening destroys
> adjacency — a pixel and its neighbour have no special relationship after flatten.
> This is why it needs 530K parameters to get 97.5%.*
>
> *The Custom CNN uses the Keras Functional API rather than Sequential. This isn't
> just style — it makes the architecture explicit as a computation graph, supports
> branching and multi-output designs that Sequential cannot express, and is how
> every serious model (ResNet, EfficientNet, BERT) is built.*
>
> *The key architectural choices in my CNN are:*
> - *`Conv2D(use_bias=False)` before `BatchNormalization` — BN's learned `beta`
>   acts as the bias, so keeping the Conv bias adds parameters with zero effect*
> - *`GlobalAveragePooling2D` instead of `Flatten` — after the last conv block
>   the spatial maps are 7×7×64, which Flatten would turn into 3,136 features.
>   GAP reduces each of the 64 maps to a single value, giving us 64 features.
>   This is inherently regularising and is used in ResNet, MobileNet, EfficientNet*
> - *Two convolutions before each MaxPool — this allows the network to compose
>   more complex features than a single convolution would permit at each scale"*

### Grad-CAM

> *"Grad-CAM answers: which pixels caused this prediction? The algorithm:*
>
> *1. Build a gradient model that outputs both the final Conv2D layer's feature maps
>    and the model's predictions simultaneously*
> *2. Use `tf.GradientTape` to compute the gradient of the predicted class score
>    with respect to the conv layer's activations*
> *3. Average those gradients spatially — this gives one importance weight per filter*
> *4. Weight each filter's feature map by its importance weight and sum them*
> *5. Apply ReLU to keep only regions that positively contributed — we don't want
>    to highlight regions that suppressed the prediction*
> *6. Resize to 28×28 and normalize to `[0, 1]` for colourmap application*
>
> *The Dense NN has no Conv2D layers, so `compute_gradcam()` returns `None` for it.
> The UI handles this explicitly."*

### Testing Strategy

> *"The test suite has 145 tests across 8 files and requires no trained models and
> no network connection. I achieved this through two patterns:*
>
> *First, stub Keras models — minimal 2-layer networks built in `conftest.py`
> that produce the right output shape (N, 10) without any real weights. These
> let me test prediction caching, Grad-CAM layer detection, and batch inference
> without downloading a 150 MB model file.*
>
> *Second, the Streamlit component tests. Streamlit components are Python functions
> that return HTML strings or Plotly figures. I test them by calling the functions
> directly, checking the output string for expected content, and verifying that
> chart traces have the right structure. No browser, no Streamlit server, no
> rendering — just pure Python function testing."*

---

## Common Interview Questions

### Why three models?

The three architectures tell a chronological story. The Dense NN (1980s-era approach)
establishes why spatial structure matters. LeNet-5 (1998) shows how convolution solved
the translation problem. The Custom CNN (post-2015 practices) shows what BatchNorm
and GAP add on top. Each improvement is explainable and measurable.

### Why MNIST and not a harder dataset?

MNIST is the right scope for a portfolio project that aims to demonstrate
*engineering quality* rather than model sophistication. The value here is the
architecture, testing, deployment, and explainability — not solving an unsolved
research problem. Using MNIST keeps training times in the 3–15 minute range,
making the project reproducible by anyone with a laptop.

### What is `ImageDataGenerator` and why is it deprecated?

`ImageDataGenerator` is a class from the pre-Keras-3 era that applies augmentation
on the CPU while the GPU trains. In Keras 3 / TF 2.16+, the preferred approach is
augmentation layers inside `tf.data.Dataset`, which can run on the GPU or in
parallel with training. I kept `ImageDataGenerator` because it is stable, correct,
and well-tested — replacing it would be a refactor with no accuracy benefit for
this project.

### Why not use a pre-trained model?

Using a pre-trained model (like MobileNet) for MNIST would be architecturally
inappropriate — ImageNet-pretrained weights expect 224×224 RGB images, not 28×28
grayscale. More importantly, training from scratch is the pedagogically valuable
choice: it demonstrates that I understand why each layer exists and what it does.

### Explain the preprocessing pipeline for canvas images

The canvas produces an RGBA image where the background is fully transparent
(`alpha=0`) and the stroke is fully opaque (`alpha=255`). I extract only the
alpha channel rather than converting RGB to grayscale — this gives a cleaner
signal regardless of stroke colour. I then invert the image (MNIST has white
digits on black; the canvas has black strokes on a transparent background), resize
to 28×28 using `cv2.INTER_AREA` (the correct choice for downscaling, as it averages
pixels rather than sampling), normalize to `[0.0, 1.0]`, and reshape to
`(1, 28, 28, 1)` for the model.

### Why `@dataclass(frozen=True)`?

Three reasons: immutability (no code can accidentally overwrite `data.X_train`
mid-training), named access (clearer than positional indexing), and explicitness
(the fields document exactly what the dataset contains). Frozen dataclasses are
also hashable, which enables future caching patterns.

### How does model caching work in the Streamlit app?

The `predict.py` module maintains a module-level dictionary `_model_cache` that
maps model name to a loaded Keras model. On first call, the model is loaded from
disk (1–2 seconds). On every subsequent call in the same Streamlit session, the
cached instance is returned immediately. Without this, every user interaction
would trigger a full model reload, making the UI feel sluggish.

### What does EarlyStopping do and why does it matter?

EarlyStopping monitors `val_loss`. If it does not improve for 5 consecutive epochs
(`patience=5`), training stops. This prevents overfitting past the optimal point
and avoids wasted compute. `restore_best_weights=True` ensures the final model
state is the best checkpoint, not the last epoch — important when the final epochs
show a validation dip.

### Walk me through a prediction from canvas to output

1. User draws on the Streamlit canvas → `streamlit-drawable-canvas` returns a NumPy RGBA array `(280, 280, 4)`
2. `predict_from_canvas()` calls `canvas_image_to_model_input()`:
   - Extract alpha channel `(280, 280)`
   - Invert (`255 - alpha`) to flip the polarity to MNIST convention
   - Resize to `(28, 28)` with `cv2.INTER_AREA`
   - Normalize to `[0.0, 1.0]` and reshape to `(1, 28, 28, 1)`
3. `_run_inference()` loads the cached model and calls `model.predict()`
4. The softmax output `(1, 10)` is converted to a `PredictionResult` dataclass:
   - `predicted_digit = int(np.argmax(probabilities))`
   - `confidence = float(probabilities[predicted_digit])`
   - `all_probabilities`: dict mapping `"0"–"9"` to probability float
5. The UI renders the digit, confidence bar chart, and model name

---

## Key Numbers to Remember

| Fact | Value |
|---|---|
| Dense NN parameters | ~530,000 |
| LeNet-5 parameters | ~61,000 |
| Custom CNN parameters | ~75,000 |
| Dense NN test accuracy | ~97.5% |
| LeNet-5 test accuracy | ~98.5% |
| Custom CNN test accuracy | ~99.3% |
| MNIST training set | 60,000 images |
| MNIST test set | 10,000 images |
| Validation split | 10% of training (6,000 images) |
| Number of digit classes | 10 (0–9) |
| Input image size | 28×28×1 |
| Canvas size | 280×280 (10× MNIST) |
| Test suite size | 145 tests across 8 files |
| Random seed | 42 (used everywhere for reproducibility) |

---

## What to Say if Asked About Weaknesses

Be direct and show you have already thought about it:

> *"The main limitations are scope — MNIST is a clean, centred dataset, and real
> handwriting is messier. The model will degrade on digits drawn at unusual angles
> or with unusual stroke styles. The augmentation (10° rotation, 10% zoom/shift)
> helps somewhat but does not fully close this gap.*
>
> *The other known limitation is that `ImageDataGenerator` is deprecated. I chose
> to keep it because replacing it with `tf.data` would be a refactor with no
> accuracy benefit for this project. The roadmap includes this migration in v1.1.*
>
> *The uploaded-image inversion heuristic uses corner sampling to detect background
> color, which is more robust than a global mean but still not perfect for
> edge-case images."*
