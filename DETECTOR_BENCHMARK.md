# Face detector benchmark: opencv vs YuNet vs RetinaFace

Benchmarked 2026-10-05/06 on this machine (Windows 11, 24-core CPU, RTX 4090 Laptop GPU
via WSL2 Ubuntu 22.04), deepface 0.0.99, TensorFlow 2.21, ArcFace embeddings.
StarMapr currently uses `detector_backend='opencv'` (Haar cascade) in `utils_deepface.py`.

The work had two parts:
- **A detector benchmark** of 1,056 images, measuring speed, detection and matching.
- **An end-to-end comparison** on a real SketchTV sketch job. Production StarMapr (opencv)
  ran as usual, and a RetinaFace copy of the same run went on the GPU.

## Summary

- **RetinaFace needs a GPU to be affordable.** On the Windows CPU it takes about 5.4 s per
  image, roughly 20× opencv's 0.26 s. Under WSL2 with the RTX 4090 it takes 0.18 s.
- **With a GPU, a RetinaFace pipeline run costs about what today's opencv run does.** Each
  StarMapr step starts a new Python process and reloads TensorFlow and the models. That
  overhead, not detection, dominates step time, so faster detection barely moves the total.
- **RetinaFace is the better detector on photos.** It matches the actor in more test photos
  (61.2% vs 54.2%, statistically significant), finds 2.3× more usable faces in video
  frames, and recovers 42% of the training photos opencv rejected. Its extra detections
  are almost all real faces, mostly small or in the background.
- **opencv's extra detections are mostly false positives.** These include sign lettering,
  mouths, shirts and cartoon toys. The existing low-information (blank-similarity) gate
  already discards nearly all of them, so opencv's real weakness is missed faces.
- **YuNet as configured by DeepFace underperforms.** DeepFace shrinks YuNet's input to a
  640 px maximum side, after first adding a 50% border, so it finds roughly half the faces
  in group photos. With that cap removed, YuNet runs close to opencv's speed and slightly
  beats it on detection. However, it then misses some large faces, so it would need its own
  resolution policy.
- **Switching detectors is a migration, not a setting change.** The same face embeds at a
  median cosine of 0.87–0.90 across detectors, so every model in `04_models` and every
  `.pkl` cache must be rebuilt. The face-size, face-count and match thresholds were tuned
  on opencv and need recalibration.
- **On a real sketch job, swapping in RetinaFace did not produce more headshots.** It found
  more faces, but StarMapr's headshot crop geometry rejected many of them. Fred Armisen
  dropped from 4 headshots to 1, and the other three actors were unchanged (see
  [End-to-end](#end-to-end-a-real-sketch-job)). The detection gains only pay off once the
  crop geometry and size gates are adapted to RetinaFace's boxes.

**Recommendation:** keep opencv for now. RetinaFace is worth pursuing only with WSL and the
GPU, and only as a deliberate project (see [Next steps](#next-steps)). On the Windows CPU,
RetinaFace is too slow; YuNet is the only affordable upgrade there, and it needs tuning
before it is clearly better.

## What was measured

Each detector ran through `DeepFace.represent(model_name='ArcFace', normalization='base',
align=True, enforce_detection=False)` exactly as `utils_deepface.py` calls it. The
whole-image fallback "face" was dropped as production does. Images were passed in memory,
so no project caches were written.

| Variant | Meaning |
|---|---|
| `opencv` | Current production setting |
| `yunet` | DeepFace default YuNet: resizes input to a 640 px max side, score threshold 0.9 |
| `yunet_nocap` | Same model and threshold at full resolution; requires patching DeepFace's YuNet wrapper |
| `retinaface` | DeepFace default RetinaFace: upscales so the short side is 1024 px, threshold 0.9 |

The sample was a frozen manifest of 1,056 images:

- 48 actors chosen by a stable hash from the 174 that have both training and testing folders.
- Per actor: up to 12 accepted training portraits (576 total), 5 testing group photos (240),
  and 3 photos from `bad_face_count/` (144).
- 96 video frames at 1280×720: 50 from `mock_video` and 46 from a Little Britain video.

Metrics use the pipeline's own rules:

- Training gate: exactly 1 face, and not low-information.
- Testing gate: 3–10 faces.
- Low-information: cosine ≥ 0.5 to a blank image's embedding.
- Match: score ≥ 0.4 and beats other actors by ≥ 0.08.
- Usable face: informative and at least 50 px on its shorter side.

For "actor matched", each detector builds actor centroids from its own embeddings of the
sampled portraits. The other 47 sampled actors serve as competitors.

## Speed

Speeds are mean seconds per image for the full detect + align + ArcFace call. Each timing
pass used the same 75 images (25 portraits, 25 group photos, 25 video frames) on an otherwise
idle machine.

| Detector | Windows CPU | WSL2 + GPU | GPU speedup |
|---|---|---|---|
| opencv | 0.26 | 0.10 | 2.6× |
| yunet | 0.26 | 0.12 | 2.2× |
| yunet_nocap | 0.34 | 0.14 | 2.4× |
| retinaface | 5.39 | 0.18 | 30× |

- **RetinaFace on CPU** is slow mostly because of its input size. With `align=True`, DeepFace
  adds a 50% border to every image before detection (4× the pixels). RetinaFace then
  upscales the result so its short side is 1024 px. Turning the upscale off (patched) saved
  only 25–30% on stills and nothing on 720p frames.
- **RetinaFace on GPU needs `TF_CUDNN_USE_AUTOTUNE=0`.** With TensorFlow's default cuDNN
  autotuning, each new input shape costs 2–40 s the first time it is seen. Varied-size
  stills averaged 7–10 s/image as a result, while fixed-size video frames were unaffected
  (0.27 s). Disabling autotuning removed the penalty, with no change to outputs.
- **GPU results match CPU results.** All 75 images produced identical face counts for every
  detector. Matching faces had embedding cosine 1.000 (minimum 0.97 for RetinaFace).
- **opencv and YuNet detection runs on the CPU in both setups.** Their GPU gain comes from
  ArcFace.
- **These are per-image speeds inside one long-running process.** The real pipeline starts a
  new process per step, which adds a fixed cost per step (see
  [End-to-end](#end-to-end-a-real-sketch-job)).

## Detection and matching

| | opencv | yunet | yunet_nocap | retinaface |
|---|---|---|---|---|
| **Accepted training portraits (576)** | | | | |
| Still pass the training gate | 100% | 97.0% | 96.9% | 95.1% |
| **Rejected training photos (144)** | | | | |
| No face found | 38.9% | 13.2% | 11.8% | 1.4% |
| Would now pass the training gate | 0% | 38.9% | 35.4% | 41.7% |
| **Testing group photos (240)** | | | | |
| Faces per photo | 5.07 | 2.74 | 5.38 | 7.35 |
| Low-information share of faces | 11.3% | 4.4% | 9.5% | 22.7% |
| Pass the 3–10 face gate | 100% | 56.2% | 90.8% | 82.1% |
| Actor matched | 54.2% | 38.8% | 57.5% | **61.2%** |
| Matched only by this / only by opencv | — | 3 / 40 | 15 / 7 | 22 / 5 |
| McNemar p vs opencv | — | <0.001 (worse) | 0.13 | 0.002 (better) |
| **Video frames (96)** | | | | |
| Detections | 76 | 20 | 28 | 63 |
| Low-information share | 85.5% | 5.0% | 35.7% | 60.3% |
| Usable faces | 11 | 19 | 18 | **25** |
| Frames with a usable face | 11.5% | 19.8% | 18.8% | **26.0%** |
| **Recognition on training portraits (48 actors)** | | | | |
| Rank-1 vs other actors' centroids | 98.1% | 97.2% | 97.5% | 97.2% |
| Image-pair AUC | 0.997 | 0.987 | 0.989 | 0.988 |
| Match rate at 0.1% false-match rate | 80.5% | 75.0% | 77.0% | 76.2% |
| **Compatibility with opencv** | | | | |
| Box width relative to opencv (median) | 1.00 | 0.80 | 0.80 | 0.80 |
| Same-face embedding cosine vs opencv (median / 10th pct) | — | 0.88 / 0.74 | 0.90 / 0.78 | 0.87 / 0.73 |

Notes on the table:

- **Test-photo matches.** When both opencv and RetinaFace matched the actor, they chose the
  same face in 124 of 125 photos.
- **YuNet portrait misses.** Both YuNet variants found no face in 7–8 of the 576 portraits.
  Their rank-1 figures cover only the portraits where a face was found.
- **Recognition numbers favour opencv.** These portraits were selected by the opencv
  pipeline, including face-count gates, outlier removal and identity anchoring on opencv
  embeddings. So the small opencv edge is not evidence it embeds better.
- **RetinaFace and the testing gate.** RetinaFace fails 18% of current group photos only
  because it counts tiny background faces. Counting only informative faces, opencv and
  RetinaFace both pass about 89%.

### What the disagreements look like

Random samples of faces found by only one detector were reviewed visually. The crops are not
included here because they are photos of real people.

- **Found only by opencv (vs RetinaFace):** almost entirely false positives. These were
  repeated hits on a wooden sign across video frames, mouths, ties and shirts, a road sign,
  and cartoon toys. All were flagged low-information.
- **Found only by RetinaFace:** almost entirely real faces. Most are small, profile or
  background faces; roughly ten per 80-crop sample were large, clear faces opencv missed.
- **Found only by uncapped YuNet:** real faces, including many large ones opencv missed.
- **Found only by opencv (vs uncapped YuNet):** about half junk and half real faces. Some of
  those faces are large (120–420 px), which fits DeepFace's note that YuNet struggles on
  large inputs. A middle cap or a two-scale pass would likely fix this but was not tested.

## End-to-end: a real sketch job

On 2026-10-06, a real SketchTV addition was run with production StarMapr, unchanged: opencv
on the Windows CPU. The sketch was Portlandia S8E3 "Command Center" (2:28, 720p), with Rachel
Bloom, Fred Armisen, Carrie Brownstein and Sean Tarjyoto. The published sketch used those
production results.

A RetinaFace shadow then repeated the same `01_run_headshot_detection.py` run in an
isolated clone on the WSL2 GPU. The clone differed only in the detector:
- The same video file.
- The same cached Google Image pages, so the training photo pool was identical and no new
  searches were made.
- Unchanged thresholds.
- 165 RetinaFace competitor models, each built from that actor's current production
  training photos.

| | Production (opencv, CPU) | Shadow (RetinaFace, GPU) |
|---|---|---|
| Work done | Trained 2 actors (Fred and Carrie reused existing models); 1 video pass of 50 frames; 143 s of new Google downloads | Trained all 4 actors from the cached pages; 2 video passes, 100 frames in all |
| Wall time | 9.0 min | 7.2 min |
| Frames sampled / with a face | 50 / 40 | 100 / 95 |
| Rachel Bloom | Trained on 22 photos (cohesion median 0.76). 5 headshots, scores 0.62–0.68 | Trained on 16 (0.71). 5 headshots, scores 0.61–0.67 |
| Fred Armisen | Existing model from an earlier run, 22 photos (0.83). 4 headshots, scores 0.51–0.86 | Retrained on 19 photos (0.75). 1 headshot, score 0.443, margin 0.086 |
| Carrie Brownstein | Existing model from an earlier run, 15 photos (0.66). 5 headshots, scores 0.49–0.55 | Retrained on 21 photos (0.70). 5 headshots, scores 0.45–0.52 |
| Sean Tarjyoto | Training abstained (3 coherent photos of 15) | Training abstained (3 of 15) |
| Faces rejected as not headshotable | 5 per actor | 11–18 per actor |
| Wrong-person acceptances | 0 | 0 |

Findings:

- **Fred lost 3 headshots to crop geometry.** His best face (frame 1246) scored 0.857 under
  RetinaFace, essentially the same as opencv's 0.856. It was still rejected as not
  headshotable: RetinaFace's taller, narrower box (133×186 px) makes the padded crop in
  `headshot_geometry` run past the frame edge. His two large close-ups (0.71) were rejected
  the same way.
- **Fred's dark command-center shots scored lower.** RetinaFace gave 0.37–0.41 there,
  against 0.55–0.71 from opencv. One of those shots also lost to a competitor. The
  alignment may be responsible. However, the two runs also used different Fred models:
  production reused an older model (22 photos, cohesion 0.83), while the shadow retrained
  from the two cached search pages (19 photos, 0.75). So this cannot be pinned on the
  detector alone.
- **The extra RetinaFace detections didn't become headshots.** It found faces in 95% of
  sampled frames (opencv: 80%), but the extra detections were mostly small or partial faces
  that the existing gates reject anyway.
- **RetinaFace crops come out squarer.** About 416×400 px against opencv's 4:3 crops (about
  544×408), because the crop is derived from the box. The crops are usable but framed
  differently.
This is a single sketch, so treat it as a smoke test rather than a measurement. It shows the
detector cannot be swapped in isolation: `headshot_geometry`, `MIN_FACE_SIZE` and the match
thresholds were all tuned on opencv's boxes and embeddings.

### Where the time goes

Step overhead, not detection, sets the pipeline's pace. Every step is a new Python process
that imports TensorFlow and loads its models.

| Step (mean per call) | opencv, CPU | RetinaFace, GPU |
|---|---|---|
| Training photo filter (`12`), about 20 photos | 18.3 s (14 calls) | 15.4 s (17 calls) |
| Frame face extraction (`32`), 50 frames | 19.8 s | 15.7 s |
| Test-photo evaluation (`20`) | 9.7 s | 6.9 s |

At opencv's 0.26 s per photo, detection is only about 5 s of each 18 s filter call; the
rest is startup. Allowing for the extra work, the GPU run was about as fast as production or
slightly faster:

| | Production | Shadow |
|---|---|---|
| Wall time | 397 s, excluding its Google downloads | 430 s |
| Page iterations | 14 | 17 |
| Testing pipelines | 1 | 3 |
| Video passes | 1 | 2 |

On the Windows CPU, RetinaFace would instead add about 5 s per photo or frame.

## Cost of switching

- **Models and caches.** `validation.embedding_spec()` records `detector='opencv'`, so
  existing caches and models invalidate automatically. All models and test results must
  then be regenerated, and old and new embeddings cannot be mixed.
- **Re-embedding time.** Roughly 5,400 current training and testing images need
  re-embedding: about 15 minutes at GPU speed, or about 8 hours for RetinaFace on the Windows CPU.
- **Thresholds.** Boxes are about 20% narrower for both YuNet and RetinaFace, which changes
  what `MIN_FACE_SIZE`, `isHeadshotable` and the headshot crop padding mean. The 3–10 testing
  gate should count only informative faces. `OPERATIONS_HEADSHOT_MATCH_THRESHOLD`,
  `OPERATIONS_MIN_MATCH_MARGIN` and `MAX_BLANK_SIMILARITY` should be re-checked on held-out
  data (see `90_benchmark_validation.py`).
- **Configuration.** The DeepFace detector backend is not configurable today; the string is
  hard-coded in `utils_deepface.py` and in `validation.embedding_spec()`.

## Next steps

If RetinaFace is pursued, in this order:

1. **Run StarMapr under WSL2 with the GPU** (see below). Without the GPU, RetinaFace is not
   affordable.
2. **Make the detector configurable.** Keep it recorded in `embedding_spec()`, so caches and
   models stay tied to the detector that produced them.
3. **Adapt the crop geometry and size gates.** Make `headshot_geometry`, `isHeadshotable`
   and `MIN_FACE_SIZE` account for RetinaFace's narrower, taller boxes. This is what cost
   Fred his best headshots. The 3–10 testing gate should count only informative faces.
4. **Rebuild every model, then recalibrate the thresholds.** The match threshold, margin and
   blank-similarity limit need re-checking on held-out positives and negatives with
   `90_benchmark_validation.py`.
5. **Re-run the end-to-end comparison on several sketches.** Switch only if headshot yield
   and quality are at least as good as opencv's.

Separately, and for any detector: step startup is a large share of pipeline time. Running
the per-page steps (`10`–`15`, `20`, `32`) in one long-lived process instead of a new
process per call would likely save minutes per actor. This has not been measured.

## Running on WSL2 with the GPU

This benchmark used two throwaway items in the Ubuntu distro, both of which can be deleted:
- A GPU environment at `~/starmapr-gpu`.
- An experiment-only clone at `~/StarMapr-retina`, used for the end-to-end shadow. In that
  clone the detector comes from `STARMAPR_DETECTOR`, and `STARMAPR_SKIP_TOOL_CHECKS` skips
  the node/ffmpeg preflight. Neither patch is in this repository.

Native-Windows TensorFlow (≥ 2.11) has no GPU support, so a GPU pipeline has to run under
WSL2. The WSL checkout at `/home/swax/StarMapr` mentioned in `AUTOMATED_VALIDATION.md` does
not exist in this machine's Ubuntu distro.

What worked:

```bash
uv venv --python 3.12 .venv
uv pip install "tensorflow[and-cuda]==2.21.0" tf-keras==2.21.0 keras==3.13.2 deepface==0.0.99 \
  retina-face==0.0.17 numpy==2.4.3 opencv-python-headless==4.13.0.92
uv pip uninstall opencv-python   # deepface pulls in the GUI build; keep the headless one
# TensorFlow found every pip CUDA library except libcusolver.so.11 until they were all on the path:
export LD_LIBRARY_PATH="$(ls -d .venv/lib/python3.12/site-packages/nvidia/*/lib | paste -sd:):/usr/lib/wsl/lib"
export TF_CUDNN_USE_AUTOTUNE=0   # otherwise each new image size costs 2-40 s
export DEEPFACE_HOME=/mnt/c/Users/johnm   # optional: share the Windows model weights
```

Moving the full pipeline to WSL would also need:

- A Linux environment for the project kept separate from the Windows `.venv`, for example
  via `UV_PROJECT_ENVIRONMENT`, with `tensorflow[and-cuda]` on Linux.
- The video tooling the CLI preflight checks for (Node, ffmpeg, yt-dlp, PO token provider)
  installed in Linux.
- A decision on whether the data stays on `/mnt/c`, which has slower I/O, or moves into
  the WSL filesystem.

## Caveats

- **Small video sample.** The video results come from two videos (96 frames), the weakest
  part of the data.
- **Thresholds held fixed.** All thresholds were tuned for opencv and were not changed. A
  recalibrated YuNet or RetinaFace may do better than shown.
- **No face-level ground truth.** "Real face vs junk" relies on the low-information gate and
  visual review of random samples.
- **Live jobs during collection.** StarMapr jobs ran during accuracy collection. The sample
  was frozen and excluded actors and videos that were in progress. Timing passes ran on an
  idle machine.
- **Patched variants.** `yunet_nocap` and the no-upscale RetinaFace variant patch DeepFace
  internals; they are not supported DeepFace options.
- **One end-to-end sketch.** The real-job comparison is a single 2.5-minute sketch with
  four actors, so treat it as a smoke test.
- **Shadow models were approximations.** The shadow's 165 competitor models were averaged
  from production's opencv-curated training photos; they did not go through a full
  RetinaFace training run. Production also reused older Fred and Carrie models, while the
  shadow retrained them from the currently cached search pages.
