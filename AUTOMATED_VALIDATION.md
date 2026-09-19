# Automated headshot validation

This change adds automatic rejection and bounded retries, with no mandatory human
face-review queue. The WSL checkout at `/home/swax/StarMapr` includes the merged
validation and API/JSON changes. Its existing API server and downloader changes
are preserved. No production models, portraits or `.env` settings were changed
by this source update; model migration and real-photo calibration remain separate.

## Behavior

Readiness update: valid current models are reused; legacy/stale models are retrained
on demand. Training/testing data are archived under `.history`, the previous
promoted model remains until replacement succeeds, and AWS cache/budget usage is
preserved across training retries. Do not delete budget databases to retry.

The top-level CLI now checks Node, ffmpeg, yt-dlp and AWS permission (one generated
blank-image request) and establishes a decodable video download before training.
A metadata-only video folder is not a successful download cache. AWS preflight
requests are separate from the 40-request training/video budget.

Pass the complete on-screen cast to a single `--actors` invocation. Each actor
receives a deterministic share of the video request cap, including across retry
passes/processes. `actor_budget_exhausted` ends that actor's attempts without
spending other actors' shares. Confirmed voice-only roles do not need extraction.

`STARMAPR_PROGRESS` lines report phase, actor, counts and timestamps. CLI status is
saved under `06_jobs/cli-*/progress.json`; API jobs expose `progress` in their
status response. Full subprocess logs live beside the progress file; captured
error tails are bounded. Waiting heartbeats indicate a live wait, not useful work.

- Training must meet both the image minimum and leave-one-out cohesion gates.
  Each embedding is normalized before averaging. A larger incoherent group cannot
  replace a qualifying group; all-noise clustering produces zero eligible images.
- Model quality reports are tied to the model file's SHA-256 and the explicit
  ArcFace/detector/normalization/DeepFace version. Legacy models are preserved, but
  cannot produce new portraits until successfully retrained with this code.
- Every candidate must exceed the local similarity threshold and beat every other
  available actor model by the configured margin. Legacy competitor models can
  veto a match; they cannot authorize one. Missing competitors limit this check.
- Candidates need similar appearances in at least two separated video frames.
  This is corroboration, not full tracking or independent identity evidence.
- An optional AWS gate checks single-face training images and each final headshot.
  It requires the exact catalog name (case/space normalized), or an explicit
  celebrity ID override, and sufficient confidence. It
  rejects unknown identities, multiple faces, mismatches and unavailable service.
- Only successful actors are removed from the retry queue. Actors with no match
  get at most five frame-sampling passes. Configuration, unavailable-model and
  cloud-budget outcomes stop immediately. Crops are limited to five per actor;
  at most ten candidate verifications are considered in each extraction pass.
- `headshots/<actor>/result.json` is authoritative. Consumers must use its listed
  `headshots`, not count files or infer success from exit code alone. Old image
  files are removed on a completed abstention. The video-level summary is
  `headshot-results.json`.

Typical statuses are `accepted`, `no_reliable_headshot`, `model_unvalidated`,
`missing_identity_mapping`, `verifier_unavailable` and `budget_exhausted`.
Expected abstentions return exit 0 so an optional portrait can be omitted. File,
cache and extraction failures still return nonzero. AWS failures are explicitly
reported abstentions, never a switch to local-only output. No portrait does not
mean the actor is absent from the sketch; cast metadata remains independent.

## AWS setup

Install the optional dependency:

```sh
uv sync --extra aws
```

The AWS extra includes CRT, which Boto3 needs for console-login credentials.
To check API permission without sending any photos, make one request with a
generated blank PNG:

```sh
uv run --extra aws python 91_check_aws_connection.py --region us-east-1
```

Use the normal SDK credential chain. A console browser session alone does not
authenticate a Python process. On supported AWS CLI versions, `aws login` can
use a console sign-in for temporary development credentials. For unattended agent
hosts use an appropriate workload identity/role; interactive sessions expire.
Do not embed credentials in code or agent prompts.

The operation requires `rekognition:RecognizeCelebrities`. This branch does not
create IAM identities, grant permissions, make buckets or provision a server.
It sends JPEG bytes directly to Rekognition when explicitly enabled.

Ordinary actors use their exact catalog names automatically. No manual mapping
is needed for each new actor. Namesakes and alternate/stage names need an explicit
ID from your trusted catalog; do not guess or fuzzy-match. An optional identity
map overrides name matching (do not copy this placeholder ID):

```json
{"example_actor": "PROVIDER_CELEBRITY_ID"}
```

Then configure `.env` using `.env.example`:

```dotenv
CELEBRITY_VERIFIER=aws_required
AWS_REGION=us-east-1
AWS_CELEBRITY_ID_MAP=/absolute/path/to/celebrity-ids.json
AWS_CELEBRITY_MIN_CONFIDENCE=99
AWS_CELEBRITY_MAX_REQUESTS=40
```

`off` makes no AWS calls and reports `local_only` assurance. Required mode never
uses fuzzy name matching. An explicit ID takes priority over a matching name.
Neither names nor provider IDs guarantee against a provider misidentification.
Budget usage
and compact responses are stored in `celebrity-cache.sqlite` in each training
directory or video folder. The video budget is shared across all actors and
retries, including separate processes. Failed requests consume budget too. One
SDK attempt per request, five-second connection and fifteen-second read timeouts
bound network retries. Cached responses are re-evaluated against the current ID
and confidence threshold. Deleting that database or retraining from scratch
resets that scope's budget. API calls may incur charges; no pricing assumption
is built into the request cap.

Official references:
[RecognizeCelebrities](https://docs.aws.amazon.com/boto3/latest/reference/services/rekognition/client/recognize_celebrities.html),
[AWS CLI console login](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html).

## Calibration and rollout

The initial cohesion, ambiguity and corroboration thresholds are conservative
starting values, **not measured accuracy guarantees**. Cohesion can find mixed
reference sets but cannot establish that a coherent cluster has the correct name.
The old four-detection group-photo test remains a pipeline smoke test; it is not
a labeled accuracy test. An AWS confidence score is not a guarantee either.

1. Retrain representative models in a separate data directory using this branch.
   Enable the AWS gate there if desired. Keep the existing production directory
   intact while evaluating changes. Production CLI jobs now retrain stale models
   on demand, so perform this calibration before launching those jobs.
2. Use a held-out set of known positives and hard negatives, including similar
   cast members, varying ages, costumes and low-resolution frames. Do not use
   training images or near-duplicates as held-out evidence. Include previously
   working cases to measure lost coverage as well as incorrect acceptances.
3. Run the benchmark below yourself. It measures single-photo gates separately
   from temporal corroboration and never promotes a model or alters the site.
4. Choose thresholds based on observed false positives, recall and abstentions.
   A benchmark is a one-time validation dataset, not a per-sketch review queue.
5. Update the agent handoff to read `result.json`, upload only listed accepted
  portraits and continue without optional images when a model abstains. Deploy
   only after evaluating coverage. Do not lower thresholds just to fill every slot.

Manifest example (paths relative to the manifest; use single-face images):

```json
[
  {"actor": "Example Actor", "image": "heldout/positive.jpg", "expected_match": true},
  {"actor": "Example Actor", "image": "heldout/negative.jpg", "expected_match": false}
]
```

From the isolated StarMapr data/code directory:

```sh
uv run python 90_benchmark_validation.py benchmark.json --output local-results.json
uv run --extra aws python 90_benchmark_validation.py benchmark.json --output aws-results.json --aws
```

Both labels are required for each actor. The benchmark rejects duplicate image
bytes and any image hash found in the models' training manifests. Near-duplicate
separation must be maintained in dataset preparation. Labels are supplied by the
manifest author, not by the recognizer being tested. Results report false
positives/negatives and cloud abstentions. Image embedding caches are created next
to benchmark photos, so use a disposable copy of the dataset. No real-photo
benchmark or live celebrity-identification request was executed in preparing this
change; mock tests and the blank-image connectivity check do not establish
real-photo accuracy.

## Regression tests

```sh
uv run python -m unittest discover -s tests -v
```

Fixtures contain synthetic vectors and blank generated images. AWS responses
are mocked; the suite makes no network requests and needs no API keys.

Preparation checks on September 18, 2026: 31 synthetic regression tests passed,
the dependency lock passed `uv lock --check --offline`, and the live blank-PNG
permission check returned HTTP 200 with zero faces in `us-east-1`. The existing
local AWS login was refreshed through the signed-in browser. Its temporary
credentials were not copied into StarMapr or onto the WSL/remote agent hosts.

The WSL merge adds five API/CLI regression tests (36 total), preserves the existing
FastAPI/uvicorn and yt-dlp/bgutil pins, and adds only the five optional AWS SDK
packages. API artifacts now come from the current job's accepted validation
results; stale image files or a failed job cannot silently publish headshots.
The CLI preserves `--json` and includes `headshot_outcomes` and `outcome` so API
clients can distinguish omitted portraits from execution failures.

## WSL temporary AWS connection

Local testing is configured in `/home/swax/StarMapr/.env` with
`CELEBRITY_VERIFIER=aws_required`, region `us-east-1`, confidence threshold 99,
and a persistent limit of 40 requests per training directory/video.
The SDK uses the existing Windows `default` profile through
`AWS_CONFIG_FILE=/mnt/c/Users/johnm/.aws/config` and
`AWS_LOGIN_CACHE_DIRECTORY=/mnt/c/Users/johnm/.aws/login/cache`.
No access keys or copied credential-cache files are stored in this repository.

When this temporary sign-in expires, refresh it from Windows with
`aws login --profile default --region us-east-1`. WSL uses the refreshed cache.
This is the local testing setup, not an unattended workload identity.
The connection-check script loads `.env`, so from this checkout run:

```sh
uv run --locked --extra aws python 91_check_aws_connection.py
```

Existing legacy models are migrated/retrained on demand by the next CLI job
before the strict quality checks allow new headshots. Enabling AWS does not
bypass that gate. The readiness update passes 44 synthetic regression tests;
no real-photo benchmark or model retraining was run during implementation.
