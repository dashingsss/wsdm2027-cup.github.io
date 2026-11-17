# Agentic Search

Develop algorithms that retrieve and adapt historical trajectories to help a fixed agent solve new tasks. Your algorithm returns guidance; the official agent chooses actions, and the environment computes the score.

| Dataset | Task | Public trajectories | Development tasks |
| --- | --- | ---: | ---: |
| ALFWorld | Find, process and place household objects | 1,000 | 50 |
| WebShop | Search for products, select options and make a purchase | 1,000 | 50 |
| HotpotQA | Search articles, connect evidence and answer questions | 1,000 | 50 |

## 1. Get the code and host tools

Use **Linux x86_64 with Docker Engine, Docker Compose v2, Git, Python 3.10+ and unzip**. Python needs `venv` support for the data download command. Use a maintained Docker release with its default security settings. The host does not need Conda, Java or a GPU. On Windows/macOS, Docker Desktop must use Linux containers; the target architecture is `linux/amd64`.

```bash
git clone https://github.com/wsdm2027-cup/wsdm2027-cup.github.io.git
cd wsdm2027-cup.github.io/starter-kit
docker version
docker compose version
python3 --version
```

Docker builds two images: the **official image** runs the agent, environments and scoring; the **algorithm image** runs your method and its dependencies. The launcher starts and connects them automatically.

| Dependency | Prepared by Docker |
| --- | --- |
| Python 3.11.13 and JDK 11 | Official image |
| openJiuwen 0.1.18, ALFWorld 0.4.2 and other Python packages | Pinned `requirements.txt` |
| WebShop environment and HotpotQA scoring script | Official Git repositories, pinned and checked by `docker/source-lock.json` |
| BGE-small-en-v1.5 Dense encoder | Algorithm image; ONNX inference on CPU |

No manual third-party clone is needed. Source code is downloaded inside the official image, not into your host checkout. Builds need Internet access for these sources, packages and model files.

## 2. Download the data

The released trajectory corpus is also provided during hidden evaluation. You may collect or generate additional trajectories for training and local experimentation. Development tasks cover different task types and difficulty levels. Hidden tasks are designed to introduce harder problems and new combinations of difficulties; you may generate additional tasks to assess your method.

Open the [Hugging Face dataset page](https://huggingface.co/datasets/agentic-search-review-team/agentic-search-review), sign in and request access (automatically approved). Then log in with the same account using a read token:

```bash
python3 -m venv .download-env
.download-env/bin/python -m pip install huggingface_hub
.download-env/bin/hf auth login
.download-env/bin/hf download agentic-search-review-team/agentic-search-review data-resources.zip SHA256SUMS --repo-type dataset --local-dir downloads
(cd downloads && sha256sum -c SHA256SUMS)
unzip downloads/data-resources.zip -d downloads/unpacked
```

Move the extracted `data/` and `resources/` directories beside `run.py`. If the archive has a containing directory, use the directories inside it. The layout should be:

```text
starter-kit/
├── run.py
├── data/
│   ├── alfworld/   # tasks.jsonl and trajectories.jsonl
│   ├── webshop/    # tasks.jsonl and trajectories.jsonl
│   └── hotpotqa/   # tasks.jsonl and trajectories.jsonl
└── resources/      # game files, product catalog/search index and article database
```

Use the complete matching bundle; do not rebuild its indexes. Extracted data/resources need about 18 GB, with additional space for images, build caches and results. To keep existing data elsewhere, add `--data /absolute/path/to/data --resources /absolute/path/to/resources` to every `check`, `verify` and `run` command below.

## 3. Run the baselines

From `starter-kit/`:

```bash
python3 docker/launch.py build
python3 docker/launch.py check
python3 docker/launch.py verify
```

`check` verifies the data layout. `verify` checks the supplied baseline services, container communication and recovery. Neither calls the language model.

Provide your own OpenRouter API key on the host (Bash), then run one development task per dataset:

```bash
read -rsp 'OpenRouter API key: ' OPENROUTER_API_KEY
export OPENROUTER_API_KEY
python3 docker/launch.py run --dataset all --method algorithms/baselines/bm25.py --smoke 1
```

For all 50 tasks in a dataset, omit `--smoke`:

```bash
python3 docker/launch.py run --dataset alfworld --method algorithms/baselines/bm25.py
```

Use `--dataset all` for all three datasets. Other method paths are `algorithms/baselines/no_retrieval.py`, `algorithms/baselines/random.py` and `algorithms/baselines/dense_static.py`.

Keep the supplied model settings: **Qwen3.5-9B via OpenRouter/Together, temperature 0, reasoning disabled and provider fallback disabled**. Your key needs access to that provider/model. Only the official container receives it; `OPENROUTER_API_KEY_FILE` is also supported. Do not put keys in source code or images.

The launcher removes the containers after execution. Results remain in `results/containers_*/`; each dataset produces `episodes.jsonl`, `attempts.jsonl` and `summary.json`. A smoke run is marked `rankable: false` and is only a quick execution check.

## 4. Develop your algorithm

Start with [algorithms/methods/example.py](algorithms/methods/example.py) and implement `create_algorithm(config)`. The [interface](algorithms/base.py) provides:

- `prepare(corpus)`: build an index of the public trajectories once per dataset, except after recovery.
- `reset(initial, corpus)`: clear task-local state while retaining the index.
- `get_guidance(state)`: return `Guidance(text, retrieved_ids)` before each agent model call.

The state includes the task instruction, observation, available actions and permitted history. You may retrieve again, combine experiences or summarize them on each call. See [protocol.py](algorithms/protocol.py) for the exact fields. The algorithm receives JSON inputs, not access to the official filesystem, hidden answers or API key.

Put your method and helper modules under `algorithms/methods/`. Keep the supplied interface and service files. Add Python dependencies to `docker/requirements.algorithm.txt`; add system dependencies to `docker/Dockerfile.algorithm`. Models must work on CPU and be included during image construction: the algorithm container runs offline. Download weights in the Dockerfile, or copy local weights from a directory under `algorithms/`, which the build context includes.

After changing code, dependencies or model files, rebuild your image and run it. The host source directory is not mounted into the algorithm container:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.algorithm -t my-algorithm:cpu .
python3 docker/launch.py run --dataset hotpotqa --algorithm-image my-algorithm:cpu --method algorithms/methods/example.py --smoke 1
```

Replace the example path with your own method path. The image must retain the JSON service entry point and port 8765, work as UID 10002 with a read-only root filesystem, and declare no Docker `VOLUME`. Write temporary files to `/cache` or `/tmp`; both are cleared when the container is replaced. Organizers use their own official image and configuration, so changes to the official agent, environments or scoring are not part of your submission.

## 5. Submit your method

Export the algorithm image you used for development:

```bash
mkdir -p submission
docker save my-algorithm:cpu -o submission/algorithm-image.tar
```

Include these files:

```text
submission/
├── algorithm-image.tar
├── algorithms/                   # source matching the exported image
├── docker/
│   ├── Dockerfile.algorithm
│   └── requirements.algorithm.txt
├── .dockerignore
└── README.md
```

Include any additional files required by your Dockerfile. In the submission README, list:

- Image name and tag, for example `my-algorithm:cpu`.
- Method path inside the image, for example `algorithms/methods/my_method.py`.
- The build command and any extra setup needed to reproduce the image.
- Any additional models, their versions/sources and whether they were trained using extra data; write `none` if not applicable.

Do not include API keys, the official image or the downloaded benchmark data. The submission channel will be announced on the [challenge website](https://wsdm2027-cup.github.io/agenticsearch/).

## 6. Scores and runtime limits

ALFWorld uses task success (0/1), WebShop the official reward (0–1), and HotpotQA normalized answer exact match (0/1). The ranking score is the equally weighted mean of the three dataset means. Time and tokens are diagnostics only.

Each attempt allows **300 seconds, 50 actions and 50 model calls**. The deadline includes initialization, task-local computation and model waits. Shared index preparation is outside it and has no automatic timeout; stalled preparation may be stopped manually. There is no global retrieval-count limit. Model context capacity and a 64 MiB JSON message bound still apply; overlong inputs are not silently truncated.

Local defaults are **4 CPUs and 8 GB RAM**, with `/cache` up to 2 GB and `/tmp` up to 1 GB. `--cpus` and `--memory` adjust local allocations; compare methods under the same allocation. Host swap limiting must be supported to enforce the combined RAM/swap ceiling. Final contest resource allocations will be announced separately.

Timeouts and algorithm errors are recorded. Unresolved service errors make a run incomplete; do not treat missing tasks as a valid score. A task may be retried for a service failure, but normal wrong answers are not rerun to improve scores. Containers are separated by mounts, permissions and networking; they share the host kernel.

This starter kit runs public development tasks. Hidden tasks, GT and organizer evaluation tooling are not included.
