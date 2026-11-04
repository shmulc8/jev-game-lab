# JEV Game Lab

Snake and Tetris experiments exploring where typed AI decisions fit inside a game loop. By Shmulik Cohen.

[Watch the recorded Snake runs](https://shmulc8.github.io/jev-game-lab/snake.html) · [Watch the recorded Tetris runs](https://shmulc8.github.io/jev-game-lab/tetris.html)

Jev is [TypeSafe AI's System One model](https://typesafe.ai/blog/introducing-system-one-models-and-jev). This is an independent experiment, not an official TypeSafe product. The game engines and code baselines own legal-move enumeration and state transitions; Jev chooses among the supplied candidates.

## Watch without an API key

The `viz/` folder is a static replay site. It includes six historical, single-seed recordings: Snake with flood-fill, Jev raw state, and Jev features; Tetris with an oracle heuristic, Jev board input, and Jev rich features. Each run stays separate and retains its recorded summary. Playback makes no model requests and cannot spend API credits.

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory viz
```

Open http://127.0.0.1:8000/ or open `viz/index.html` directly. Rebuild the HTML from the supplied JSON recordings with `python3 -m viz.replay`.

These are recorded experiments, not live inference or general performance benchmarks. Some recordings predate the publication cleanup. The saved `jev-latest` model alias does not identify an immutable model version. Token-cost figures are historical estimates, not a statement of current provider pricing. Confidence values are model outputs, not evidence that they are calibrated on these games.

## Run offline baselines

Python 3.10 or newer; standard library only. No dependencies to install.

```sh
python3 snake/play.py --policy flood --steps 150 --seed 7 --quiet
python3 play.py --agent heuristic --pieces 40 --seed 7 --quiet
python3 play.py --agent oracle --pieces 40 --seed 7 --quiet
python3 -m unittest discover -s tests -v
```

The Tetris `oracle` is a code heuristic that inspects legal placements, not a perfect-information solver. Rich-mode Jev normally receives features and a code-ranked top-12 candidate shortlist. Results describe this hybrid setup, not unaided spatial reasoning. Snake similarly distinguishes raw-state, feature-assisted, and composite-score policies. See [Snake notes](snake/README.md).

## Optional live model experiments

Only explicit Jev commands and `--compare` use the external TypeSafe API and may incur charges. Set `TYPESAFE_API_KEY` in your environment, then choose a small run:

```sh
python3 snake/play.py --policy jev --mode features --steps 10 --seed 7 --quiet
python3 play.py --agent jev --mode rich --pieces 10 --seed 7 --quiet
```

No API key, `.env`, private run folder, or live-control web server is included. Do not expose a key-backed game controller as a public server. The static GitHub Pages site serves only the recorded replay files.

## Publication checks

The publication copy fixes empty replay output, clipped Tetris boards, preview cropping, pipeline alignment, Snake replay-divergence handling, and unsupported composite-mode budget combinations. Offline regressions cover those paths. The archived model runs have not been rerun against the current provider API.
