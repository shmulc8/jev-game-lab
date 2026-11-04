# Jev Snake decision loop

A small Snake experiment comparing code policies with TypeSafe/Jev judgments.
The engine owns collisions, food, BFS distance and flood-fill area. Jev chooses
between survivable moves in `raw` or `features` mode; `composite` mode combines
separate room and food scores with weights controlled by code.

Python 3.9+; standard library only. Run these commands from the repository root:

```sh
# Offline, no API key required
python3 snake/play.py --policy flood --seed 7 --steps 150 --quiet
python3 snake/play.py --policy greedy --seed 7 --steps 150 --quiet
python3 snake/play.py --policy random --seed 7 --steps 150 --quiet
```

On this seed and the default 12×12 board, greedy and flood each eat 15 food in
150 steps; random eats 0. This is one deterministic example, not a benchmark
across seeds or evidence that the policies are generally equivalent.

Live commands below send synthetic game state to `api.typesafe.ai` and may
incur charges. Supply `TYPESAFE_API_KEY` in your environment. The client does
not load `.env` files. `--compare` also includes live API policies.

```sh
python3 snake/play.py --policy jev --mode features --steps 30 --quiet
python3 snake/play.py --policy jev --mode composite --steps 30 --trace runs/snake.jsonl --quiet
python3 snake/play.py --rerank runs/snake.jsonl --weight-room 0.9 --weight-food 0.1
```

`--min-confidence` routes low-confidence answers to flood-fill. Immediate
collision filtering and ticks with at most one survivable move run in code.
Question wording and score levels live in `questions.py`.

Composite traces record judgments and the initial board/random state. Replaying
with original weights reproduces the trajectory. With different weights, the
first changed action ends judgment replay; subsequent ticks use flood-fill.
The resulting food count therefore describes a hybrid replay, not how Jev
would play an unobserved trajectory. Replay makes no API calls.

Raw/features modes also expose experimental `--pipeline` speculation and
`--tick-ms` decision budgets. Speculation can increase paid requests, and
outstanding calls finish during shutdown. Composite mode rejects these options.
Latency and cost summaries are observational; cost uses a fixed input-token
price estimate in `client.py`, not current billing verification. No historical
live-API performance claims are bundled here.
