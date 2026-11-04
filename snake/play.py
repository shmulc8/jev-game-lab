"""CLI for the Snake decision-loop experiment."""

import argparse
import json
import statistics
import sys
import time

import engine
import policies
import questions

CLEAR = "\x1b[H\x1b[2J"


def parse_args():
    p = argparse.ArgumentParser(description="Snake real-time decision-loop experiment")
    p.add_argument("--policy", choices=["random", "greedy", "flood", "jev"])
    p.add_argument("--mode", choices=["raw", "features", "composite"], default="raw")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--walls", type=int, default=0, help="fixed obstacle cells")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--rerank", metavar="FILE")
    p.add_argument("--weight-room", type=float, default=questions.DEFAULT_WEIGHT_ROOM)
    p.add_argument("--weight-food", type=float, default=questions.DEFAULT_WEIGHT_FOOD)
    p.add_argument("--min-confidence", type=float, default=questions.DEFAULT_MIN_CONFIDENCE)
    p.add_argument("--trace", metavar="FILE")
    p.add_argument("--pipeline", action="store_true", help="speculatively pipeline raw/features Jev requests")
    p.add_argument("--tick-ms", type=float, default=0.0, help="decision budget per tick; 0 disables the budget")
    p.add_argument("--dump-request", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.mode == "composite" and (args.tick_ms or args.pipeline):
        p.error("composite mode does not support --tick-ms or --pipeline")
    return args


def build_policy(name, args):
    if name == "random":
        return policies.RandomPolicy(seed=args.seed)
    if name == "greedy":
        return policies.GreedyPolicy()
    if name == "flood":
        return policies.FloodPolicy()
    if name == "jev":
        trace_path = args.trace
        if args.mode == "composite" and not trace_path:
            trace_path = f"runs/seed{args.seed}.jsonl"
        policy = policies.JevPolicy(
            mode=args.mode,
            min_confidence=args.min_confidence,
            weight_room=args.weight_room,
            weight_food=args.weight_food,
            trace_path=trace_path,
            pipeline=args.pipeline,
        )
        policy.tick_ms = args.tick_ms
        return policy
    raise ValueError(f"unknown policy {name}")


def dump_request(args):
    """Print the first request the jev policy would send for this seed, as sorted-key JSON, and exit."""
    game = engine.SnakeGame(grid=args.grid, seed=args.seed, walls=args.walls)
    policy = policies.JevPolicy(
        mode=args.mode,
        min_confidence=args.min_confidence,
        weight_room=args.weight_room,
        weight_food=args.weight_food,
        trace_path=None,
        pipeline=args.pipeline,
    )
    policy.tick_ms = args.tick_ms
    request = policy.build_request(game)
    print(json.dumps(request, sort_keys=True))


def percentile(data, pct):
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (pct / 100)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_game(game, policy, steps, render, fps):
    frame_delay = 1.0 / fps if fps else 0.0
    tick_times_ms = []
    started = time.perf_counter()
    while game.alive and game.steps < steps:
        if render:
            print(CLEAR + game.render(), flush=True)
        t0 = time.perf_counter()
        deadline_s = policy.tick_ms / 1000.0 if hasattr(policy, "tick_ms") and policy.tick_ms else None
        direction = policy.choose(game, deadline_s=deadline_s) if hasattr(policy, "tick_ms") else policy.choose(game)
        blocked_ms = (time.perf_counter() - t0) * 1000
        tick_times_ms.append(blocked_ms)
        game.step(direction)
        if deadline_s is not None:
            time.sleep(max(0.0, deadline_s - blocked_ms / 1000.0))
        if render and frame_delay:
            time.sleep(frame_delay)
    if render:
        print(CLEAR + game.render(), flush=True)
    death_cause = game.death_cause if not game.alive else "step_limit"
    if hasattr(policy, "stats"):
        policy.stats["effective_elapsed_s"] = time.perf_counter() - started
    return death_cause, tick_times_ms


def report_lines(policy_name, game, death_cause, tick_times_ms, policy):
    lines = [
        f"policy={policy_name} food_eaten={game.food_eaten} steps={game.steps} "
        f"length={len(game.snake)} death={death_cause}"
    ]
    if tick_times_ms:
        elapsed = policy.stats.get("effective_elapsed_s") if policy is not None else None
        dps = len(tick_times_ms) / elapsed if elapsed else 1000.0 * len(tick_times_ms) / sum(tick_times_ms)
        lines.append(f"  effective_steps_per_sec={dps:.2f}")
    if policy is not None and hasattr(policy, "stats"):
        s = policy.stats
        lat = s["latencies_ms"]
        p50 = percentile(lat, 50)
        p95 = percentile(lat, 95)
        mean_conf = statistics.mean(s["confidences"]) if s["confidences"] else None
        lines.append(
            f"  requests={s['requests']} speculative_requests={s['speculative_requests']} "
            f"pipeline_hits={s['pipeline_hits']} pipeline_misses={s['pipeline_misses']} "
            f"wasted_requests={s['wasted_requests']} budget_misses={s['budget_misses']} "
            f"skipped_calls={s['skipped_calls']} "
            f"input_tokens={s['input_tokens']} cost_usd={s['cost_usd']:.6f}"
        )
        considered = s["pipeline_hits"] + s["pipeline_misses"]
        lines.append(f"  pipeline_hit_rate={(100.0 * s['pipeline_hits'] / considered) if considered else 0.0:.1f}%")
        lines.append(
            f"  api_latency_p50_ms={p50:.1f} api_latency_p95_ms={p95:.1f} "
            f"api_decisions_per_sec={(1000.0 / statistics.mean(lat)) if lat else 0.0:.2f}"
        )
        blocked = s["blocked_ms"]
        lines.append(f"  blocked_ms_p50={percentile(blocked, 50):.1f} blocked_ms_p95={percentile(blocked, 95):.1f}")
        lines.append(
            f"  mean_confidence={round(mean_conf, 3) if mean_conf is not None else None} "
            f"confidence_fallbacks={s['confidence_fallbacks']} invalid_choices={s['invalid_choices']}"
        )
    return lines


def run_single(args):
    if not args.policy:
        print("error: --policy is required (or use --compare / --rerank)", file=sys.stderr)
        sys.exit(2)
    game = engine.SnakeGame(grid=args.grid, seed=args.seed, walls=args.walls)
    policy = build_policy(args.policy, args)
    if args.policy == "jev" and args.mode == "composite":
        policy.start_trace(game, args.steps, args.seed)
    death_cause, tick_times_ms = run_game(game, policy, args.steps, not args.quiet, args.fps)
    if args.policy == "jev" and args.mode == "composite":
        policy.finish_trace(game)
    if hasattr(policy, "close"):
        policy.close()
    for line in report_lines(args.policy, game, death_cause, tick_times_ms, policy):
        print(line)


def run_compare(args):
    runs = [
        ("random", {}),
        ("greedy", {}),
        ("flood", {}),
        ("jev", {"mode": "raw"}),
        ("jev", {"mode": "features"}),
    ]
    print(f"compare: seed={args.seed} steps={args.steps} grid={args.grid}\n")
    for name, overrides in runs:
        run_args = argparse.Namespace(**vars(args))
        for k, v in overrides.items():
            setattr(run_args, k, v)
        game = engine.SnakeGame(grid=run_args.grid, seed=run_args.seed, walls=getattr(run_args, 'walls', 0))
        policy = build_policy(name, run_args)
        death_cause, tick_times_ms = run_game(game, policy, run_args.steps, False, run_args.fps)
        if hasattr(policy, "close"):
            policy.close()
        label = f"{name}/{run_args.mode}" if name == "jev" else name
        for line in report_lines(label, game, death_cause, tick_times_ms, policy):
            print(line)
        print()


def load_trace(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _tuples(value):
    """Restore the nested tuples expected by random.Random.setstate."""
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def rerank(args):
    records = load_trace(args.rerank)
    meta = next(r for r in records if r.get("type") == "meta")
    ticks = [r for r in records if "tick" in r]
    summary = next((r for r in records if r.get("type") == "summary"), None)

    game = engine.SnakeGame(grid=meta["grid"], seed=meta["seed"])
    # Older wall-free traces do not carry wall coordinates.
    if "walls" in meta:
        game.walls = {tuple(cell) for cell in meta["walls"]}
        game.food = tuple(meta["initial_food"]) if meta["initial_food"] is not None else None
        game.rng.setstate(_tuples(meta["rng_state"]))
    diverged_at = None
    tick_idx = 0
    while game.alive and game.steps < meta["steps_limit"]:
        moves = game.survivable_moves()
        if len(moves) <= 1:
            direction = moves[0]["direction"] if moves else policies.flood_choice(game)
            game.step(direction)
            continue
        if tick_idx >= len(ticks):
            diverged_at = game.steps
            break
        rec = ticks[tick_idx]
        tick_idx += 1
        survivable = {m["direction"] for m in moves}
        recorded = set(rec["candidates"].keys())
        if rec["tick"] != game.steps or survivable != recorded:
            diverged_at = rec["tick"]
            break
        if rec["confidence_fallback"]:
            direction = policies.flood_choice(game)
        else:
            direction = max(
                rec["candidates"],
                key=lambda d: policies.weighted_value(rec["candidates"][d], args.weight_room, args.weight_food),
            )
        changed_action = direction != rec["chosen"]
        game.step(direction)
        if changed_action:
            diverged_at = rec["tick"]
            break

    flood_ticks = 0
    if diverged_at is not None:
        while game.alive and game.steps < meta["steps_limit"]:
            game.step(policies.flood_choice(game))
            flood_ticks += 1

    print(f"rerank file={args.rerank} weight_room={args.weight_room} weight_food={args.weight_food}")
    print(f"original recorded weights: room={meta['weight_room']} food={meta['weight_food']}")
    if summary:
        print(f"original outcome: food_eaten={summary['food_eaten']} steps={summary['steps']} death={summary['death_cause']}")
    print(f"reranked outcome:  food_eaten={game.food_eaten} steps={game.steps} death={game.death_cause or 'step_limit'}")
    if diverged_at is not None:
        print(
            f"trajectory diverged at tick {diverged_at} "
            f"(changed action or trace mismatch); the remaining "
            f"{flood_ticks} ticks used the flood-fill fallback, zero API calls"
        )
    else:
        print("replay matched recorded judgments for the entire game, zero API calls")


def main():
    args = parse_args()
    if args.dump_request:
        if args.policy != "jev":
            print("error: --dump-request requires --policy jev", file=sys.stderr)
            sys.exit(2)
        dump_request(args)
        return
    if args.rerank:
        rerank(args)
        return
    if args.compare:
        run_compare(args)
        return
    run_single(args)


if __name__ == "__main__":
    main()
