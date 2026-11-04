"""CLI runner for Tetris agents: random, heuristic, oracle, and Jev (board/preview/rich modes)."""

import argparse
import json
import time

import agents
import tetris


def build_agent(name, mode, seed, payload="top12", pipeline=False):
    if name == "random":
        return agents.RandomAgent(seed=seed)
    if name == "heuristic":
        return agents.HeuristicAgent()
    if name == "oracle":
        return agents.OracleAgent()
    if name == "jev":
        return agents.JevAgent(mode=mode, payload=payload, pipeline=pipeline)
    raise ValueError(f"unknown agent: {name}")


def label_for(name, mode):
    return f"jev-{mode}" if name == "jev" else name


def run(name, mode, pieces, seed, quiet, payload="top12", pipeline=False):
    board = tetris.Board()
    bag = tetris.PieceBag(seed)
    agent = build_agent(name, mode, seed, payload=payload, pipeline=pipeline)
    label = label_for(name, mode)

    lines_cleared_total = 0
    placed = 0
    topped_out = False
    started = time.perf_counter()

    try:
        for i in range(pieces):
            piece = bag.next()
            next_pieces = [bag.peek(k) for k in range(1, agents.NEXT_QUEUE_LEN + 1)]
            next_piece = next_pieces[0]
            if not board.legal_placements(piece):
                topped_out = True
                break
            rot, col = agent.choose(
                board, piece, next_piece, next_pieces=next_pieces,
                lines_cleared_total=lines_cleared_total,
            )
            lines_cleared = board.drop(piece, rot, col)
            lines_cleared_total += lines_cleared
            placed += 1
            if not quiet:
                print(
                    f"--- {label} piece {placed}/{pieces}: {piece} -> "
                    f"rot={rot} col={col} (lines cleared: {lines_cleared}) ---"
                )
                print(board.render())
                print()
    finally:
        if hasattr(agent, "close"):
            agent.close()

    return {
        "agent": label,
        "pieces_requested": pieces,
        "pieces_placed": placed,
        "lines_cleared": lines_cleared_total,
        "topped_out": topped_out,
        "holes": board.hole_count(),
        "max_height": board.max_height(),
        "aggregate_height": board.aggregate_height(),
        "bumpiness": board.bumpiness(),
        "stats": {**agent.stats, "wall_time_s": time.perf_counter() - started},
    }


def dump_request(args):
    """Print the first request a Jev agent would send for this seed, as sorted-key JSON, and exit."""
    board = tetris.Board()
    bag = tetris.PieceBag(args.seed)
    agent = agents.JevAgent(mode=args.mode, api_key="offline-request-preview", payload=args.payload, pipeline=False)
    piece = bag.next()
    next_pieces = [bag.peek(k) for k in range(1, agents.NEXT_QUEUE_LEN + 1)]
    request = agent.build_request(board, piece, next_pieces[0], next_pieces=next_pieces, lines_cleared_total=0)
    print(json.dumps(request, sort_keys=True))


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def percentile(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values) - 1, int((len(values) - 1) * p / 100))]


def print_report(result):
    stats = result["stats"]
    print(f"=== final report: {result['agent']} ===")
    print(f"pieces placed: {result['pieces_placed']}/{result['pieces_requested']}")
    print(f"lines cleared: {result['lines_cleared']}")
    print(f"topped out: {result['topped_out']}")
    print(f"holes: {result['holes']}")
    print(f"max height: {result['max_height']}")
    print(f"aggregate height: {result['aggregate_height']}")
    print(f"bumpiness: {result['bumpiness']}")
    if "confidences" in stats:
        print(f"requests: {stats['requests']}")
        print(f"input tokens: {stats['input_tokens']}")
        print(f"cost usd: {stats['cost_usd']:.6f}")
        if stats["requests"]:
            print(f"input tokens per decision: {stats['input_tokens'] / stats['requests']:.1f}")
        print(f"mean confidence: {mean(stats['confidences'])}")
        print(f"mean topout score: {mean(stats['topout_scores'])}")
        print(f"invalid choices: {stats['invalid_choices']}")
        print(f"candidate capping active: {stats['capped']}")
        if stats.get("latencies_ms"):
            print(f"api p50 ms: {percentile(stats['latencies_ms'], 50):.1f}")
            print(f"api p95 ms: {percentile(stats['latencies_ms'], 95):.1f}")
        if stats.get("speculative_requests"):
            print(f"speculative requests: {stats['speculative_requests']}")
            print(f"pipeline hits: {stats['pipeline_hits']}")
            print(f"pipeline misses: {stats['pipeline_misses']}")
            print(f"wasted requests: {stats['wasted_requests']}")
            if stats["blocked_ms"]:
                print(f"blocked p50 ms: {percentile(stats['blocked_ms'], 50):.1f}")
                print(f"blocked p95 ms: {percentile(stats['blocked_ms'], 95):.1f}")
            elapsed = stats.get("wall_time_s")
            if elapsed:
                print(f"effective decisions/sec: {result['pieces_placed'] / elapsed:.3f}")


def print_compare_table(results):
    headers = [
        "agent", "placed", "lines", "topped", "holes", "max_h",
        "requests", "in_tokens", "tok/req", "cost_usd", "mean_conf", "invalid",
    ]
    rows = []
    for r in results:
        stats = r["stats"]
        requests = stats.get("requests", 0)
        input_tokens = stats.get("input_tokens", 0)
        tok_per_req = f"{input_tokens / requests:.0f}" if requests else "-"
        rows.append([
            r["agent"],
            f"{r['pieces_placed']}/{r['pieces_requested']}",
            str(r["lines_cleared"]),
            str(r["topped_out"]),
            str(r["holes"]),
            str(r["max_height"]),
            str(requests),
            str(input_tokens),
            tok_per_req,
            f"{stats.get('cost_usd', 0.0):.6f}",
            f"{mean(stats.get('confidences', [])):.3f}" if stats.get("confidences") else "-",
            str(stats.get("invalid_choices", 0)) if "invalid_choices" in stats else "-",
        ])
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line_fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(line_fmt.format(*headers))
    for row in rows:
        print(line_fmt.format(*row))


def main():
    parser = argparse.ArgumentParser(description="Play Tetris with random, heuristic, oracle, or Jev agents.")
    parser.add_argument("--agent", choices=["random", "heuristic", "oracle", "jev"])
    parser.add_argument("--mode", choices=["board", "preview", "rich"], default="preview")
    parser.add_argument("--payload", choices=["full", "nostate", "top12", "both"], default="top12")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--pieces", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--dump-request", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.dump_request:
        if args.agent != "jev":
            parser.error("--dump-request requires --agent jev")
        dump_request(args)
        return

    if args.compare:
        configs = [
            ("random", None),
            ("heuristic", None),
            ("oracle", None),
            ("jev", "board"),
            ("jev", "preview"),
            ("jev", "rich"),
        ]
        results = []
        for name, mode in configs:
            print(f"running {label_for(name, mode)} ...")
            results.append(run(
                name, mode, args.pieces, args.seed, quiet=True,
                payload=args.payload if name == "jev" else "full",
                pipeline=args.pipeline if name == "jev" else False,
            ))
        print()
        print_compare_table(results)
        return

    if not args.agent:
        parser.error("--agent is required unless --compare is given")

    result = run(args.agent, args.mode, args.pieces, args.seed, args.quiet, args.payload, args.pipeline)
    print_report(result)


if __name__ == "__main__":
    main()
