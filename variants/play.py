"""Run standard, adversarial Hatetris, or pentomino Tetris."""

import argparse
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agents
import tetris
from variants import agents as variant_agents
from variants.engine import Board, PieceBag, all_orientations
from variants.hatetris import HatetrisGenerator


PENTOMINO_BASES = {
    "F": [(1, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
    "I": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    "L": [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3)],
    "N": [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)],
    "P": [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)],
    "T": [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
    "U": [(0, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
    "V": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
    "W": [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
    "X": [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],
    "Y": [(1, 0), (0, 1), (1, 1), (1, 2), (1, 3)],
    "Z": [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
}
PENTO_PIECES = {k: all_orientations(v) for k, v in PENTOMINO_BASES.items()}


def make_game(variant, seed):
    if variant == "pento":
        return Board(PENTO_PIECES, 12, 20), PieceBag(seed, PENTO_PIECES)
    if variant == "hatetris":
        return Board(tetris.PIECES, 10, 20), HatetrisGenerator(tetris.PIECES)
    return tetris.Board(), tetris.PieceBag(seed)


def build_agent(name, mode, seed, variant, lean, payload, pipeline):
    if name == "heuristic":
        return agents.HeuristicAgent()
    if name == "oracle":
        return agents.OracleAgent()
    if name == "jev":
        return variant_agents.JevAgent(mode=mode, variant=variant, lean=lean, payload=payload, pipeline=pipeline)
    raise ValueError(f"unknown agent: {name}")


def run(name, mode, pieces, seed, variant, quiet=False, lean=False, payload="top12", pipeline=False):
    board, generator = make_game(variant, seed)
    agent = build_agent(name, mode, seed, variant, lean, payload, pipeline)
    samples = []
    restore = None
    if name == "jev":
        samples, restore = variant_agents.install_latency_recorder()
    total_lines = 0
    placed = 0
    topped = False
    try:
        for _ in range(pieces):
            if variant == "hatetris":
                piece = generator.next(board)
                next_pieces = []
                next_piece = None
            else:
                piece = generator.next()
                next_pieces = [generator.peek(k) for k in range(1, agents.NEXT_QUEUE_LEN + 1)]
                next_piece = next_pieces[0]
            legal = board.legal_placements(piece)
            if not legal:
                topped = True
                break
            rot, col = agent.choose(board, piece, next_piece, next_pieces=next_pieces, lines_cleared_total=total_lines)
            cells = board.placement_cells(piece, rot, col)
            cleared = board.drop(piece, rot, col)
            assert cells is not None
            placed += 1
            total_lines += cleared
            if not quiet:
                print(f"--- {name}{'-lean' if lean else ''} piece {placed}/{pieces}: {piece} -> rot={rot} col={col} (lines cleared: {cleared}) ---")
                print(board.render())
    finally:
        if restore:
            restore()
        if hasattr(agent, "close"):
            agent.close()
    p50, p95 = variant_agents.percentiles(samples)
    stats = dict(agent.stats)
    if samples:
        stats["latency_p50_ms"], stats["latency_p95_ms"] = p50, p95
    return {"agent": f"{name}-{mode}{'-lean' if lean else ''}" if name == "jev" else name,
            "pieces_requested": pieces, "pieces_placed": placed, "lines_cleared": total_lines,
            "topped_out": topped, "holes": board.hole_count(), "max_height": board.max_height(),
            "aggregate_height": board.aggregate_height(), "bumpiness": board.bumpiness(), "stats": stats}


def mean(values):
    return statistics.mean(values) if values else None


def print_report(r):
    s = r["stats"]
    req = s.get("requests", 0)
    print(f"=== final report: {r['agent']} ===")
    print(f"placed: {r['pieces_placed']}/{r['pieces_requested']}  lines: {r['lines_cleared']}  topped: {r['topped_out']}  holes: {r['holes']}  max_h: {r['max_height']}")
    print(f"requests: {req}  tokens: {s.get('input_tokens', 0)}  tok/req: {s.get('input_tokens', 0) / req if req else 0:.1f}  cost: {s.get('cost_usd', 0.0):.6f}  mean_conf: {mean(s.get('confidences', []))}  invalid: {s.get('invalid_choices', 0)}")
    if "latency_p50_ms" in s:
        print(f"latency p50: {s['latency_p50_ms']:.1f} ms  p95: {s['latency_p95_ms']:.1f} ms")


def print_table(results):
    print("agent  placed  lines  topped  holes  max_h  requests  tokens  tok/req  cost  mean_conf  invalid  p50_ms  p95_ms")
    for r in results:
        s = r["stats"]; req = s.get("requests", 0)
        print(f"{r['agent']}  {r['pieces_placed']}/{r['pieces_requested']}  {r['lines_cleared']}  {r['topped_out']}  {r['holes']}  {r['max_height']}  {req}  {s.get('input_tokens', 0)}  {s.get('input_tokens', 0)/req if req else 0:.0f}  {s.get('cost_usd', 0):.6f}  {mean(s.get('confidences', [])) if s.get('confidences') else '-'}  {s.get('invalid_choices', 0)}  {s.get('latency_p50_ms', '-')}  {s.get('latency_p95_ms', '-')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["standard", "hatetris", "pento"], required=True)
    ap.add_argument("--agent", choices=["heuristic", "oracle", "jev"])
    ap.add_argument("--mode", choices=["board", "preview", "rich"], default="rich")
    ap.add_argument("--pieces", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--payload", choices=["full", "nostate", "top12", "both"], default="top12")
    ap.add_argument("--pipeline", action="store_true")
    args = ap.parse_args()
    if args.compare:
        if args.variant != "pento":
            ap.error("--compare is specified for pento")
        print_table([run(n, args.mode, args.pieces, args.seed, args.variant, True, args.lean, args.payload, args.pipeline) for n in ("heuristic", "oracle", "jev")])
        return
    if not args.agent:
        ap.error("--agent is required unless --compare is given")
    print_report(run(args.agent, args.mode, args.pieces, args.seed, args.variant, args.quiet, args.lean, args.payload, args.pipeline))


if __name__ == "__main__":
    main()
