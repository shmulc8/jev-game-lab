"""Only question text that differs from the root experiment."""

import questions as root_questions

CAP_CANDIDATES = root_questions.CAP_CANDIDATES
TOPOUT_INSTRUCTIONS = root_questions.TOPOUT_INSTRUCTIONS
TOPOUT_SCORE_LEVELS = root_questions.TOPOUT_SCORE_LEVELS

RICH_PLACEMENT_INSTRUCTIONS = root_questions.RICH_PLACEMENT_INSTRUCTIONS


def hatetris_instructions():
    return (
        "Choose exactly one candidate id for the current piece. Optimize for clearing "
        "lines while avoiding holes, high stacks, and a rough surface. The next piece "
        "is unknown: an adversarial generator will choose the next kind after this move, "
        "so do not use a lookahead term."
    )


def coordinate_system(columns, rows, piece_set, adversarial=False):
    system = {
        "columns": columns,
        "rows": rows,
        "column_meaning": "the reported column is the piece's leftmost occupied cell after landing",
        "cells": "active and candidate pieces list exact occupied cells as [column, row] integer pairs",
        "piece_set": piece_set,
    }
    if adversarial:
        system["generator"] = "next piece is unknown and selected adversarially from the piece table"
    return system


def rich_candidate_criteria(info, lean=False, include_lookahead=True):
    cells = ", ".join(f"[{c},{r}]" for c, r in info["occupied_cells"])
    a = info["afterstate"]
    if include_lookahead:
        look = f" look {info['lookahead']:.2f}" if info["lookahead"] is not None else " look TOP_OUT"
    else:
        look = ""
    if lean:
        return (f"rot {info['rotation']} col {info['column']} row {info['landing_row']} cells [{cells}] "
                f"clears {info['lines_cleared']} holes {a['holes']} maxh {a['max_height']} "
                f"aggh {a['aggregate_height']} bump {a['bumpiness']}{look}")
    if not include_lookahead:
        return (f"Rotation {info['rotation']}, leftmost column {info['column']}, landing row "
                f"{info['landing_row']}. Occupied cells after placement: {cells}. "
                f"Clears {info['lines_cleared']} line(s) immediately. Resulting board afterstate: "
                f"holes={a['holes']}, max_height={a['max_height']}, aggregate_height={a['aggregate_height']}, "
                f"bumpiness={a['bumpiness']}.")
    return root_questions.rich_candidate_criteria(info)
