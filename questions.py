"""Every question Jev is asked about Tetris, and the thresholds that gate its answers.

`agents.py` imports from here and contains only mechanics (computing the numbers that
fill these templates, selecting candidates, parsing responses). Edit this file to change
what the model is asked, how criteria read, or where thresholds sit.
"""

# --- thresholds ---

# Cap on how many placement candidates go in one request. Ask: is this still few enough
# for the model to weigh every candidate carefully in a single request, and large enough
# that a placement a human would expect to see isn't silently dropped?
CAP_CANDIDATES = 12


# --- choice: placement (board / preview modes) ---

# Ask: read only this sentence with no other context -- is it obvious what "best" means here?
BASIC_PLACEMENT_INSTRUCTIONS = (
    "Choose the best placement for the current piece to keep the board "
    "low, flat, and free of holes, clearing lines when possible."
)


def board_mode_criteria(rot, col, width, resting_row):
    """One candidate's criteria text in 'board' mode: rotation, columns, and landing row only."""
    return (
        f"rotation {rot}, piece occupies columns {col}-{col + width - 1}, "
        f"lands resting on row {resting_row}"
    )


def preview_mode_criteria(top_rows, ascii_region, lines_cleared):
    """One candidate's criteria text in 'preview' mode: the resulting board, cropped to the bottom top_rows rows."""
    return f"resulting board (bottom {top_rows} rows):\n{ascii_region}\nlines cleared: {lines_cleared}"


# --- choice: placement (rich mode) ---

# Ask: rich mode gives the model numbers instead of a board picture -- does this sentence
# still make sense to someone who has never seen the ASCII board render?
RICH_PLACEMENT_INSTRUCTIONS = (
    "Choose exactly one candidate id for where to place the active piece. Optimize for "
    "clearing lines while avoiding holes, high stacks, and a rough (bumpy) surface. Prefer "
    "a placement that leaves a strong resulting board and future flexibility over an "
    "immediate line clear that creates holes. Use each candidate's one-piece lookahead "
    "score as a tiebreaker between otherwise similar options."
)

# Legend sent once per rich-mode request, explaining its coordinate conventions. Ask: if a
# convention below changes (e.g. which corner is [0,0]), does this legend still match it?
RICH_COORDINATE_SYSTEM = {
    "columns": "integers 0 through 9, left to right",
    "rows": "integers 0 through 19, top to bottom (row 0 is the topmost row)",
    "column_meaning": (
        "the 'column' reported for a placement is the leftmost occupied "
        "cell (minimum column index) of the piece once it has landed"
    ),
    "cells": (
        "the active piece and every candidate list their exact occupied "
        "cells as [column, row] integer pairs"
    ),
}


def rich_candidate_criteria(info):
    """One candidate's criteria text in 'rich' mode, built from its precomputed afterstate info."""
    cells_str = ", ".join(f"[{c},{r}]" for c, r in info["occupied_cells"])
    a = info["afterstate"]
    lookahead = info["lookahead"]
    lookahead_str = (
        f"{lookahead:.2f}"
        if lookahead is not None
        else "N/A: the next piece has no legal placement on this resulting board (it would top out)"
    )
    return (
        f"Rotation {info['rotation']}, leftmost column {info['column']}, landing row "
        f"{info['landing_row']}. Occupied cells after placement: {cells_str}. "
        f"Clears {info['lines_cleared']} line(s) immediately. Resulting board afterstate: "
        f"holes={a['holes']}, max_height={a['max_height']}, aggregate_height={a['aggregate_height']}, "
        f"bumpiness={a['bumpiness']}. One-piece lookahead score (the best afterstate value "
        f"achievable by optimally placing the next piece on this resulting board): {lookahead_str}."
    )


# --- score: topout ---

TOPOUT_INSTRUCTIONS = "Rate how close this board is to topping out (losing)."

# Ask: could a person sketch a board matching each level from its words alone, without
# being told the numeric score that comes with it?
TOPOUT_SCORE_LEVELS = ["plenty of room", "filling up", "about to top out"]
