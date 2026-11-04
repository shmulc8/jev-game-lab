"""Every question Jev is asked about Snake, and the thresholds that gate its answers.

`policies.py` (and `play.py`'s CLI defaults) import from here and contain only mechanics
(computing the numbers that fill these templates, parsing responses, falling back to code).
Edit this file to change what the model is asked, how criteria read, or where thresholds sit.
"""

# --- thresholds ---

# Confidence floor (0-1) below which a tick falls back to FloodPolicy instead of trusting
# the model's answer. Ask: at this value, how often would a plausible-but-wrong answer
# actually get filtered out, versus how often would a good answer get discarded?
DEFAULT_MIN_CONFIDENCE = 0.0

# Composite mode's default room/food weights, used to combine the two Score judgments into
# one direction choice. Ask: if a reviewer only tunes one of the pair, does the other still
# make sense (e.g. do they still add to 1.0, if that invariant matters)?
DEFAULT_WEIGHT_ROOM = 0.5
DEFAULT_WEIGHT_FOOD = 0.5


# --- choice: direction (raw / features modes) ---

# Ask: read only this sentence with no other context -- is it obvious what "best" means here?
DIRECTION_INSTRUCTIONS = (
    "Choose the best next move for the snake in this game of Snake. "
    "Balance making progress toward the food with staying alive as long as "
    "possible. Only choose one of the given candidate moves."
)


def raw_candidate_criteria(direction, x, y):
    """One candidate's criteria text in 'raw' mode: where the head would land, nothing else."""
    return f"move {direction}: head goes to column {x} row {y}"


def feature_candidate_criteria(direction, x, y, feats):
    """One candidate's criteria text in 'features' mode: raw text plus precomputed geometry."""
    dist = feats["bfs_distance"]
    dist_text = str(dist) if dist is not None else "unreachable"
    return (
        f"move {direction}: head goes to column {x} row {y}"
        f". distance to food after this move: {dist_text}"
        f". reachable open area after this move: {feats['reachable_cells']} cells"
        f". ends adjacent to own body: {feats['adjacent_to_body']}"
        f". free-cell ratio of the board after this move: {feats['free_ratio']:.2f}"
    )


# --- score: room / food (composite mode, one pair per candidate direction) ---

# Each level should describe a situation independently of its numeric position.
ROOM_LEVELS = [
    "very little open room, mostly boxed in with few free cells reachable",
    "limited room, the space is tightening",
    "comfortable room, plenty of reachable free cells",
    "wide open room, a large reachable area with many escape routes",
]
FOOD_LEVELS = [
    "moves away from the food, increasing the distance to it",
    "roughly sideways, the distance to the food stays about the same",
    "moves closer to the food, decreasing the distance to it",
    "the most direct possible move toward the food this turn",
]


def room_instructions(direction, x, y):
    return (
        f"Rate how much open room and free space would remain reachable if the "
        f"snake moves {direction}, ending with its head at column {x} row {y}."
    )


def food_instructions(direction, x, y):
    return (
        f"Rate how directly moving {direction} (head to column {x} row {y}) closes the "
        f"distance to the food shown in state."
    )
