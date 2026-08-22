"""Decision-grouped taxonomy for unstructured-traffic detection.

WHY THIS EXISTS
----------------
The detector was confusing truck / bus / tanker / tractor / LCV with each
other — five classes competing for the same visual evidence. But the collision
and overtaking layers treat all five identically: large, slow to stop, blocks
the forward view. The fine-grained distinction costs accuracy and buys the
system nothing.

So classes here are grouped by **what the decision layer must do differently
for each**, not by how similar they look. Collapsing truck/bus/tanker into one
class does not lose information the pipeline uses; it removes a confusion mode
by construction.

RELATION TO THE DATASET AUTHORS' OWN HIERARCHY
-----------------------------------------------
IDD (Varma et al., WACV 2019) ships a 4-level label hierarchy — 7 labels at
level 1, 16 at level 2, 26 at level 3, 30 at level 4 — where each level is the
union of the level below and higher levels are deliberately less ambiguous.
That hierarchy is organised by visual and semantic similarity.

`IDD_LEVEL3_GROUPS` below reproduces a comparable semantic grouping so the two
can be compared experimentally. `DECISION_GROUPS` is ours. Reporting both is
the point: it lets us measure whether grouping by decision consequence beats
grouping by appearance, rather than asserting it.

USAGE
-----
    from models.taxonomy import DecisionTaxonomy
    tx = DecisionTaxonomy()
    tx.map_name("autorickshaw")        -> "THREE_WHEELER"
    tx.map_id_from_names(source_names) -> {src_id: decision_id}
    tx.write_yaml(Path("data/x/decision.yaml"), root)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# The six decision groups, in a fixed order that defines their class ids.
# ---------------------------------------------------------------------------

DECISION_CLASSES = [
    "VULNERABLE",        # 0
    "TWO_WHEELER",       # 1
    "THREE_WHEELER",     # 2
    "LIGHT_VEHICLE",     # 3
    "HEAVY_VEHICLE",     # 4
    "STATIC_OBSTACLE",   # 5
]

# What the pipeline actually does differently per group. Kept in code rather
# than prose so the downstream modules can read it instead of hardcoding
# per-class constants in three separate files.
#
#   ttc_margin_scale   multiplies the collision warning/brake thresholds
#   lateral_clearance_m minimum side gap required when passing
#   blocks_view        object is tall/wide enough to hide the road ahead
#   can_move           False for genuinely static obstacles (no closing-speed TTC)
GROUP_BEHAVIOUR = {
    "VULNERABLE":      dict(ttc_margin_scale=1.8, lateral_clearance_m=1.5,
                            blocks_view=False, can_move=True),
    "TWO_WHEELER":     dict(ttc_margin_scale=1.4, lateral_clearance_m=1.2,
                            blocks_view=False, can_move=True),
    "THREE_WHEELER":   dict(ttc_margin_scale=1.2, lateral_clearance_m=1.0,
                            blocks_view=False, can_move=True),
    "LIGHT_VEHICLE":   dict(ttc_margin_scale=1.0, lateral_clearance_m=0.8,
                            blocks_view=False, can_move=True),
    "HEAVY_VEHICLE":   dict(ttc_margin_scale=1.3, lateral_clearance_m=1.0,
                            blocks_view=True,  can_move=True),
    "STATIC_OBSTACLE": dict(ttc_margin_scale=1.0, lateral_clearance_m=0.8,
                            blocks_view=False, can_move=False),
}

# Source class name -> decision group. Names are normalised (lowercased,
# separators collapsed) before lookup, so "Auto-Rickshaw", "auto rickshaw" and
# "autorickshaw" all resolve. Covers the vocabularies of IDD, DriveIndia,
# UVH-26, KITTI and BDD100K.
NAME_TO_GROUP = {
    # --- VULNERABLE: unprotected, unpredictable, small ---
    "person": "VULNERABLE", "pedestrian": "VULNERABLE",
    "person sitting": "VULNERABLE", "rider": "VULNERABLE",
    "other person": "VULNERABLE",
    "animal": "VULNERABLE", "cattle": "VULNERABLE", "cow": "VULNERABLE",
    "dog": "VULNERABLE", "buffalo": "VULNERABLE",

    # --- TWO_WHEELER: lane-splits, sudden lateral movement ---
    "motorcycle": "TWO_WHEELER", "motorbike": "TWO_WHEELER",
    "two wheeler": "TWO_WHEELER", "2 wheeler": "TWO_WHEELER",
    "scooter": "TWO_WHEELER",
    "bicycle": "TWO_WHEELER", "cycle": "TWO_WHEELER",
    "cyclist": "TWO_WHEELER",

    # --- THREE_WHEELER: slow, stops without warning, India-specific ---
    "autorickshaw": "THREE_WHEELER", "auto rickshaw": "THREE_WHEELER",
    "three wheeler": "THREE_WHEELER", "3 wheeler": "THREE_WHEELER",
    "auto": "THREE_WHEELER", "tuktuk": "THREE_WHEELER",

    # --- LIGHT_VEHICLE: standard road behaviour ---
    "car": "LIGHT_VEHICLE", "van": "LIGHT_VEHICLE", "jeep": "LIGHT_VEHICLE",
    "suv": "LIGHT_VEHICLE", "muv": "LIGHT_VEHICLE",
    "hatchback": "LIGHT_VEHICLE", "sedan": "LIGHT_VEHICLE",
    "taxi": "LIGHT_VEHICLE", "ambulance": "LIGHT_VEHICLE",
    "police vehicle": "LIGHT_VEHICLE", "caravan": "LIGHT_VEHICLE",

    # --- HEAVY_VEHICLE: blocks view, long stopping distance ---
    "truck": "HEAVY_VEHICLE", "bus": "HEAVY_VEHICLE",
    "mini bus": "HEAVY_VEHICLE", "minibus": "HEAVY_VEHICLE",
    "lcv": "HEAVY_VEHICLE", "light commercial vehicle": "HEAVY_VEHICLE",
    "commercial vehicle": "HEAVY_VEHICLE",
    "tempo traveller": "HEAVY_VEHICLE", "tempo": "HEAVY_VEHICLE",
    "trailer": "HEAVY_VEHICLE", "tanker": "HEAVY_VEHICLE",
    "water tanker": "HEAVY_VEHICLE", "tractor": "HEAVY_VEHICLE",
    "excavator": "HEAVY_VEHICLE", "construction vehicle": "HEAVY_VEHICLE",
    "train": "HEAVY_VEHICLE", "tram": "HEAVY_VEHICLE",

    # --- STATIC_OBSTACLE: does not move ---
    "cart": "STATIC_OBSTACLE", "pushcart": "STATIC_OBSTACLE",
    "hand cart": "STATIC_OBSTACLE", "street cart": "STATIC_OBSTACLE",
    "barrier": "STATIC_OBSTACLE", "traffic cone": "STATIC_OBSTACLE",
    "cone": "STATIC_OBSTACLE", "debris": "STATIC_OBSTACLE",
    "obstacle": "STATIC_OBSTACLE",
}

# A comparable grouping organised by APPEARANCE rather than decision
# consequence, standing in for IDD's own level-3 style hierarchy. Used as the
# control condition in the granularity experiment — without it we could not
# tell whether any gain comes from grouping per se or from grouping *this way*.
IDD_LEVEL3_GROUPS = {
    "person": "person", "pedestrian": "person", "person sitting": "person",
    "rider": "rider",
    "animal": "animal", "cattle": "animal", "cow": "animal", "dog": "animal",
    "motorcycle": "motorcycle", "motorbike": "motorcycle",
    "two wheeler": "motorcycle", "scooter": "motorcycle",
    "bicycle": "bicycle", "cycle": "bicycle", "cyclist": "bicycle",
    "autorickshaw": "autorickshaw", "auto rickshaw": "autorickshaw",
    "three wheeler": "autorickshaw", "auto": "autorickshaw",
    "car": "car", "jeep": "car", "suv": "car", "hatchback": "car",
    "sedan": "car", "taxi": "car",
    "van": "vehicle fallback", "muv": "vehicle fallback",
    "tempo traveller": "vehicle fallback", "lcv": "vehicle fallback",
    "truck": "truck", "tanker": "truck", "trailer": "truck",
    "bus": "bus", "mini bus": "bus", "minibus": "bus",
    "tractor": "vehicle fallback", "excavator": "vehicle fallback",
    "construction vehicle": "vehicle fallback",
    "commercial vehicle": "vehicle fallback",
    "light commercial vehicle": "vehicle fallback",
    "tempo": "vehicle fallback", "caravan": "vehicle fallback",
    "water tanker": "truck",
    "ambulance": "car", "police vehicle": "car",
    "other person": "person", "buffalo": "animal",
    "2 wheeler": "motorcycle", "3 wheeler": "autorickshaw",
    "tuktuk": "autorickshaw",
    "cart": "cart", "pushcart": "cart", "hand cart": "cart",
    "street cart": "cart",
    "barrier": "obstacle", "traffic cone": "obstacle", "cone": "obstacle",
    "debris": "obstacle", "obstacle": "obstacle",
    "train": "train", "tram": "train",
}

# The two groupings MUST cover exactly the same source vocabulary. If one
# recognises a class the other does not, that class's boxes are dropped from
# one dataset and kept in the other -- the three granularities would then be
# trained on different data and the comparison between them would measure the
# difference in data, not the difference in taxonomy. Checked at import so a
# later edit to one dict cannot silently invalidate the experiment.
_missing_l3 = set(NAME_TO_GROUP) - set(IDD_LEVEL3_GROUPS)
_missing_dec = set(IDD_LEVEL3_GROUPS) - set(NAME_TO_GROUP)
if _missing_l3 or _missing_dec:
    raise AssertionError(
        "taxonomy vocabularies out of sync - the granularity comparison would "
        "be invalid. "
        f"missing from IDD_LEVEL3_GROUPS: {sorted(_missing_l3)}; "
        f"missing from NAME_TO_GROUP: {sorted(_missing_dec)}")


def normalise(name: str) -> str:
    """Lowercase, collapse separators, strip. So 'Auto-Rickshaw' == 'auto rickshaw'."""
    s = re.sub(r"[_\-/]+", " ", str(name).strip().lower())
    return re.sub(r"\s+", " ", s)


# ---------------------------------------------------------------------------
# Safety-weighted misclassification cost
# ---------------------------------------------------------------------------

# Ordering by caution: an object higher in this list demands more caution from
# the pipeline. Predicting something LOWER than the truth means the system
# treated a hazard as something safer — that is the error we penalise hardest.
CAUTION_RANK = {
    "VULNERABLE": 5,
    "TWO_WHEELER": 4,
    "THREE_WHEELER": 3,
    "HEAVY_VEHICLE": 3,
    "LIGHT_VEHICLE": 2,
    "STATIC_OBSTACLE": 1,
}

# A missed object is modelled as a prediction of a virtual class that demands
# NO caution at all -- rank 0, below every real group. This is what anchors the
# scale: the most expensive single event the metric can score is a vulnerable
# road user going entirely undetected, and that event costs exactly 1.0.
#
# Anchoring here rather than at "pedestrian called a barrier" is not cosmetic.
# With the ceiling on a misclassification, the arithmetic made calling a
# two-wheeler a barrier (0.825) cost MORE than not seeing it at all (0.80),
# which inverts the actual hazard: a wrong label still puts an obstacle in the
# world model, a miss puts nothing there. Treating a miss as rank 0 makes it
# dominate every misclassification of the same object by construction.
RANK_UNDETECTED = 0

COST_SAME_GROUP = 0.0     # bus predicted as truck: benign, same treatment
COST_MORE_CAUTIOUS = 0.1  # over-caution is a nuisance, not a hazard
COST_SIMILAR = 0.3        # different group, comparable consequence
COST_LESS_CAUTIOUS = 1.0  # the dangerous limit: a vulnerable user, undetected


def _less_cautious_cost(t_rank: int, p_rank: int) -> float:
    """Cost when the prediction demands less caution than the truth.

    Grows with the distance fallen down the caution ordering, so "vulnerable
    called static" costs more than "vulnerable called two-wheeler", and a miss
    (rank 0) costs more than either.
    """
    span = max(CAUTION_RANK.values()) - RANK_UNDETECTED
    return COST_SIMILAR + (COST_LESS_CAUTIOUS - COST_SIMILAR) * ((t_rank - p_rank) / span)


def misclassification_cost(true_group: str, pred_group: str) -> float:
    """Asymmetric cost of predicting `pred_group` when the truth is `true_group`.

    Deliberately not symmetric. A symmetric confusion matrix scores
    "pedestrian called a barrier" and "barrier called a pedestrian" the same,
    but only the first removes a protection the system would otherwise apply.

    This extends to detection the principle IDD-AW (WACV 2024) established for
    segmentation with its Safe mIoU: use the hierarchy to distinguish dangerous
    mispredictions from harmless ones.
    """
    if true_group == pred_group:
        return COST_SAME_GROUP
    t = CAUTION_RANK.get(true_group)
    p = CAUTION_RANK.get(pred_group)
    if t is None or p is None:
        return COST_SIMILAR
    if p > t:
        return COST_MORE_CAUTIOUS
    if p == t:
        return COST_SIMILAR
    return _less_cautious_cost(t, p)


def miss_cost(true_group: str) -> float:
    """Cost of failing to detect an object of `true_group` at all.

    The limiting case of the less-cautious error: the object is absent from the
    world model, so the pipeline applies no caution whatever. Scored as a
    prediction of the virtual rank-0 class, which makes it strictly worse than
    any misclassification of the same object while still scaling with how much
    caution that object demanded -- missing a pedestrian is not the same event
    as missing a traffic cone.
    """
    t = CAUTION_RANK.get(true_group)
    if t is None:
        return COST_LESS_CAUTIOUS
    return _less_cautious_cost(t, RANK_UNDETECTED)


def false_positive_cost(pred_group: str) -> float:
    """Cost of a detection with no corresponding object -- a phantom.

    Charged as an over-caution error rather than a hazard: the system brakes
    or refuses an overtake for nothing. Small, but deliberately not zero,
    because phantoms erode trust in the alerts and a driver who learns to
    ignore warnings has lost the protection entirely. Scaled by caution rank
    for the same reason as `miss_cost` -- a phantom pedestrian triggers a
    harder intervention than a phantom cone.
    """
    p = CAUTION_RANK.get(pred_group)
    if p is None:
        return COST_MORE_CAUTIOUS
    return COST_MORE_CAUTIOUS * (p / max(CAUTION_RANK.values()))


# ---------------------------------------------------------------------------

@dataclass
class DecisionTaxonomy:
    """Maps any source dataset's class names onto the decision groups."""

    classes: list = field(default_factory=lambda: list(DECISION_CLASSES))
    unmapped: dict = field(default_factory=dict, repr=False)

    def index(self, group: str) -> int:
        return self.classes.index(group)

    def map_name(self, name: str) -> Optional[str]:
        """Source class name -> decision group, or None if unrecognised.

        Returns None rather than guessing: silently bucketing an unknown class
        would corrupt the training labels with no way to notice.
        """
        g = NAME_TO_GROUP.get(normalise(name))
        if g is None:
            self.unmapped[normalise(name)] = self.unmapped.get(normalise(name), 0) + 1
        return g

    def map_id_from_names(self, source_names) -> dict:
        """Build {source_class_id: decision_class_id} from a source id->name map.

        Accepts either a dict {id: name} or a list of names in id order — both
        are common in YOLO dataset configs.
        """
        if isinstance(source_names, dict):
            items = [(int(k), v) for k, v in source_names.items()]
        else:
            items = list(enumerate(source_names))
        out = {}
        for sid, name in items:
            g = self.map_name(name)
            if g is not None:
                out[sid] = self.index(g)
        return out

    def behaviour(self, group_or_id) -> dict:
        """Decision parameters for a group, by name or class id."""
        g = self.classes[group_or_id] if isinstance(group_or_id, int) else group_or_id
        return GROUP_BEHAVIOUR[g]

    def write_yaml(self, path: Path, dataset_root: Path,
                   train: str = "train/images", val: str = "val/images") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = "\n".join(f"  {i}: {c}" for i, c in enumerate(self.classes))
        path.write_text(
            "# Decision-grouped taxonomy - classes defined by what the decision\n"
            "# layer must do differently, not by visual similarity.\n"
            "# See models/taxonomy.py and SIH_2026_PLAN.md Part 2.\n"
            f"path: {Path(dataset_root).resolve()}\n"
            f"train: {train}\nval: {val}\n"
            f"names:\n{names}\n", encoding="utf-8")
        return path

    def report_unmapped(self) -> None:
        if not self.unmapped:
            return
        print("[taxonomy] class names not in NAME_TO_GROUP (skipped, not guessed):")
        for k, v in sorted(self.unmapped.items(), key=lambda kv: -kv[1]):
            print(f"    {k!r}: {v}")


def cost_matrix() -> list:
    """Full |C| x |C| cost matrix, rows = truth, columns = prediction."""
    return [[misclassification_cost(t, p) for p in DECISION_CLASSES]
            for t in DECISION_CLASSES]


if __name__ == "__main__":
    tx = DecisionTaxonomy()
    print("Decision groups:")
    for i, c in enumerate(tx.classes):
        b = GROUP_BEHAVIOUR[c]
        print(f"  {i}  {c:<16} ttc x{b['ttc_margin_scale']:.1f}  "
              f"lateral {b['lateral_clearance_m']:.1f} m  "
              f"blocks_view={b['blocks_view']}  can_move={b['can_move']}")

    print("\nSafety-weighted misclassification cost (row = truth, col = predicted):")
    w = 17
    print(" " * w + "".join(f"{c[:9]:>11}" for c in DECISION_CLASSES))
    for t, row in zip(DECISION_CLASSES, cost_matrix()):
        print(f"{t:<{w}}" + "".join(f"{v:>11.2f}" for v in row))

    print("\nSanity: a few source names ->")
    for n in ["Auto-Rickshaw", "water tanker", "Cow", "Sedan", "pushcart", "flying saucer"]:
        print(f"  {n:<16} -> {tx.map_name(n)}")
    tx.report_unmapped()
