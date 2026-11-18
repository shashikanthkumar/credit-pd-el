"""Feature-set guards: leakage and circularity are enforced in code, not by convention."""


class LeakageError(ValueError):
    pass


def assert_no_banned(features, cfg: dict) -> None:
    banned = set(cfg["features"]["banned"])
    bad = [f for f in features if f.replace("_woe", "") in banned]
    if bad:
        raise LeakageError(f"Banned post-origination/outcome features in PD feature set: {bad}")


def feature_sets(cfg: dict) -> dict:
    f = cfg["features"]
    core_n, core_c = list(f["numeric"]), list(f["categorical"])
    sets = {
        "core": (core_n, core_c),                                    # champion: no LC pricing / grade
        "with_lc": (core_n + f["lc_numeric"], core_c + f["lc_categorical"]),
        "grade_only": ([], list(f["lc_categorical"])),               # benchmark: LC's own grade
    }
    for n, c in sets.values():
        assert_no_banned(n + c, cfg)
    return sets
