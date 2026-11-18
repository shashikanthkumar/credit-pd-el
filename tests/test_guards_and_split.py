import pytest
from src.utils.config import load_config
from src.features.guards import assert_no_banned, feature_sets, LeakageError


def test_guard_blocks_outcome_features():
    cfg = load_config("configs/config.yaml")
    for bad in ["recoveries", "out_prncp", "funded_amnt", "recoveries_woe"]:
        with pytest.raises(LeakageError):
            assert_no_banned(["dti", bad], cfg)


def test_feature_sets_clean_and_champion_excludes_lc():
    cfg = load_config("configs/config.yaml")
    fs = feature_sets(cfg)
    champ = set(fs["core"][0] + fs["core"][1])
    assert not champ & {"int_rate", "grade", "sub_grade"}
    assert {"int_rate", "grade"} <= set(fs["with_lc"][0] + fs["with_lc"][1])


def test_split_years_are_disjoint_and_ordered():
    s = load_config("configs/config.yaml")["split"]
    assert s["train_years"][1] < s["valid_clean_years"][0] <= s["valid_clean_years"][1] < s["valid_flagged_years"][0]
    assert s["valid_flagged_years"][1] < s["shift_only_years"][0]
    # 36m loans issued by Dec-2015 are fully matured at the 2018-12 snapshot; later ones are not
    assert s["valid_clean_years"][1] <= 2015
