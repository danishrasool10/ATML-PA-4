"""End-to-end smoke on the tiny model + selection-lock guards."""
import pytest

from pa4.common.results import read_json
from pa4.task1.cli import build_smoke
from pa4.task1.runner import Task1Runner


def test_full_smoke_and_guards(tmp_path):
    cfg, model, tok = build_smoke(str(tmp_path))
    r = Task1Runner(cfg, model, tok)
    with pytest.raises(AssertionError):      # test-time P3/Q3 refuse to run before the dev-only selection is locked
        r.run_p3(0.5)
    with pytest.raises(AssertionError):
        r.run_q3("int4")
    r.run_all()
    out = r.out
    for sel in ("p3", "q3"):
        s = read_json(out / "selection" / f"{sel}_selection.json")
        assert s["uses_test_data"] is False and s["chosen"] in s["scores"]
    n = sum(u.numel for u in r.units)
    for s in (50, 75):
        for c in ("p2", "p3"):
            res = read_json(out / "conditions" / f"{c}_s{s}" / "result.json")
            assert res["compression"]["zeroed_by_mask"] == round(s / 100 * n)       # matched budget
    for bn in ("int4", "int3"):
        b3 = read_json(out / "conditions" / f"q3_{bn}" / "result.json")["storage"]["quantized_model"]["packed_total_bytes"]
        b2 = read_json(out / "conditions" / f"q2_{bn}" / "result.json")["storage"]["quantized_model"]["packed_total_bytes"]
        assert b3 <= b2
    # model restored between conditions: baseline fingerprint equals pristine
    assert read_json(out / "conditions" / "baseline" / "result.json")["model_fingerprint"] == r.snap.fingerprint
    # hellaswag: full on required conditions, 512-subset otherwise (tiny set has 24 examples -> capped)
    assert read_json(out / "conditions" / "p2_s50" / "result.json")["hellaswag_mode"] == "full"
    assert read_json(out / "conditions" / "p2_s25" / "result.json")["hellaswag_mode"] == "subset"
