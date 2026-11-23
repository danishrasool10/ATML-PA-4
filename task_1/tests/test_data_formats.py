"""Loaders against the real atml-pa4-datasets layouts (synthetic content, real field names, gzip JSONL)."""
import gzip
import json

from pa4t1.manifests import load_hellaswag, load_ppl_windows, load_token_sequences


def _w(path, rows):
    with gzip.open(path, "wt", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_locked_layouts(tmp_path):
    cal = [{"source_row": i, "token_offset": 0, "token_ids": list(range(3, 3 + 16)), "raw_text_sha256": "x"} for i in range(5)]
    _w(tmp_path / "cal.jsonl.gz", cal)
    s = load_token_sequences(tmp_path / "cal.jsonl.gz", None, name="calibration", expected_n=5, seq_len=16)
    assert len(s) == 5 and s.record_ids == [0, 1, 2, 3, 4]
    wins = [{"source": "wikitext2_test", "source_row": 3, "token_offset": 0, "token_ids": list(range(10)), "score_target_offsets": [1, 10], "scored_tokens": 9},
            {"source": "wikitext2_test", "source_row": 3, "token_offset": 512, "token_ids": list(range(4)), "score_target_offsets": [1, 4], "scored_tokens": 3}]
    _w(tmp_path / "w.jsonl.gz", wins)
    w = load_ppl_windows(tmp_path / "w.jsonl.gz", None, name="wiki", window_len=512, expected_scored=12)
    assert w.n_scored_tokens == 12 and w.doc_ids == ["3:0", "3:512"]
    hs = [{"order": 1, "source_row_index": 9, "example": {"ind": 9, "activity_label": "A", "ctx_a": "a", "ctx_b": "b", "endings": ["1", "2", "3", "4"], "label": "2"}},
          {"order": 0, "source_row_index": 4, "example": {"ind": 4, "activity_label": "A", "ctx_a": "a", "ctx_b": "b", "endings": ["1", "2", "3", "4"], "label": "0"}}]
    _w(tmp_path / "h.jsonl.gz", hs)
    h = load_hellaswag(tmp_path / "h.jsonl.gz", expected_n=2)
    assert h.ids == [4, 9] and h.examples[0]["label"] == "0"       # manifest order restored from "order"
