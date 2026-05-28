"""Tests for the curriculum warm-start checkpoint resolution in training.train.

Pure path logic — no rlgym_sim / rlgym_ppo / torch needed (those are imported
lazily inside `train()`), so this runs under `test-fast`.
"""
from __future__ import annotations

# rlbot.training.__init__ rebinds the name `train` to the train() *function*, which
# shadows the submodule for normal `import ... as` access. Fetch the real module
# object from sys.modules, where it always lives under its full dotted name.
import sys

import rlbot.training.train  # noqa: F401  (ensures the submodule is imported)
from rlbot.utils.config import Config

train_mod = sys.modules["rlbot.training.train"]


def test_numbered_checkpoints_filters_and_lists(tmp_path):
    run = tmp_path / "exp_004_chase"
    run.mkdir()
    (run / "1000000").mkdir()
    (run / "2000000").mkdir()
    (run / "latest").mkdir()        # non-numeric → ignored
    (run / "run_metadata.json").write_text("{}", encoding="utf-8")  # file → ignored

    names = sorted(d.name for d in train_mod._numbered_checkpoints(run))
    assert names == ["1000000", "2000000"]


def test_numbered_checkpoints_missing_dir(tmp_path):
    assert train_mod._numbered_checkpoints(tmp_path / "nope") == []


def test_resolve_init_checkpoint_picks_numeric_max(tmp_path, monkeypatch):
    monkeypatch.setattr(train_mod, "CHECKPOINT_ROOT", tmp_path)
    base = tmp_path / "exp_004_chase"
    base.mkdir()
    for ts in ("5000000", "100000000", "50000000"):
        (base / ts).mkdir()

    resolved = train_mod._resolve_init_checkpoint("exp_004_chase")
    assert resolved is not None
    # numeric max, not lexicographic ("50000000" > "100000000" as strings)
    assert resolved.name == "100000000"


def test_resolve_init_checkpoint_none_when_no_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(train_mod, "CHECKPOINT_ROOT", tmp_path)
    (tmp_path / "exp_empty").mkdir()
    assert train_mod._resolve_init_checkpoint("exp_empty") is None
    assert train_mod._resolve_init_checkpoint("never_trained") is None


def test_run_metadata_written_outside_checkpoint_folder(tmp_path, monkeypatch):
    # Regression: rlgym_ppo int()-parses every entry in checkpoints_save_folder, so
    # run_metadata.json must NOT live there (it crashes saves with add_unix_timestamp:false).
    monkeypatch.setattr(train_mod, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(train_mod, "CHECKPOINT_ROOT", tmp_path / "checkpoints")
    cfg = Config.from_dict({"experiment_name": "exp_test", "seed": 1})

    meta_path = train_mod._snapshot_run_metadata(cfg)

    assert meta_path.exists()
    assert meta_path.parent == tmp_path / "logs"
    # Nothing must be created inside the (future) checkpoint folder.
    assert not (tmp_path / "checkpoints" / "exp_test").exists()
