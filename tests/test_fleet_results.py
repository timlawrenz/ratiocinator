"""Tests for ResultStore."""

from __future__ import annotations

import pytest

from ratiocinator.fleet.results import ArmResult, ResultStore


@pytest.fixture
def store(tmp_path):
    return ResultStore(tmp_path / "results.json")


@pytest.fixture
def sample_result():
    return ArmResult(
        experiment="test-exp",
        arm_name="baseline",
        description="Baseline config",
        exit_code=0,
        metrics={"loss": 0.5, "it_per_sec": 3.5},
    )


class TestResultStore:
    def test_record_and_retrieve(self, store, sample_result):
        store.record(sample_result)
        result = store.get_arm("test-exp", "baseline")
        assert result is not None
        assert result["exit_code"] == 0
        assert result["metrics"]["loss"] == 0.5

    def test_experiments_list(self, store, sample_result):
        store.record(sample_result)
        assert "test-exp" in store.experiments

    def test_get_experiment(self, store):
        r1 = ArmResult(experiment="exp", arm_name="a", exit_code=0, metrics={"v": 1})
        r2 = ArmResult(experiment="exp", arm_name="b", exit_code=0, metrics={"v": 2})
        store.record(r1)
        store.record(r2)
        results = store.get_experiment("exp")
        assert len(results) == 2
        assert results[0]["arm_name"] == "a"  # sorted
        assert results[1]["arm_name"] == "b"

    def test_missing_experiment(self, store):
        assert store.get_experiment("nonexistent") == []

    def test_missing_arm(self, store):
        assert store.get_arm("nonexistent", "arm") is None


class TestResultStoreMerge:
    def test_success_overrides_failure(self, store):
        fail = ArmResult(experiment="exp", arm_name="arm1", exit_code=1, error="crash")
        store.record(fail)
        assert store.get_arm("exp", "arm1")["exit_code"] == 1

        success = ArmResult(experiment="exp", arm_name="arm1", exit_code=0, metrics={"v": 1})
        store.record(success)
        assert store.get_arm("exp", "arm1")["exit_code"] == 0

    def test_failure_does_not_override_success(self, store):
        success = ArmResult(experiment="exp", arm_name="arm1", exit_code=0, metrics={"v": 1})
        store.record(success)

        fail = ArmResult(experiment="exp", arm_name="arm1", exit_code=1, error="crash")
        store.record(fail)

        assert store.get_arm("exp", "arm1")["exit_code"] == 0  # still success

    def test_first_success_wins(self, store):
        s1 = ArmResult(experiment="exp", arm_name="arm1", exit_code=0, metrics={"v": 1})
        store.record(s1)

        s2 = ArmResult(experiment="exp", arm_name="arm1", exit_code=0, metrics={"v": 2})
        store.record(s2)

        assert store.get_arm("exp", "arm1")["metrics"]["v"] == 1  # first success

    def test_failure_overrides_failure(self, store):
        f1 = ArmResult(experiment="exp", arm_name="arm1", exit_code=1, error="old")
        store.record(f1)

        f2 = ArmResult(experiment="exp", arm_name="arm1", exit_code=1, error="new")
        store.record(f2)

        assert store.get_arm("exp", "arm1")["error"] == "new"  # latest failure


class TestResultStorePersistence:
    def test_persists_to_disk(self, tmp_path):
        path = tmp_path / "results.json"
        store1 = ResultStore(path)
        store1.record(ArmResult(experiment="exp", arm_name="arm", exit_code=0, metrics={"v": 1}))

        # Load from disk in new store
        store2 = ResultStore(path)
        result = store2.get_arm("exp", "arm")
        assert result is not None
        assert result["metrics"]["v"] == 1

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "results.json"
        store = ResultStore(path)
        store.record(ArmResult(experiment="exp", arm_name="arm", exit_code=0))
        assert path.exists()


class TestResultStoreAnalysis:
    def test_get_best_minimize(self, store):
        store.record(ArmResult(experiment="exp", arm_name="a", exit_code=0, metrics={"loss": 0.5}))
        store.record(ArmResult(experiment="exp", arm_name="b", exit_code=0, metrics={"loss": 0.3}))
        store.record(ArmResult(experiment="exp", arm_name="c", exit_code=1, metrics={"loss": 0.1}))

        best = store.get_best("exp", "loss")
        assert best["arm_name"] == "b"  # 0.3 is lowest among successful

    def test_get_best_maximize(self, store):
        store.record(ArmResult(experiment="exp", arm_name="a", exit_code=0, metrics={"speed": 3.5}))
        store.record(ArmResult(experiment="exp", arm_name="b", exit_code=0, metrics={"speed": 4.2}))

        best = store.get_best("exp", "speed", maximize=True)
        assert best["arm_name"] == "b"

    def test_get_best_no_results(self, store):
        assert store.get_best("exp", "loss") is None

    def test_compare_arms(self, store):
        store.record(ArmResult(experiment="exp", arm_name="a", exit_code=0, metrics={"v": 1}))
        store.record(ArmResult(experiment="exp", arm_name="b", exit_code=1, error="crash"))

        comparison = store.compare_arms("exp")
        assert len(comparison) == 2
        assert comparison[0]["success"] is True
        assert comparison[1]["success"] is False

    def test_export_table(self, store):
        store.record(ArmResult(experiment="exp", arm_name="a", exit_code=0, metrics={"loss": 0.5}))
        store.record(ArmResult(experiment="exp", arm_name="b", exit_code=0, metrics={"loss": 0.3}))

        table = store.export_table("exp", ["loss"])
        assert "a" in table
        assert "b" in table
        assert "0.3" in table

    def test_record_many(self, store):
        results = [
            ArmResult(experiment="exp", arm_name="a", exit_code=0, metrics={"v": 1}),
            ArmResult(experiment="exp", arm_name="b", exit_code=0, metrics={"v": 2}),
        ]
        store.record_many(results)
        assert len(store.get_experiment("exp")) == 2
