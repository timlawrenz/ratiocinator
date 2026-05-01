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


class TestConfigHashAndDiff:
    def test_record_persists_config_hash(self, store):
        r = ArmResult(
            experiment="exp",
            arm_name="a",
            exit_code=0,
            metrics={"v": 1.0},
            config_hash="abc123",
        )
        store.record(r)
        got = store.get_arm("exp", "a")
        assert got["config_hash"] == "abc123"

    def test_diff_identical_config_diff_metrics(self, store):
        store.record(ArmResult(
            experiment="exp", arm_name="a", exit_code=0,
            metrics={"loss": 0.5}, config_hash="hash-shared",
        ))
        store.record(ArmResult(
            experiment="exp", arm_name="b", exit_code=0,
            metrics={"loss": 0.7}, config_hash="hash-shared",
        ))
        diffs = store.diff_results("exp")
        flagged = diffs["identical_config_diff_metrics"]
        assert len(flagged) == 1
        assert {flagged[0]["arm_a"], flagged[0]["arm_b"]} == {"a", "b"}
        assert "loss" in flagged[0]["diffs"]

    def test_diff_different_config_same_metrics(self, store):
        store.record(ArmResult(
            experiment="exp", arm_name="a", exit_code=0,
            metrics={"loss": 0.5}, config_hash="hash-A",
        ))
        store.record(ArmResult(
            experiment="exp", arm_name="b", exit_code=0,
            metrics={"loss": 0.5}, config_hash="hash-B",
        ))
        diffs = store.diff_results("exp")
        flagged = diffs["different_config_same_metrics"]
        assert len(flagged) == 1
        assert {flagged[0]["arm_a"], flagged[0]["arm_b"]} == {"a", "b"}

    def test_diff_ignores_failures(self, store):
        store.record(ArmResult(
            experiment="exp", arm_name="a", exit_code=0,
            metrics={"loss": 0.5}, config_hash="h",
        ))
        store.record(ArmResult(
            experiment="exp", arm_name="b", exit_code=1,
            metrics={"loss": 0.7}, config_hash="h",
        ))
        diffs = store.diff_results("exp")
        assert diffs["identical_config_diff_metrics"] == []
        assert diffs["different_config_same_metrics"] == []

    def test_diff_near_zero_atol_avoids_false_positive(self, store):
        # Both metrics are near zero — without atol the relative
        # difference would explode. With default atol=1e-9 they should
        # be treated as agreeing.
        store.record(ArmResult(
            experiment="exp", arm_name="a", exit_code=0,
            metrics={"loss": 0.0}, config_hash="hash-shared",
        ))
        store.record(ArmResult(
            experiment="exp", arm_name="b", exit_code=0,
            metrics={"loss": 1e-12}, config_hash="hash-shared",
        ))
        diffs = store.diff_results("exp")
        assert diffs["identical_config_diff_metrics"] == []

    def test_diff_warns_on_missing_config_hash(self, store, caplog):
        import logging
        store.record(ArmResult(
            experiment="exp", arm_name="a", exit_code=0,
            metrics={"loss": 0.5}, config_hash="",
        ))
        store.record(ArmResult(
            experiment="exp", arm_name="b", exit_code=0,
            metrics={"loss": 0.5}, config_hash="hash-B",
        ))
        with caplog.at_level(logging.WARNING):
            store.diff_results("exp")
        assert any("missing config_hash" in r.message for r in caplog.records)


class TestCostFields:
    """Tests for actual-cost tracking fields on ArmResult."""

    def test_cost_fields_default_to_zero(self):
        r = ArmResult(experiment="exp", arm_name="arm")
        assert r.estimated_cost == 0.0
        assert r.actual_cost is None
        assert r.boot_time_s == 0.0
        assert r.instance_dph == 0.0

    def test_cost_fields_round_trip_via_store(self, store):
        result = ArmResult(
            experiment="exp",
            arm_name="arm",
            exit_code=0,
            instance_dph=0.52,
            boot_time_s=180.0,
            estimated_cost=0.12,
            actual_cost=0.15,
        )
        store.record(result)
        loaded = store.get_arm("exp", "arm")
        assert loaded["instance_dph"] == 0.52
        assert loaded["boot_time_s"] == 180.0
        assert loaded["estimated_cost"] == 0.12
        assert loaded["actual_cost"] == 0.15
