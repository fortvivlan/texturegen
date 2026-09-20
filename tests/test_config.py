from texturegen.config import TrainingConfig


def test_config_round_trip(tmp_path):
    original = TrainingConfig(dataset_dir="prepared", max_train_steps=12, rank=8)
    path = original.save(tmp_path / "config.json")
    assert TrainingConfig.load(path) == original
