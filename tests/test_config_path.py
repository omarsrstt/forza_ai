"""A dry run must not touch the real checkpoints.

The 10-epoch augmented run overwrote ``logs/checkpoints/last.ckpt`` with a
one-epoch smoke-test model because ``main()`` hardcodes the output directory.
Recovering it needed a backup and a hash check. A dry run that can clobber the
production checkpoint is a footgun, so the checkpoint directory has to follow the
config file rather than being a constant.
"""

import inspect

import train_model


def test_main_accepts_a_config_path():
    params = inspect.signature(train_model.main).parameters
    assert "config_path" in params
    assert params["config_path"].default == "config.json"


def test_checkpoint_dir_is_derived_from_the_config_not_hardcoded():
    """Two different config files must be able to write to two different places."""
    src = inspect.getsource(train_model.main)
    assert 'config.get("checkpoint_dir"' in src, (
        "checkpoint_dir must be configurable so a dry run cannot overwrite "
        "logs/checkpoints/last.ckpt"
    )


def test_dry_run_config_writes_somewhere_else():
    import json

    cfg = json.load(open("_dryrun_config.json"))
    assert cfg["checkpoint_dir"] != "logs/checkpoints"
    assert cfg["use_wandb"] is False
