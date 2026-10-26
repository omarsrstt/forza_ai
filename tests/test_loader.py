import torch, pytest
import train_model

def _save_ckpt(tmp_path, model):
    class M(torch.nn.Module):
        def __init__(self, model): super().__init__(); self.model = model
    p = tmp_path / "c.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
                "hyper_parameters": {}}, str(p))
    return str(p)

def test_loads_lightning_prefixed_checkpoint(tmp_path):
    src = train_model.ConvNextTinyLSTMRegression(3)
    p = _save_ckpt(tmp_path, src)
    m, hp = train_model.load_forza_checkpoint(p, 3)
    for k, v in src.state_dict().items():
        assert torch.equal(m.state_dict()[k], v)

def test_rejects_mismatched_checkpoint(tmp_path):
    src = train_model.ConvNextTinyLSTMRegression(3)
    p = _save_ckpt(tmp_path, src)
    with pytest.raises(RuntimeError):
        train_model.load_forza_checkpoint(p, num_outputs=7)   # old 7-output shape
