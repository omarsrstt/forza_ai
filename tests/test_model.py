import torch
import train_model

def _m():
    return train_model.ConvNextTinyLSTMRegression(num_outputs=3).eval()

def test_forward_is_dense_and_ranged():
    m = _m()
    x = torch.randn(2,4,3,64,64)
    with torch.no_grad(): y = m(x)
    assert y.shape == (2,4,3)
    assert (y[...,0].abs() <= 1.0001).all()      # steer tanh
    assert (y[...,1] >= 0).all() and (y[...,1] <= 1).all()  # throttle sigmoid
    assert (y[...,2] >= 0).all() and (y[...,2] <= 1).all()  # brake sigmoid

def test_output_depends_on_input_fresh_model():
    m = _m()
    a = torch.zeros(1,3,3,64,64); b = torch.randn(1,3,3,64,64)
    with torch.no_grad():
        ya = m(a); yb = m(b)
    assert (ya - yb).abs().max() > 1e-3   # guards against the v1 collapse

def test_forward_from_features_matches_forward():
    m = _m()
    x = torch.randn(1,3,3,64,64)
    with torch.no_grad():
        y1 = m(x)
        y2 = m.forward_from_features(m.encode(x.reshape(-1,3,64,64)).reshape(1,3,-1))
    assert torch.allclose(y1, y2, atol=1e-5)

def test_encode_shape():
    m = _m()
    with torch.no_grad(): f = m.encode(torch.randn(5,3,64,64))
    assert f.shape == (5,768)
