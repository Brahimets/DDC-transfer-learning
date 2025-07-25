import torch
import mmd
from model import Alexnet_finetune, DCCNet

def test_alexnet_forward():
    net = Alexnet_finetune(num_classes=5)
    x = torch.randn(2, 3, 227, 227)
    y = net(x)
    assert y.shape == (2, 5)

def test_ddcnet_forward(monkeypatch):
    def fake_mmd(source, target):
        return torch.tensor(0.0)
    monkeypatch.setattr(mmd, "mmd_linear", fake_mmd, raising=False)
    net = DCCNet(num_classes=5)
    s = torch.randn(2, 3, 227, 227)
    t = torch.randn(2, 3, 227, 227)
    y, loss = net(s, t)
    assert y.shape == (2, 5)
    assert torch.is_tensor(loss)
