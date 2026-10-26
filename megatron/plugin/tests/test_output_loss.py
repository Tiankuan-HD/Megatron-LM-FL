from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.plugin.Ascend.fusions import linear_cross_entropy as ce
from megatron.plugin.decorators import get_override_method


class Output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(13, 7, dtype=torch.float64))

    def forward(self, hidden, weight=None, runtime_gather_output=None):
        return (torch.nn.functional.linear(hidden, self.weight if weight is None else weight), None)


@pytest.mark.parametrize("external_weight", [False, True])
@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("bound_callback", [False, True])
def test_cpu_reference_fallback_hooks_and_gradients(
    monkeypatch, external_weight, scale, bound_callback
):
    monkeypatch.setenv("MG_FL_PREFER", "npu")
    assert get_override_method("LanguageModule.compute_output_loss") is ce.compute_output_loss
    torch.manual_seed(223)
    layer = Output()
    hidden = torch.randn(5, 2, 7, dtype=torch.float64, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    weight = layer.weight.detach().clone().requires_grad_() if external_weight else None
    reference_weight = (
        (layer.weight if weight is None else weight).detach().clone().requires_grad_()
    )
    labels = torch.arange(10).reshape(2, 5)
    hooks = []
    layer.register_forward_pre_hook(lambda *unused: hooks.append("pre"))
    layer.register_forward_hook(lambda *unused: hooks.append("post"))

    def loss_fn(target, logits):
        return (
            torch.nn.functional.cross_entropy(
                logits.reshape(-1, 13), target.T.reshape(-1), reduction="none"
            )
            .reshape(5, 2)
            .T.contiguous()
        )

    if bound_callback:

        class Model:
            config = SimpleNamespace()
            training = True
            compute_language_model_loss = staticmethod(loss_fn)

            def loss(self, target, logits):
                return self.compute_language_model_loss(target, logits)

        callback = Model().loss
    else:
        callback = loss_fn
    scale_fn = lambda logits: logits * scale
    model = SimpleNamespace(
        compute_language_model_loss=callback, config=SimpleNamespace(), training=True
    )
    actual = LanguageModule.compute_output_loss(
        model, hidden, layer, weight, labels, scale_logits_fn=scale_fn
    )
    expected = loss_fn(
        labels, torch.nn.functional.linear(reference_hidden, reference_weight) * scale
    )
    upstream = torch.linspace(0, 1, 10).reshape(2, 5)
    actual.backward(upstream)
    expected.backward(upstream)
    assert hooks == ["pre", "post"]
    for value, reference in (
        (actual, expected),
        (hidden.grad, reference_hidden.grad),
        ((layer.weight if weight is None else weight).grad, reference_weight.grad),
    ):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_callback_exception_is_not_silently_swallowed():
    def fail(*unused):
        raise RuntimeError("caller loss failed")

    with pytest.raises(RuntimeError, match="caller loss failed"):
        LanguageModule.compute_output_loss(
            SimpleNamespace(
                compute_language_model_loss=fail, config=SimpleNamespace(), training=True
            ),
            torch.zeros(1, 1, 7, dtype=torch.float64),
            Output(),
            None,
            torch.zeros(1, 1, dtype=torch.long),
            fail,
        )
