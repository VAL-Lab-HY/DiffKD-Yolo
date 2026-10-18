import torch
import torch.nn.functional as F
from torch import nn


class LightConditionEncoder(nn.Module):
    """Predict a full-resolution illumination map ``L`` in the range [0, 1]."""

    def __init__(self, in_channels=3, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.apply(self.init_weights)
        # Map of the latest forward pass, read by LightConditionedLayer and the KD loss.
        self.illumination = None

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def __getstate__(self):
        # Never copy/pickle the cached map (it may hold an autograd graph, e.g. on EMA deepcopy).
        state = self.__dict__.copy()
        state["illumination"] = None
        return state

    def forward(self, images):
        return self.encoder(images)


class LightModulation(nn.Module):
    """FiLM conditioning of a feature map on darkness ``D = 1 - L``: ``x * (1 + gamma) + beta``."""

    def __init__(self, channels, hidden_dim=16):
        super().__init__()
        self.film = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2 * channels, kernel_size=1),
        )
        # Identity at init, so the pretrained detector is unchanged at the start of training.
        nn.init.constant_(self.film[-1].weight, 0)
        nn.init.constant_(self.film[-1].bias, 0)

    def forward(self, x, illumination):
        darkness = 1.0 - F.adaptive_avg_pool2d(illumination, x.shape[-2:]).to(x.dtype)
        gamma, beta = self.film(darkness).chunk(2, dim=1)
        return x * (1 + gamma) + beta


def _copy_layer_attrs(wrapper, layer):
    # Attributes used by Ultralytics' BaseModel._predict_once and model summaries.
    for attr in ("f", "i", "type", "np"):
        if hasattr(layer, attr):
            setattr(wrapper, attr, getattr(layer, attr))


class LCEStem(nn.Module):
    """Wraps the first Ultralytics layer: runs the LCE on the input image and caches ``L``."""

    def __init__(self, layer, lce, downsample=1):
        super().__init__()
        self.layer = layer
        self.lce = lce
        self.downsample = downsample
        _copy_layer_attrs(self, layer)

    def forward(self, x):
        img = F.avg_pool2d(x, self.downsample) if self.downsample > 1 else x
        self.lce.illumination = self.lce(img)
        return self.layer(x)


class LightConditionedLayer(nn.Module):
    """Wraps an Ultralytics layer; its output is modulated by the illumination map cached by ``LCEStem``."""

    def __init__(self, layer, lce, channels, hidden_dim=16):
        super().__init__()
        self.layer = layer
        self.modulation = LightModulation(channels, hidden_dim)
        self.out_channels = channels
        # Plain reference (not a submodule): the LCE is registered once, under LCEStem.
        self.__dict__["_lce"] = lce
        _copy_layer_attrs(self, layer)

    def forward(self, x):
        return self.modulation(self.layer(x), self._lce.illumination)


@torch.no_grad()
def _output_channels(model, indices, imgsz=64):
    channels, handles = {}, []
    for i in indices:
        def hook(m, inp, out, i=i):
            channels[i] = out.shape[1]

        handles.append(model.model[i].register_forward_hook(hook))
    training = model.training
    model.eval()
    model(torch.zeros(1, 3, imgsz, imgsz, device=next(model.parameters()).device))
    model.train(training)
    for h in handles:
        h.remove()
    return channels


def attach_lce(model, layers, hidden_dim=32, modulation_dim=16, downsample=4):
    """Insert an LCE into an Ultralytics DetectionModel and modulate the outputs of ``model.model[i]``, i in ``layers``.

    Module names change from ``model.{i}.*`` to ``model.{i}.layer.*`` for wrapped layers; ``model.{i}`` itself
    outputs the modulated feature.
    """
    layers = [int(i) for i in layers]
    channels = _output_channels(model, layers)
    lce = LightConditionEncoder(hidden_dim=hidden_dim)
    model.model[0] = LCEStem(model.model[0], lce, downsample)
    for i in layers:
        if i == 0:
            raise ValueError("Layer 0 hosts the LCE stem and cannot be modulated")
        model.model[i] = LightConditionedLayer(model.model[i], lce, channels[i], modulation_dim)
    return model


def darkness_loss(illumination, images):
    """Salience-DETR ``loss_darkness``: L1(D, 1 - Y) = L1(L, Y), Y = Rec.601 luminance of the RGB input in [0, 1]."""
    luminance = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
    target = F.adaptive_avg_pool2d(luminance, illumination.shape[-2:])
    return F.l1_loss(illumination.float(), target.float())


def find_lce(model):
    for m in model.modules():
        if isinstance(m, LightConditionEncoder):
            return m
    return None
