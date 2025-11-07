import torch
import torch.nn as nn


class ScaledTanh(nn.Module):
    def __init__(self, scale_factor=1.0, input_scale_factor=1.0):
        """
        Output regularization for the velocites.
        args:
            scale_factor: Determines the output range [-scale_factor, scale_factor]
            input_scale_factor: Determines how fast the tanh should saturate. Lower (positive) input scale factor: slower saturation
        """
        super(ScaledTanh, self).__init__()
        assert scale_factor > 0.0
        assert input_scale_factor > 0.0
        self.scale_factor = scale_factor
        self.input_scale_factor = input_scale_factor

    def forward(self, inp):
        return self.scale_factor * torch.tanh(inp * self.input_scale_factor)


if __name__ == "__main__":
    a = torch.linspace(-10, 10, 100)
    tanh = ScaledTanh(scale_factor=0.5, input_scale_factor=0.2)
    out = tanh(a)

    import matplotlib.pyplot as plt

    plt.plot(a, out)
    plt.show()
