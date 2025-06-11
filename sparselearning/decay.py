import torch
import torch.nn as nn
import torch.optim as optim


class CosineDecay(object):
    def __init__(self, init_value, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(nn.ParameterList([nn.Parameter(torch.zeros(1))]), lr=init_value)
        self.cosine_stepper = optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_current_value(self):
        return self.sgd.param_groups[0]['lr']


class ExponentialDecay(object):
    def __init__(self, init_value, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency
        self.value = init_value

    def step(self):
        self.steps += 1
        if self.steps % self.frequency == 0:
            self.value *= self.factor

    def get_current_value(self):
        return self.value


class LinearDecay(object):
    def __init__(self, init_value, final_value, num_steps):
        self.init_value = init_value
        self.final_value = final_value
        self.num_steps = num_steps
        self.steps = 0
        self.value = init_value

    def step(self):
        self.steps += 1
        progress = min(self.steps / self.num_steps, 1)
        self.value = self.init_value + (self.final_value - self.init_value) * progress

    def get_current_value(self):
        return self.value


class ConstantDecay(object):
    """ Actually no decay, just keeps the value constant. """
    def __init__(self, value):
        self.value = value

    def step(self):
        pass

    def get_current_value(self):
        return self.value

class WSDDecay(object):
    def __init__(self, init_value, total_steps, final_ratio=0.1):
        """
        Args:
            init_value (float):
            total_steps (int):
            final_ratio (float):
        """
        self.init_value = init_value
        self.total_steps = total_steps
        self.final_ratio = final_ratio
        self.current_step = 0

        self.constant_steps = int(total_steps * 0.9)
        self.decay_steps = total_steps - self.constant_steps

    def step(self):
        self.current_step += 1

    def get_current_value(self):
        if self.current_step < self.constant_steps:
            return self.init_value
        else:
            progress = (self.current_step - self.constant_steps) / max(self.decay_steps, 1)
            decay_value = self.init_value * (1 - progress * (1 - self.final_ratio))
            return max(decay_value, self.init_value * self.final_ratio)
