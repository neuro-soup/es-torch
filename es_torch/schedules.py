import math

from es_torch.optim import Schedule


def constant(init_value: float) -> Schedule:
    def schedule(step: int) -> float:
        return init_value

    return schedule


def linear(init_value: float, end_value: float, decay_steps: int) -> Schedule:
    def schedule(step: int) -> float:
        if step >= decay_steps:
            return end_value
        progress = step / decay_steps
        return init_value + (end_value - init_value) * progress

    return schedule


def cosine(init_value: float, end_value: float, decay_steps: int) -> Schedule:
    def schedule(step: int) -> float:
        if step >= decay_steps:
            return end_value
        progress = step / decay_steps
        return end_value + (init_value - end_value) * 0.5 * (1 + math.cos(math.pi * progress))

    return schedule


SCHEDULES = {
    "constant": constant,
    "linear": linear,
    "cosine": cosine,
}
