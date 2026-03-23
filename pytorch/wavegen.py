import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def normalize(
    array: np.array,
    min_: float = None,
    max_: float = None
):
    if len(np.unique(array)) == 1:
        value = np.unique(array)[0]
        norm_value = 0.5
        if max_ != min_:
            norm_value = (value - min_) / (max_ - min_)
        return np.ones_like(array) * norm_value
    if min_ is None:
        min_ = array.min()
    if max_ is None:
        max_ = array.max()
    return (array - min_) / (max_ - min_)


def denormalize(
    normalized_array: np.array,
    min_: float,
    max_: float
) -> np.array:
    return normalized_array * (max_ - min_) + min_


def generate_wave(
    n_years=None,
    frequency=2 * np.pi / 365.25,
    beta=1,
    variation=False,
    elevation_power=0,
    amplitude_change_power=0,
    noise_level=1e-1
):
    if n_years is None:
        n_years = np.random.randint(1, 15)
    amplitude = np.random.randint(-100, 100)
    phase_shift = np.random.randint(-180, 180)
    period_starts = pd.Timestamp('now')
    a = np.random.randint(-500, 500) / (n_years * 365.25)
    b = np.random.randint(-100, 100)
    duration_days = int(365 * n_years)
    period_ends = period_starts + pd.Timedelta(days=duration_days)
    x = np.linspace(1, duration_days, duration_days)
    if not variation:
        y = amplitude * np.sin(frequency * x + phase_shift)
    else:
        y = amplitude * (x ** amplitude_change_power) * np.sin(x ** beta * frequency + phase_shift)
    y += a * x + b
    y += -1 + x ** elevation_power
    if noise_level:
        min_, max_ = y.min(), y.max()
        y = normalize(y, min_, max_)
        y += noise_level * np.random.randn(len(y))
        y = denormalize(y, min_, max_)
    x_date = pd.date_range(period_starts, period_ends, freq='D')[:-1]
    return x, y, x_date


def generate_multiwave(
    N,
    n_years=None,
    beta=1,
    variation=True,
    elevation_power=0,
    amplitude_change_power=0,
    noise_level=1e-1
):
    if n_years is None:
        n_years = np.random.randint(1, 15)
    period_starts = pd.Timestamp('now')
    duration_days = int(365 * n_years)
    period_ends = period_starts + pd.Timedelta(days=duration_days)
    x_date = pd.date_range(period_starts, period_ends, freq='D')[:-1]
    x = np.linspace(1, duration_days, duration_days)
    a = np.random.randint(-500, 500) / (n_years * 365.25)
    b = np.random.randint(-100, 100)
    waves = {}
    for i in range(N):
        frequency = 2 * np.pi / 365.25 * (i + 1)
        amplitude = np.random.randint(-100, 100)
        phase_shift = np.random.randint(-180, 180)
        if not variation:
            waves[i] = amplitude * np.sin(frequency * x + phase_shift)
        else:
            waves[i] = amplitude * (x ** amplitude_change_power) * np.sin(x ** beta * frequency + phase_shift)
    y = np.sum(list(waves.values()), axis=0)
    linear_component = a * x + b
    exponential_component = x ** elevation_power
    y = y + linear_component + exponential_component
    if noise_level:
        min_, max_ = y.min(), y.max()
        y = normalize(y, min_, max_)
        y = y + noise_level * np.random.randn(len(y))
        y = denormalize(y, min_, max_)
    return x, y, x_date, waves, linear_component, exponential_component


# x, y, x_date, waves, linear_component, exponential_component = generate_multiwave(5)
# x, y, x_date = generate_wave(n_years=None)
# plt.figure(figsize=(24, 3))
# plt.plot(x_date, y)
# plt.title(f'Single SIN-wave generated.')
# plt.show()