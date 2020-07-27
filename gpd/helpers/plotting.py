
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

from obspy import Stream


def probability_plot(
    st: Stream, 
    probability_times: np.ndarray,
    p_probabilities: np.ndarray,
    s_probabilities: np.ndarray,
    p_picks: Iterable[float],
    s_picks: Iterable[float],
    show: bool = True,
) -> plt.Figure:
    assert len(st) == 3, "Plot requires three-channel data"

    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 12), sharex=True)

    for i, tr in enumerate(st):
        ax[i].plot(tr.times(), tr.data, lw=0.5, label=tr.id)
        ax[i].legend()

    ax[3].plot(probability_times, p_probabilities, c='r', lw=0.5, 
               label="P-probability")
    ax[3].plot(probability_times, s_probabilities, c='b', lw=0.5,
               label="S-probability")

    for p_pick in p_picks:
        for i in range(3):
            ax[i].axvline(p_pick - st[0].stats.starttime, c='r', lw=1.0)

    for s_pick in s_picks:
        for i in range(3):
            ax[i].axvline(s_pick - st[0].stats.starttime, c='b', lw=1.0)

    ax[3].legend()
    ax[3].set_ylabel("Probability")
    ax[3].set_xlabel(f"Time (s) from UTC: {st[0].stats.starttime}")
    ax[3].set_ylim(0.0, 1.0)

    ax[0].set_xlim(0, max(st[0].times()))

    fig.tight_layout()
    if show:
        plt.show()
    return fig