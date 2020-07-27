#!/usr/bin/env Python3
import logging

import numpy as np

from typing import Dict, Tuple

from keras.engine.training import Model
from keras.utils import multi_gpu_model
from obspy import Stream, UTCDateTime
from obspy.core.event import (
    Event, CreationInfo, Pick, WaveformStreamID, ResourceIdentifier, 
    QuantityError)
from obspy.signal.trigger import trigger_onset

from gpd.helpers import sliding_window, get_components
from gpd.helpers.plotting import probability_plot
from gpd.models import MODELS


Logger = logging.getLogger(__name__)


class Phase(object):
    def __init__(
        self, 
        network: str,
        station: str, 
        time: UTCDateTime, 
        probability: float,
        phase_hint: str
    ):
        self.network = network
        self.station = station
        self.time = time
        self.probability = probability
        self.phase_hint = phase_hint

    def __repr__(self):
        return (
            f"Phase(station={self.station}, time={self.time}, "
            f"probability={self.probability}, phase_hint={self.phase_hint})"
        )

    def to_pick(self, trace_id: str) -> Pick:
        n, s, l, c = trace_id.split('.')
        assert s == self.station, f"Station {s} in trace id does not match {self.station}"
        return Pick(
            time=self.time, phase_hint=self.phase_hint, 
            method_id=ResourceIdentifier("smi:local/GeneralizedPhaseDetector"),
            evaluation_mode="automatic", 
            time_errors=QuantityError(confidence_level=self.probability),
            creation_info=CreationInfo(creation_time=UTCDateTime.now(),
                                       author="GeneralizedPhaseDetector"))


class GPD(object):
    sample_rate = 100.0  # Sample-rate trained on - Zach trained on 100 Hz data.
    verbose = False
    def __init__(
        self, 
        model: Model = None,
        freq_min: float = 3.0,
        freq_max: float = 20.0,
        n_shift: int = 10,
        n_gpu: int = None,
        batch_size: int = 3000,
        half_dur: float = 2.0,
        min_proba: float = 0.95,
    ):

        if model is None:
            model = MODELS["Ross_original"]
        self.model = model
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.n_shift = n_shift
        self.n_gpu = n_gpu
        self.batch_size = batch_size
        self.half_dur = half_dur
        self.n_win = int(half_dur * self.sample_rate)
        self.n_feat = 2 * self.n_win
        self.min_proba = min_proba
        if n_gpu > 1:
            tfback._get_available_gpus = _get_available_gpus
            model = multi_gpu_model(model, gpus=n_gpu)


    def detect(self, st: Stream, plot: bool = False):
        """ Make detections in a 3-component Stream. """
        networks = {tr.stats.network for tr in st}
        stations = {tr.stats.station for tr in st}
        p_phases, s_phases = [], []
        for network in networks:
            network_stream = st.select(network=network)
            stations = {tr.stats.station for tr in network_stream}
            for station in stations:
                Logger.info(f"Working on station {station}")
                station_stream = self.process_stream(
                    st=network_st.select(station=station))
                Logger.info(f"Using data: {station_stream}")
                try:
                    probabilities = self.predict(st=station_stream)
                except AssertionError as e:
                    Logger.warning(f"Skipping station {station} due to {e}")
                    continue
                _p_phases, _s_phases = self.trigger(
                    probability_times=probabilities["time"], 
                    p_probability=probabilities["p"], network=network,
                    s_probability=probabilities["s"], station=station,
                    starttime=station_stream[0].stats.starttime)
                Logger.info(
                    f"Found {len(_p_phases)} possible P phases and "
                    f"{len(_s_phases)} possible S phases.")
                p_phases.extend(_p_phases)
                s_phases.extend(_s_phases)
                if plot:
                    probability_plot(
                        st=station_stream, 
                        probability_times=probabilities['time'],
                        p_probabilities=probabilities["p"],
                        s_probabilities=probabilities["s"],
                        p_picks=[p.time for p in _p_phases],
                        s_picks=[p.time for p in _s_phases], show=True)
        return dict(p=p_phases, s=s_phases)


    def pick(self, st: Stream, plot: bool = False) -> Event:
        """ 
        Pick phases for a known event. No association is done. 
        
        Stream should be trimmed around the time of a expected phase
        arrivals to avoid picking the wrong event.
        """
        all_phases = self.detect(st=st, plot=plot)
        picks = []
        stations = {tr.stats.station for tr in st}
        for station in stations:
            station_phases = [p for _phases in all_phases.values() 
                              for p in _phases if p.station == station]
            if len(station_phases) == 0:
                Logger.info(f"No phases found for {station}.")
                continue
            tr_z, tr_n, tr_e = get_components(st.select(station=station))
            p_ids, s_ids = (tr_z.id, ), (tr_n.id, tr_e.id)
            for phase_hint, trace_ids in zip(("P", "S"), (p_ids, s_ids)):
                phases = [p for p in station_phases 
                          if p.phase_hint == phase_hint]
                if len(phases) > 1:
                    Logger.info(
                        f"Multiple {phase_hint}-phases found for {station}, "
                        "taking earliest")
                elif len(phases) == 0:
                    Logger.info(f"No {phase_hint}-phase found for {station}")
                    continue
                phase = sorted(phases, key=lambda p: p.time)[0]
                picks.extend(
                    [phase.to_pick(trace_id) for trace_id in trace_ids])
        event = Event(
            creation_info=CreationInfo(
                author="GeneralizedPhaseDetector", 
                creation_time=UTCDateTime.now()),
            picks=picks)
        return event

    def trigger(
        self, 
        probability_times: np.ndarray, 
        p_probability: np.ndarray,
        s_probability: np.ndarray,
        network: str,
        station: str,
        starttime: UTCDateTime,
    ) -> Tuple[Phase, Phase]:
        
        phases = {"P": [], "S": []}
        for phase_hint, detector in zip(("P", "S"), (p_probability, s_probability)):
            trigs = trigger_onset(
                detector, thres1=self.min_proba, thres2=0.1)
            for trig in trigs:
                if trig[1] == trig[0]:
                    continue
                pick = np.argmax(detector[trig[0]:trig[1]]) + trig[0]
                stamp_pick = starttime + probability_times[pick]
                phase = Phase(
                    station=station, time=stamp_pick, phase_hint=phase_hint,
                    probability=detector[pick], network=network)
                phases[phase_hint].append(phase)
        return phases["P"], phases["S"]

    def predict(self, st: Stream) -> dict:
        """ Predict P and S arrivals using the model. """
        assert len(st) == 3, "Only works with three-channel data"
        assert len({tr.stats.station for tr in st}) == 1, "Requires data from a single station"
        assert len({tr.stats.npts for tr in st}) == 1, "All channels must be equal length"
        assert len({tr.stats.delta for tr in st}) == 1, "Sampling rates of all channels must be equal"

        dt = st[0].stats.delta
        data_length = st[0].stats.npts
        tr_z, tr_n, tr_e = get_components(st)

        tt = (np.arange(0, data_length, self.n_shift, 
                        dtype=np.float32) + self.n_win) * dt
        # tt_i = np.arange(0, data_length, n_shift) + n_feat

        sliding_n = sliding_window(
            tr_n.data, self.n_feat, stepsize=self.n_shift)
        sliding_e = sliding_window(
            tr_e.data, self.n_feat, stepsize=self.n_shift)
        sliding_z = sliding_window(
            tr_z.data, self.n_feat, stepsize=self.n_shift)
        tr_win = np.zeros((sliding_e.shape[0], self.n_feat, 3))
        tr_win[:,:,0] = sliding_n
        tr_win[:,:,1] = sliding_e
        tr_win[:,:,2] = sliding_z
        tr_win = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:, None, None]
        tt = tt[:tr_win.shape[0]]
        # tt_i = tt_i[:tr_win.shape[0]]

        ts = self.model.predict(tr_win, verbose=self.verbose,
                                batch_size=self.batch_size)
        prob_S = ts[:,1]
        prob_P = ts[:,0]
        prob_N = ts[:,2]
        return dict(time=tt, p=prob_P, s=prob_S, null=prob_N)

    def process_stream(self, st: Stream) -> Stream:
        """ Process data - filtering, gap handling and resampling. """
        st_out = st.copy()

        st_out.merge()
        starttime = max(tr.stats.starttime for tr in st_out)
        endtime = min(tr.stats.endtime for tr in st_out)
        st_out.trim(starttime=starttime, endtime=endtime)
        st_out.split()
        st_out.detrend(type="linear")

        # Check for gaps and fill if needed
        gaps = st_out.split().get_gaps()
        if len(gaps) > 0:
            Logger.warning("Gaps found - interpolating.")
            st_out.merge(fill_value="interpolate")
        
        if self.freq_min and self.freq_max:
            st_out.filter(type="bandpass", freqmin=self.freq_min,
                          freqmax=self.freq_max)
        elif self.freq_min:
            st_out.filter(type="highpass", freq=freq_min)
        elif self.freq_max:
            st_out.filter(type="lowpass", freq=freq_max)

        for tr in st:
            if tr.stats.sampling_rate != self.sample_rate:
                tr.interpolate(self.sample_rate)
        return st_out
