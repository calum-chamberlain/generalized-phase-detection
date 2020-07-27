from obspy import Stream, Trace
from typing import Tuple


def get_components(st: Stream) -> Tuple[Trace, Trace, Trace]:
    """ 
    Get Z, N and E components of the stream.

    Maps 1 -> N and 2 -> E.

    Returns
    -------
    Traces ordered Z, N, E
    """
    st_z = st.select(component="Z")
    st_n = st.select(component="N")
    if len(st_n) == 0:
        st_n = st.select(component="1")
    st_e = st.select(component="E")
    if len(st_e) == 0:
        st_e = st.select(component="2")
    if len(st_z) == 0:
        raise NotImplementedError("Requires Z data")
    tr_z = st_z[0]
    if len(st_e) == 0:
        raise NotImplementedError("Requires E or 2 data")
    tr_e = st_e[0]
    if len(st_n) == 0:
        raise NotImplementedError("Requires N or 1 data")
    tr_n = st_n[0]
    return tr_z, tr_n, tr_e
