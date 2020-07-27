#! /bin/env python
# Automatic picking of seismic waves using Generalized Phase Detection 
# See http://scedc.caltech.edu/research-tools/deeplearning.html for more info
#
# Ross et al. (2018), Generalized Seismic Phase Detection with Deep Learning,
#                     Bull. Seismol. Soc. Am., doi:10.1785/0120180080
#                                              
# Author: Zachary E. Ross (2018)                
# Contact: zross@gps.caltech.edu                        
# Website: http://www.seismolab.caltech.edu/ross_z.html  
# 
# Editted heavily by Calum Chamberlain (2020) to be more Pythonic       
import os
import argparse as ap

from gpd import GPD

from gpd.helpers import sliding_window, _get_available_gpus
from gpd.helpers.plotting import probability_plot

#####################
# Hyperparameters
min_proba = 0.95 # Minimum softmax probability for phase detection
freq_min = 3.0
freq_max = 20.0
filter_data = True
decimate_data = False # If false, assumes data is already 100 Hz samprate
n_shift = 10 # Number of samples to shift the sliding window at a time
n_gpu = 1 # Number of GPUs to use (if any)
#####################
batch_size = 1000*3

half_dur = 2.00
only_dt = 0.01
n_win = int(half_dur/only_dt)
n_feat = 2*n_win

#-------------------------------------------------------------

def main():
    parser = ap.ArgumentParser(
        prog='gpd_predict.py',
        description='Automatic picking of seismic waves using'
                    'Generalized Phase Detection')
    parser.add_argument(
        '-I', "--infile", type=str, default=None,
        help='Input file')
    parser.add_argument(
        '-O', "--outfile", type=str, default=None,
        help='Output file')
    parser.add_argument(
        '-P', "--plot", action="store_true", help="Show plots")
    parser.add_argument(
        '-V', "--verbose", action="store_true",
        help='verbose')
    args = parser.parse_args()

    # Reading in input file
    fdir = []
    with open(args.infile) as f:
        for line in f:
            tmp = line.split()
            fdir.append([tmp[0], tmp[1], tmp[2]])
    nsta = len(fdir)

    detector = GPD(
        freq_min=freq_min, freq_max=freq_max, n_shift=n_shift, n_gpu=n_gpu,
        batch_size=batch_size, half_dur=half_dur, n_win=n_win, n_feat=n_feat,
        min_proba=min_proba)
    
    phases = []   
    for i in range(nsta):
        if not os.path.isfile(fdir[i][0]):
            print("%s doesn't exist, skipping" % fdir[i][0])
            continue
        if not os.path.isfile(fdir[i][1]):
            print("%s doesn't exist, skipping" % fdir[i][1])
            continue
        if not os.path.isfile(fdir[i][2]):
            print("%s doesn't exist, skipping" % fdir[i][2])
            continue
        st = oc.Stream()
        for fname in fdir[i]:
            st += oc.read(fname)
        phases.extend(detector.detect(st, plot=args.plot))

    with open(args.outfile, "w" as f):
        for phase in phases:
            f.write("%s %s S %s\n" % (
                phase.network, phase.sta, phase.time.isoformat()))


if __name__ == "__main__":
    main()
