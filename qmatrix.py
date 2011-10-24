"""
Computes sodium currents using pyqmatrix or NEURON.
"""
# C. Schmidt-Hieber, University College London
# 2010-07-01
#
# accompanies the publication:
# Schmidt-Hieber C, Bischofberger J. (2010)
# Fast sodium channel gating supports localized and efficient 
# axonal action potential initiation.
# J Neurosci 30:10233-42

import numpy as np
import time
import sys

import qmatpy as qmat
from neuron import h
import mech
import stfio_plot
import fft_filter

v_rev_na = 55.6
gna_soma = 188

class gating_model:
    def __init__(self, func, act_func, inact_func, title, pars=None,
                 filename = None, 
                 linestyle="-k",
                 plot=False, 
                 nrn_mech="", nrn_vshift=0, nrn_hScale=1.0,
                 h_vshift=0, plot_vshift=0):
        self.func = func
        self.act_func = act_func
        self.inact_func = inact_func
        self.title = title
        self.linestyle = linestyle
        if pars==None and filename==None:
            print "Error initializing model", title, "aborting now"
            sys.exit(1)
        if pars == None:
            self.pars = np.loadtxt(file(filename))
        else:
            self.pars = pars
        self.filename = filename    
        self.plot = plot
        self.nrn_mech = nrn_mech
        self.nrn_vshift = nrn_vshift
        self.nrn_hScale = nrn_hScale
        self.h_vshift = h_vshift
        self.plot_vshift = plot_vshift

def init_st8_model():
    file_st8_soma = "dat/soma_st8.txt"
    file_st8_axon = "dat/axon_st8.txt"
    st8_model = \
        gating_model(init_matrix, init_act, init_inact, 
                     "soma_st8_nrn", filename=file_st8_soma,
                     nrn_mech="na8st")

    return st8_model

def init_act(V, p, max_rate = 8e+03):
    ret = np.array([ \
            p[0]*np.exp(p[1]*V),
            p[2]*np.exp(-p[3]*V),

            p[4]*np.exp(p[5]*V),
            p[6]*np.exp(-p[7]*V),

             p[8]*np.exp( p[9]*V),
            p[10]*np.exp(-p[11]*V)
            ])
    if max_rate > 0:
        ret = ret * max_rate / (ret + max_rate)
    return ret

def init_inact(V, p, max_rate = 8e+03, h_vshift=0):
    ret = np.array([ \
            p[15]/(1+p[16]*np.exp(p[17]*(V-h_vshift))),
            p[12]/(1+p[13]*np.exp(-p[14]*(V-h_vshift)))
            ])
    if max_rate > 0:
        ret = ret * max_rate / (ret + max_rate)
    return ret

def init_matrix(V, model, debug=False):
    """Initialises a Q matrix for the 8-state gating scheme
    """
    (a_1, b_1, a_2, b_2, a_3, b_3) = init_act(V, model.pars)
    (a_h, b_h) = init_inact(V, model.pars, h_vshift=model.h_vshift)

    # calculate rates from current parameter estimates:
    #                 1    2    3    4    5    6    7    8
    Q = np.array([[   0, b_3,   0,   0,   0,   0,   0, b_h], \
                  [ a_3,   0, b_2,   0,   0,   0, b_h,   0], \
                  [   0, a_2,   0, b_1,   0, b_h,   0,   0], \
                  [   0,   0, a_1,   0, b_h,   0,   0,   0], \
                  [   0,   0,   0, a_h,   0, a_1,   0,   0], \
                  [   0,   0, a_h,   0, b_1,   0, a_2,   0], \
                  [   0, a_h,   0,   0,   0, b_2,   0, a_3], \
                  [ a_h,   0,   0,   0,   0,   0, b_3,   0]])

    # Update diagonal elements:
    qmat.init_matrix(Q)

    if debug: 
        sys.stdout.write("Initialised 8-state Q matrix: %s\n" % Q)

    return Q

def init_nrn_patch(model, dt):
    h("load_file(\"stdrun.hoc\")")
    h("load_file(\"./hoc/config.hoc\")")
    h.celsius = 24.0
    h.dt = dt
    h.steps_per_ms = 1.0/h.dt

    # create a patch:
    patch = h.Section()
    patch.L = 1
    patch.diam = 1
    patch.Ra = 1e-9
    
    # insert mechanism:
    patch.insert(model.nrn_mech)
    patch.ena = v_rev_na

    if model.nrn_mech == "na8st":
        # read somatic best-fit rates from file:
        rates = np.loadtxt(model.filename)
        for seg in patch:
            mech.set_rates_na8st(seg, rates, vshift_inact=0)
        # vShift is global, so we don't set it for a segment
        h("vShift_na8st = %f" % model.nrn_vshift) 
        h("vShift_inact_na8st = %f" % model.h_vshift)

    # insert a v clamp:
    vc = h.SEClamp(patch(0.5))
    vc.rs = 1e-9

    # record current:
    irec = h.Vector()
    irec.record(patch(0.5)._ref_ina, sec=patch)

    return patch, vc, irec

def activation(model, v_range, dt=0.001, filter=-1, plot_fit=True):
    traces, pulses = [], []
    times_qm, times_nrn = [], []
    trange = np.arange(0, 8.0, dt)

    # initialise Q-matrix
    Q_m120 = model.func(-120.0, model)
    p_m120 = qmat.p_inf(Q_m120)

    # initialise NEURON
    patch, vc, irec = init_nrn_patch(model, dt)
    vc.dur1 = trange[-1]
    h.tstop = vc.dur1

    for (n_v, v) in enumerate(v_range):
        # Q-Matrix
        time0 = time.time()
        Qv = model.func(v, model)
        pv = qmat.p_inf(Qv)
        lv, Av = qmat.mat_solve(Qv)
        trace = \
            qmat.p(trange, p_m120, pv, lv, Av)[0]# * \
            # (v-v_rev_na) * 1e-3 * gna_soma
        trace /= np.max(trace)

        times_qm.append(time.time()-time0)
        if filter > 0:
            trace = fft_filter.gaussian_filter(trace, filter, dt)
        traces.append(
            stfio_plot.timeseries(trace, dt, 
                                  xunits="ms", 
                                  yunits="",
                                  linestyle=model.linestyle, linewidth=2.0))

        # NEURON
        time0 = time.time()
        vc.amp1 = v

        h.v_init = -120.0
        h.init()
        h.run()
        times_nrn.append(time.time()-time0)
        trace_nrn = np.array(irec)
        trace_nrn /= np.min(trace_nrn)
        # trace_nrn *= norm

        if filter > 0:
            trace = fft_filter.gaussian_filter(trace, filter, dt)
        traces.append(
            stfio_plot.timeseries(trace_nrn, h.dt, 
                                  xunits="ms",
                                  yunits="",
                                  linestyle="--r", linewidth=2.0))

        pulse = np.array([v for i in range(len(trace))])
        pulse[0] = -120.0
        pulses.append(
            stfio_plot.timeseries(pulse, dt, 
                                  xunits="ms",
                                  yunits="mV",
                                  linestyle="-k", linewidth=2.0))

    stfio_plot.plot_traces(traces, pulses)

    stfio_plot.plt.show()

    sys.stdout.write("NEURON: %f\n" % np.array(times_nrn).mean())
    sys.stdout.write("Q-Matrix: %f\n" % np.array(times_qm).mean())

if __name__=="__main__":
    model = init_st8_model()
    vrange = np.arange(-70.0, 40.0, 10.0)
    activation(model, vrange)
