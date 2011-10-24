"""
Initialize mechanisms in all compartments.
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

from scipy.integrate import quad
from scipy.optimize import brentq

import sys

import ap
from ap_utils import *

gVDonnan = 12.0
gENa = 75.0
gEK = -95.0

gDebug = False

def dens_func(x, gNas, gNaa, a0, lambda2):
    """
    Returns Nav conductance density (gNa) along the axon.

    Arguments:
    x       -- Axonal distance from soma
    gNas    -- Somatic gNa
    gNaa    -- Axonal density
    a0      -- Scaling factor
    lambda2 -- Axonal decay length constant

    Returns:
    Axonal gNa
    """
    lambda1 = 5.0
    return gNas + (gNaa-gNas) * (1.0-np.exp(-x/lambda1)) * \
        (1.0 + a0*np.exp(-x/lambda2))

def read_rates_HH_EJ(cell, settings):
    """
    Reads in rates for the E&J kinetic scheme from a file
    and applies them to the cell.
    
    If settings.uniform_kin is True, rates will be set to 
    the same values throughout the cell. Otherwise, somatic 
    inactivation rates will be half the axonal values.
    """


    rates = np.loadtxt(file("./dat/EJ_HH.txt", 'r'))

    # set distance origin to axon-soma border:
    ap.h.distance(0,0.0, sec=cell.somaBorderLoc.secRef.sec)

    axo_names = [ axon.name() for axon in cell.axo ]

    for section in cell.all:
        for seg in section:
            dist = ap.h.distance(seg.x, sec = section)
            n_r = 0
            seg.hhmfb.am0 = rates[n_r]; n_r += 1
            seg.hhmfb.am1 = rates[n_r]; n_r += 1
            seg.hhmfb.am2 = rates[n_r]; n_r += 1
            seg.hhmfb.bm0 = rates[n_r]; n_r += 1
            seg.hhmfb.bm1 = rates[n_r]; n_r += 1
            seg.hhmfb.ah0 = rates[n_r]; n_r += 1
            seg.hhmfb.ah1 = rates[n_r]; n_r += 1
            seg.hhmfb.bh0 = rates[n_r]; n_r += 1
            seg.hhmfb.bh1 = rates[n_r]; n_r += 1
            seg.hhmfb.bh2 = rates[n_r]
            if not settings.uniform_kin:
                if not (section.name() in axo_names):
                    hScale = 0.5
                else:
                    hScale = 0.5 + fsigm(dist, 0.5, 20, 20)
            else:
                hScale = 1.0
            seg.hhmfb.hScale = hScale

def read_rates_HH(cell, settings):
    """
    Reads in rates for the HH kinetic scheme from files
    and applies them to the cell.
    
    If settings.uniform_kin is True, rates will be set 
    to the same values throughout the cell.
    """
    # set distance origin to axon-soma border:
    ap.h.distance(0,0.0, sec=cell.somaBorderLoc.secRef.sec)

    # read best-fit rates from files:
    rates_soma = np.loadtxt(file("./dat/soma_HH.txt", 'r'))
    if settings.uniform_kin:
        rates_axon = rates_soma
    else:
        rates_axon = np.loadtxt(file("./dat/axon_HH.txt", 'r'))
    center = 2
    slope = 2
    axo_names = [ axon.name() for axon in cell.axo ]
    for section in cell.all:
        for seg in section:
            dist = ap.h.distance(seg.x, sec = section)
            if not (section.name() in axo_names):
                dist = -dist
            n_r = 0
            rates_seg = fsigm(dist, (rates_axon - rates_soma), center, 
                                   slope) + rates_soma
            seg.HHrates.am0 = rates_seg[n_r]; n_r += 1
            seg.HHrates.am1 = rates_seg[n_r]; n_r += 1
            seg.HHrates.am2 = rates_seg[n_r]; n_r += 1
            seg.HHrates.bm0 = rates_seg[n_r]; n_r += 1
            seg.HHrates.bm1 = rates_seg[n_r]; n_r += 1
            seg.HHrates.ah0 = rates_seg[n_r]; n_r += 1
            seg.HHrates.ah1 = rates_seg[n_r]; n_r += 1
            seg.HHrates.bh0 = rates_seg[n_r]; n_r += 1
            seg.HHrates.bh1 = rates_seg[n_r]; n_r += 1
            seg.HHrates.bh2 = rates_seg[n_r]

def set_rates_na8st(seg, rates, vshift_inact=0):
    """
    Sets the rates for the 8-state model in segment seg.
    vshift_inact can be used to locally change the voltage
    dependence of inactivation rates.
    """
    n_r = 0
    seg.na8st.a1_0 = rates[n_r]; n_r += 1 #0
    seg.na8st.a1_1 = rates[n_r]; n_r += 1 #1
            
    seg.na8st.b1_0 = rates[n_r]; n_r += 1 #2
    seg.na8st.b1_1 = rates[n_r]; n_r += 1 #3
            
    seg.na8st.a2_0 = rates[n_r]; n_r += 1 #4
    seg.na8st.a2_1 = rates[n_r]; n_r += 1 #5
            
    seg.na8st.b2_0 = rates[n_r]; n_r += 1 #6
    seg.na8st.b2_1 = rates[n_r]; n_r += 1 #7
            
    seg.na8st.a3_0 = rates[n_r]; n_r += 1 #8
    seg.na8st.a3_1 = rates[n_r]; n_r += 1 #9
            
    seg.na8st.b3_0 = rates[n_r]; n_r += 1 #10
    seg.na8st.b3_1 = rates[n_r]; n_r += 1 #11
    
    seg.na8st.bh_0 = rates[n_r]; n_r += 1 #12
    seg.na8st.bh_1 = rates[n_r]; n_r += 1 #13
    seg.na8st.bh_2 = rates[n_r]; n_r += 1 #14
    
    seg.na8st.ah_0 = rates[n_r]; n_r += 1 #15
    seg.na8st.ah_1 = rates[n_r]; n_r += 1 #16
    seg.na8st.ah_2 = rates[n_r]           #17

    seg.na8st.vShift_inact_local = vshift_inact

def read_rates_na8st(cell, settings):
    """
    Reads in rates for the HH kinetic scheme from files
    and applies them to the cell.
    
    If settings.uniform_kin is True, rates will be set 
    to the same values throughout the cell.
    """
    # set distance origin to axon-soma border:
    ap.h.distance(0,0.0, sec=cell.somaBorderLoc.secRef.sec)

    # read best-fit rates from files:
    rates_soma = np.loadtxt(file(
            "./dat/soma_st8.txt", 'r'))
    rates_axon = np.loadtxt(file(
            "./dat/axon_st8.txt", 'r'))

    if settings.uniform_kin:
        rates_axon = rates_soma

    center = 2.0
    slope = 2.0
    axo_names = [ axon.name() for axon in cell.axo ]

    for section in cell.all:
        for seg in section:
            dist = ap.h.distance(seg.x, sec = section)
            if not section.name() in axo_names:
                dist = -dist
            rates_seg = fsigm(dist, (rates_axon - rates_soma), center,
                              slope) + rates_soma
            vshift_inact = 0
            set_rates_na8st(seg, rates_seg, vshift_inact)


def init_rates(cell, settings):
    """
    Initializes rates throughout the cell according to
    settings.
    """
    if settings.gating == "ej":
        read_rates_HH_EJ(cell, settings.uniform_kin)
        read_rates_na8st(cell, settings)
    if settings.gating == "hh":
        read_rates_HH(cell, settings.uniform_kin)
    if settings.gating == "8s":
        read_rates_na8st(cell, settings)

def init_mech(cell, settings):
    """
    Inserts mechanisms throughout the cell according to
    settings.
    """
    if settings.has_donnan:
        vdonnan = gVDonnan
    else:
        vdonnan = 0
    axo_names = [ axon.name() for axon in cell.axo ]

    for section in cell.all:
        if settings.gating == "ms" or settings.gating == "ms_shift":
            for seg in section:
                if not section.name() in axo_names:
                    if not hasattr(seg, "nakole"):
                        section.insert("nakole")
                else:
                    if not hasattr(seg, "naxkole"):
                        section.insert("naxkole")
                if not hasattr(seg, "KIn"):
                    section.insert("KIn")
        if settings.gating == "8s":
            for seg in section:
                if not hasattr(seg, "na8st"):
                    section.insert("na8st")
                if not hasattr(seg, "KIn"):
                    section.insert("KIn")
        if settings.gating == "hh":
            for seg in section:
                if not hasattr(seg, "HHrates"):
                    section.insert("HHrates")
                if not hasattr(seg, "KIn"):
                    section.insert("KIn")
        if settings.gating == "ej":
            for seg in section:
                if not hasattr(seg, "na8st"):
                    section.insert("na8st")
                if not hasattr(seg, "hhmfb"):
                    section.insert("hhmfb")
                if not hasattr(seg, "KIn"):
                    section.insert("KIn")
        
    if settings.gating == "ej":
        # vShift is global, so we don't set it for a segment
        ap.h("vShift_hhmfb = %f" % vdonnan) 
        ap.h("vShift_inact_hhmfb = 10")
        ap.h("vShift_na8st = %f" % vdonnan)
        ap.h("maxrate_na8st = 8.0e+03")
        ap.h("vShift_inact_na8st = 10")
    if settings.gating == "hh":
        ap.h("vShift_HHrates = %f" % vdonnan) # vShift is global, so we don't set it for a segment
        ap.h("vShift_inact_HHrates = 10")
    if settings.gating == "8s":
        # vShift is global, so we don't set it for a segment
        ap.h("vShift_na8st = %f" % vdonnan)
        ap.h("maxrate_na8st = 8.0e+03")
        ap.h("vShift_inact_na8st = 10")

    # 1. vshift has a different polarity in Mainen's model than in our model: 
    #    positive values denote a shift to the left, i.e. the Donnan potential
    #    needs to be subtracted.
    # 2. vshift of 10 mV towards the left (positive) account for the different
    #    midpoints of the activation and inactivation curves.
    # 3. inactivation shift of 10 mV as in our recordings; again, the goal is
    #    to get the inactivation curve midpoint to the same potential as in 
    #    our model
    # 4. vshift of 10 mV towards the left has been used by Kole et al. 
    if settings.gating == "ms_shift":
        for section in cell.all:
            for seg in section:
                if not section.name() in axo_names:
                    seg.nakole.vshift = 10 + 10 - vdonnan # local
                else:
                    seg.naxkole.vshift = 10 + 10 - vdonnan # local
        ap.h("vShift_inact_nakole = 10") # global
        ap.h("vShift_inact_naxkole = 10") # global
    elif settings.gating == "ms":
        # shift only as per Kole et al., 2008
        for section in cell.all:
            for seg in section:
                if not section.name() in axo_names:
                    seg.nakole.vshift = 10 # local
                else:
                    seg.naxkole.vshift = 10 # local

def init_g(cell, settings):
    """
    Initializes conductance densities throughout the cell according to
    settings. 
    """

    if not settings.uniform_g:
        gnabar_prox_axon = settings.gna_prox_axon
        gnabar_distal_axon = settings.gna_distal_axon
    else:
        gnabar_prox_axon = settings.gna_soma
        gnabar_distal_axon = settings.gna_soma

    gnabar_distal_dend = settings.gna_soma * 0.2
    lambda2 = 10.0
    dend_50 = 80.0
    dend_slope = 40.0

    # make the somatic border the origin of distance calculations:
    ap.h.distance(0,0.0, sec=cell.somaBorderLoc.secRef.sec)
    dist_x_dend = np.arange(-150,0,0.1)
    dist_x_axon = np.arange(0,150,0.1)
    dist_x = np.concatenate((dist_x_dend, dist_x_axon))

    bind_dens = \
        lambda x, amp_gauss : \
        dens_func(x, settings.gna_soma, gnabar_distal_axon, amp_gauss, lambda2)
    int_a = 0
    int_b = 40
    bind_dens_int = \
        lambda amp_gauss : \
        quad(bind_dens, int_a, int_b, args=(amp_gauss,), limit=100)[0] / \
            float(int_b-int_a)
    leastsq_bis = lambda amp_gauss_l, y: y - bind_dens_int(amp_gauss_l)
    # set the amplitude of the Gaussian so that the mean density 
    # in the axon matches the recorded values:
    if not settings.uniform_g:
        plsq_g = \
            brentq(leastsq_bis, 0.0*settings.gna_soma, 20.0*gnabar_prox_axon,
                   args=(gnabar_prox_axon,))
    else:
        plsq_g = 0.0

    gna_y_dend = \
        (settings.gna_soma - fsigm(-dist_x_dend, settings.gna_soma-gnabar_distal_dend, 
                            dend_50, dend_slope))*10.0
    gna_y_axon = bind_dens(dist_x_axon, plsq_g)*10.0
    gna_mean_prox = quad(bind_dens, int_a, int_b, args=(plsq_g,), 
                         limit=100)[0] / float(int_b-int_a)
    if gDebug:
        sys.stdout.write("Mean density in proximal axon: %f\n" %
                         (quad(bind_dens, int_a, int_b, args=(plsq_g,), 
                               limit=100)[0] / float(int_b-int_a)))
        sys.stdout.write("Axon: dens_func(x, %f, %f, %f, %f)\n" %
                         (settings.gna_soma, plsq_g, gnabar_distal_axon, 
                          lambda2))
        sys.stdout.write(
            "Dendrite: (%f-fsigm(x, %f, %f, %f)) * 10.0\n" %
            (settings.gna_soma, settings.gna_soma-gnabar_distal_dend, 
             dend_50, dend_slope))
        sys.stdout.write("amp_gauss= %f\n" % plsq_g)
        sys.stdout.write("Maximum at %f um: %f\n" % 
                         (dist_x_axon[gna_y_axon.argmax()],
                          gna_y_axon.max()))

    gna_y = np.concatenate((gna_y_dend, gna_y_axon))

    axo_names = [ axon.name() for axon in cell.axo ]

    for sec in cell.all:
        sec.ena = gENa
        sec.ek  = gEK
        for seg in sec:
            dist = -ap.h.distance(seg.x, sec=sec)
            gnabar_dend = \
                settings.gna_soma - fsigm(-dist, settings.gna_soma-gnabar_distal_dend, 
                                       dend_50, dend_slope)
            if hasattr(seg, 'HHrates'):
                seg.HHrates.gbar = 0
            if hasattr(seg, 'hhmfb'):
                seg.hhmfb.gnabar = 0
            if hasattr(seg, 'na8st'):
                seg.na8st.gbar =0
            if hasattr(seg, 'nakole'):
                seg.nakole.gbar = 0
            if hasattr(seg, 'naxkole'):
                seg.naxkole.gbar = 0
            if settings.gating == "hh":
                seg.HHrates.gbar = gnabar_dend * seg.spines.scale
            if settings.gating == "ej":
                if not sec.name() in axo_names:
                    seg.na8st.gbar =  gnabar_dend * seg.spines.scale
                    seg.hhmfb.gnabar = 0
                seg.hhmfb.gkbar  = 0
                seg.hhmfb.gl     = 0
                seg.hhmfb.hScale = 1.0
            if settings.gating == "8s":
                seg.na8st.gbar =  gnabar_dend * seg.spines.scale
            if settings.gating == "ms" or settings.gating == "ms_shift":
                if not sec.name() in axo_names:
                    # convert to pS / um^2
                    seg.nakole.gbar  = \
                        gnabar_dend * seg.spines.scale * 10.0
            seg.KIn.gkbar  = settings.gk_soma * seg.spines.scale 
            seg.KIn.scale_a = settings.gk_scale_soma
            seg.KIn.scale_i = 1.0e-9
            seg.pas.e     = -82.0

    for sec in cell.axo:
        for seg in sec:
            # distance from soma:
            dist = ap.h.distance(seg.x, sec=sec)
            gnabar_gauss = bind_dens(dist, plsq_g)
                
            gkbar_ax = \
                settings.gk_axon + \
                fsigm(dist, (settings.gk_distal_axon - settings.gk_axon), 
                           200.0, 100.0)
            if settings.gating == "hh":
                seg.HHrates.gbar = gnabar_gauss * seg.spines.scale
            if settings.gating == "8s":
                seg.na8st.gbar =   gnabar_gauss * seg.spines.scale
            if settings.gating == "ej":
                seg.hhmfb.gnabar = gnabar_gauss * seg.spines.scale * 1e-3
                seg.na8st.gbar =  0
            if settings.gating == "ms" or settings.gating == "ms_shift":
                # convert to pS / um^2
                seg.naxkole.gbar  = gnabar_gauss * seg.spines.scale * 10.0 

            seg.KIn.gkbar  = gkbar_ax * seg.spines.scale
            seg.KIn.scale_a  = settings.gk_scale_axon
            seg.KIn.scale_i  = 1.0e-9

    return dist_x, gna_y, plsq_g, gna_mean_prox


def gna_per_area(cell):
    """
    Compute actual density:
    If determined according to our experimental data and methods,
    mean \bar{g}_{Na} in the proximal axon is 940 pS um^{2}. This 
    is not necessarily the same as the summed \bar{g}_{Na} divided 
    by the summed membrane area along the proximal axon.
    Thanks to Steffen Platschek for pointing out this discrepancy.
    """

    # set distance origin to axon-soma border:
    ap.h.distance(0,0.0, sec=cell.somaBorderLoc.secRef.sec)

    area = 0.0
    G = 0.0
    for sec in cell.axo:
        for seg in sec:
            dist = ap.h.distance(seg.x, sec = sec)
            if 0 <= dist <= 40:
                G += seg.na8st.gbar * ap.h.area(seg.x, sec=sec)
                area += ap.h.area(seg.x, sec=sec)
    
    return G/area
