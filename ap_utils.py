import numpy as np

from scipy import fftpack
from scipy.interpolate import fitpack

import sys

import ap

class RunSettings(object):
    def __init__(self, gating=None, uniform_g=False, uniform_kin=False,
                 has_donnan=True,
                 gna_soma = 18.8,
                 gna_prox_axon = 94.0,
                 gna_distal_axon = 38.6152,
                 gk_scale_axon = 3.0, gk_scale_soma = 1.0, 
                 gk_axon = 0.004, gk_soma = 0.004, 
                 gk_distal_axon = 0.010):
        # Nav gating model
        self.gating = gating
        # Uniform gNa density
        self.uniform_g = uniform_g
        # Uniform gNa kinetics
        self.uniform_kin = uniform_kin
        # Apply correction for Donnan potential
        self.has_donnan = has_donnan
        # Somatic gNa
        self.gna_soma = gna_soma
        # Mean gNa in proximal axon (up to 40 um from soma)
        self.gna_prox_axon = gna_prox_axon
        # Distal axon gNa:
        self.gna_distal_axon = gna_distal_axon
        # Scaling factor for axonal Kv activation rates:
        self.gk_scale_axon = gk_scale_axon
        # Scaling factor for somatic Kv activation rates:
        self.gk_scale_soma = gk_scale_soma
        # Axonal Kv conductance density:
        self.gk_axon = gk_axon
        # Somatic Kv conductance density:
        self.gk_soma = gk_soma
        # Distal axonal Kv conductance density:
        self.gk_distal_axon = gk_distal_axon 

def fgaussColqu(x, f_c):
    """Eq. 5 from Colquhoun & Sigworth, p. 486 of the blue book"""
    # np.log(2.0)/2.0 = 0.34657359028
    return np.exp(-0.34657359028 * (x/f_c)**2)

def gaussian_filter(x, f_c, dt):
    """
    Gaussian filter according to Colquhoun & Sigworth (blue book).
    
    Arguments:
    x   -- Input data (1D numpy array)
    f_c -- Cutoff frequency in kHz (-3 dB)
    dt  -- sampling interval in ms

    Returns:
    x convolved with a Gaussian filter kernel.
    """

    xf = fftpack.rfft(x)
    # returns an array of frequencies corresponding to the indices of x:
    f = fftpack.rfftfreq(len(x), dt) 
    xf *= fgaussColqu(f, f_c)
    return fftpack.irfft(xf)

def rec_parent(ret_list, cell):
    sr = ap.h.SectionRef(sec = ret_list[-1])
    for candidate in cell.axo:
        if ap.h.secname(sec=sr.parent) == ap.h.secname(sec=candidate):
            ret_list.append(candidate)
            rec_parent(ret_list, cell)

def find_longest_axon(cell):
    max_dist = 0
    # set distance origin to border of soma:
    ap.h.distance(0,0.0, sec=cell.somaBorderLoc.secRef.sec)
    for axon in cell.axo:
        end_dist = ap.h.distance(1.0, sec=axon)
        if end_dist > max_dist:
            term_axon = axon
            max_dist = end_dist
    # recursively find parents:
    ret_list = [ term_axon, ]
    rec_parent(ret_list, cell)
    return ret_list

def maxRise(ap_wave, left, right):
    """
    returns the maximal slope of rise within the vector $o1 between indices $2 and $3
    (typically: beginning of event ($2) to index of peak ($3))
    adopted from stimfit
    """

    # Maximal rise
    maxRise=abs(ap_wave[right]-ap_wave[right-1])
    i=right-1
    while True:
        diff=abs(ap_wave[i]-ap_wave[i-1])
        if (maxRise < diff):
            maxRise=diff
        i -= 1
       	if (i<=left): break
    return maxRise

def maxDecay(ap_wave, left, right): 
    """
    returns the maximal slope of decay within the vector $o1 between indices $2 and $3
    (typically: index of peak ($2) to end of event ($3))
    adopted from stimfit
    """
    # {local left,right,maxDecay,i,diff
    if (left<0): left=0
    if (left > len(ap_wave)-3): left = len(ap_wave)-3
    if (right<0): right=0
    if (right > len(ap_wave)-1): right=len(ap_wave)-1
    # Maximal decay
    maxDecay=abs(ap_wave[left+1]-ap_wave[left])
    i=left+2
    while True:
        diff=abs(ap_wave[i]-ap_wave[i-1])
        if (maxDecay<diff):
            maxDecay=diff
        i+=1
        if (i>=right): break
    return maxDecay

def t50(ap_wave, peak_index, base, peak):
    """
    Returns the full width at half-maximal amplitude (FWHM) of an event in vector $o1.
    The index of the (estimated) peak of the event should be given as $2, baseline as $3, peak as $4
    adopted from stimfit
    """
    center = peak_index
    ampl = peak-base

    # walk left from center until HMA is reached:
    t50LeftId=center
    while True:
        t50LeftId-=1
        if t50LeftId <= 0:
            break
        if abs(ap_wave[t50LeftId]-base) < abs(0.5 * ampl):
            break

    # walk right:
    t50RightId=center
    while True:
        t50RightId += 1
        if t50RightId >= len(ap_wave):
            break
        if abs(ap_wave[t50RightId]-base)<abs(0.5 * ampl):
            break

    if t50LeftId >= len(ap_wave)-1:
        return 0
    if t50RightId < 0 or t50RightId >= len(ap_wave):
        return 0
    # calculation of real values by linear interpolation: 
    # Left side
    yLong2=ap_wave[t50LeftId+1]
    yLong1=ap_wave[t50LeftId]

    if yLong2-yLong1 != 0:
        t50LeftReal=(t50LeftId+abs((0.5*ampl-(yLong1-base))/(yLong2-yLong1)))
    else:
        t50LeftReal=t50LeftId

    # Right side
    yLong2=ap_wave[t50RightId]
    yLong1=ap_wave[t50RightId-1]

    if yLong2-yLong1 != 0:
        t50RightReal=t50RightId-abs((0.5*ampl-(yLong2-base))/abs(yLong2-yLong1))
    else:
        t50RightReal=t50RightId

    return t50RightReal-t50LeftReal

def analyse_ap(ap_wave, dt, f_c=0, resample=1):
    """
    Analyse an AP; return maximal rise, decay as a tuple
    f_c:      cutoff frequency for Gaussian filter 
              f_c == 0 means no filtering
    resample: factor for new sampling interval for resampling; i.e. dt_new = dt*resampling
              uses B-spline interpolation at present
              resampling==1 means no resampling
    """
    # local rise_axon, rise_soma, decay_axon, decay_soma, peak_index_axon, peak_index_soma, error, endInj, startInj
    if f_c > 0:
        pad1 = np.array([ap_wave[0] for i in range(1000)])
        pad2 = np.array([ap_wave[-1] for i in range(1000)])
        filtered = gaussian_filter(np.concatenate((pad1, ap_wave, pad2)), f_c, dt)[1000:1000+len(ap_wave)]
    else:
        filtered = ap_wave.copy()

    if resample != 1:
        # flin = interpolate.interp1d(np.arange(0,len(filtered),dtype=np.double)*dt, filtered)
        fspline = fitpack.splrep(np.arange(0,len(filtered),dtype=np.double)*dt, filtered)
        dt_ip = dt*resample
        y_ip = fitpack.splev(np.arange(0,len(filtered)/resample)*dt_ip, fspline)
    else:
        y_ip = filtered.copy()
        dt_ip = dt

    peak_index= y_ip.argmax()
    rise = maxRise(y_ip, 1, peak_index)/dt_ip
    decay = maxDecay(y_ip, peak_index, len(y_ip)-2)/dt_ip
    fwhm = t50(y_ip, peak_index, y_ip[0], y_ip.max()) * dt_ip
    return (rise, decay, fwhm, y_ip.max()-y_ip[0])

def whereis(wave, value):
    """
    returns the interpolated index of a vector ($o1) where
    $2 is found for the first time
    """
    # {local n,retIndex,fromtop,frombottom,m,c,x0,x1,y0,y1
    retIndex=0
    fromtop=False

    # coming from top or bottom?
    if (wave[0] > value):
	fromtop=True
    for n in range(len(wave)):
	if fromtop:
	    if (wave[n]<value):
		retIndex=n
		break
        elif (wave[n]>value):
            retIndex=n
            break
    if retIndex==0:
	sys.stderr.write("Value not found in whereis()\n")
	return 0

    # linear interpolation:
    x0=retIndex-1
    x1=retIndex
    y0=wave[x0]
    y1=wave[x1]
    m=(y1-y0)/(x1-x0)
    c=y0-m*x0
    return (value-c)/m

def init_h(bleb=False, dt=0.005, axon_seg=11, axon_lim=6, T=24.0, v_init=-80.0):
    """
    Initialize NEURON

    Arguments:
    bleb --     If True, cut the axon and add a spherical bleb
    dt --       Integration time step
    axon_seg -- Multiplication factor to increase nseg in the 
                proximal axon
    axon_lim -- Axonal section index up to which axon_seg is applied

    Returns:
    Cell object
    """

    ap.h("""load_file("stdrun.hoc")
            load_file("./hoc/config.hoc")
         """)
    
    ap.h.celsius = T
    ap.h.dt = dt
    ap.h.steps_per_ms = 1.0/ap.h.dt
    ap.h.v_init = v_init
    
    ap.h("""load_file("./hoc/membrane.hoc")""")

    arg = int(bleb)
    cell = ap.h.cell_10(arg) # 0 = no bleb, 1 = with bleb

    # Increase nseg in proximal axon:
    for (n_a,axon) in enumerate(cell.axo):
        if (n_a < axon_lim):
            axon.nseg *= axon_seg

    return cell

def fsigm(x, amp, center, slope):
    """
    Sigmoidal function
    """
    
    return amp-amp/(1.0+np.exp((x-center)/slope))
