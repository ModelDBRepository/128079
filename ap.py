"""
Runs current-clamp simulations using NEURON.
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

from neuron import h

import mech
from ap_utils import *
import stfio_plot

gF_c = 10.0 # filter frequency
gResample = 4.0 # resampling factor (exp. dt / model dt)

def run(cell, settings, ic=None, tstop=5, plot=False, print_results=False):
    """
    Run simulation in cell with the specified settings.
    
    Additional arguments:
    ic -- NEURON IClamp
    tstop -- Simulation duration.
    """

    v_record = {}
    v_record['soma'] = h.Vector()
    v_record['soma'].record(cell.somaLoc.secRef.sec(0.5)._ref_v, 
                            sec=cell.somaLoc.secRef.sec)

    v_record['bleb'] = h.Vector()
    v_record['bleb'].record(
        cell.blebLoc.secRef.sec(cell.blebLoc.loc)._ref_v, 
        sec=cell.blebLoc.secRef.sec)

    # Initialize list with somatic value:
    t50 = [[0,0],]

    start_speed_dist = 100
    end_speed_dist = 500

    # A list of sections from the most distal point of the axon back to the 
    # soma:
    longest_axon = find_longest_axon(cell)
    longest_axon_names = [ ax.name() for ax in longest_axon ]
    start_speed_index, end_speed_index = 0, 0

    # set origin of distance calculations to somatic border:
    h.distance(0, 0.0, sec=cell.somaBorderLoc.secRef.sec)

    v_record['axon'] = []
    for section in cell.axo:
        if section.name() in longest_axon_names:
            for seg in section:
                dist = h.distance(seg.x, sec=section) 
                t50.append([dist, 0])
                if start_speed_index == 0 and t50[-1][0] > start_speed_dist:
                    start_speed_index = len(t50)
                if end_speed_index == 0 and t50[-1][0] > end_speed_dist:
                    end_speed_index = len(t50)
                v_record['axon'].append(h.Vector())
                v_record['axon'][-1].record(section(seg.x)._ref_v, 
                                            sec=section)

    mech.init_mech(cell, settings) 
    mech.init_rates(cell, settings) 

    gna_x, gna_y, gna_peak, gna_mean_prox = mech.init_g(cell, settings)
    gna_mean_prox_special = mech.gna_per_area(cell)
    print gna_mean_prox
    sys.stdout.write("Mean density in point measurements from proximal axon: %f pS/um^2\n" %
                     (gna_mean_prox*10.0))
    sys.stdout.write("Mean total conductance per total membrane area in proximal axon: %f pS/um^2\n" %
                     (gna_mean_prox_special*10.0))

    h.tstop = tstop
    h.init()
    h.run()

    lat0 =  whereis(v_record['soma'], 
                    (v_record['soma'].max()+v_record['soma'].min())/2.0)

    for (i, vec) in enumerate(v_record['axon']):
        t50[i+1][1] = \
            (whereis(vec, (vec.max()+vec.min())/2.0) - lat0)*h.dt*1.0e3
        if t50[i+1][0] > 25 and t50[i+1][0] < 30:
            rise_axon, decay_axon, t50_axon, amp_axon = \
                analyse_ap(np.array(vec), h.dt, gF_c, gResample)
    t50_x = np.array([t50i[0] for t50i in t50])
    t50_y = np.array([t50i[1] for t50i in t50])

    try:
        speed = \
            (t50_x[end_speed_index]-t50_x[start_speed_index])/ \
            (t50_y[end_speed_index]-t50_y[start_speed_index])
        if speed>0:
            sys.stdout.write("Propagation speed: %f m/s\n" % speed)
            init_site = t50_x[t50_y[0:400].argmin()]
            sys.stdout.write("Initiation site: %f um\n" % init_site)
    except:
        speed = 0

    if plot:
        ts_soma = stfio_plot.timeseries(np.array(v_record['soma']), h.dt,
                                        linestyle = '-k', linewidth=2.0)
        ts_bleb = stfio_plot.timeseries(np.array(v_record['bleb']), h.dt,
                                        linestyle = '-r', linewidth=2.0)
        pulse_dt = h.dt
        pulse_t = np.arange(0, ts_soma.duration(), pulse_dt)
        pulse = np.zeros((len(pulse_t)))
        pulse[ic.delay/pulse_dt:(ic.delay+ic.dur)/pulse_dt] = ic.amp
        ts_pulse = stfio_plot.timeseries(pulse, pulse_dt, linewidth=2.0,
                                         yunits="nA")

        stfio_plot.plot_traces([ts_soma, ts_bleb], pulses=[ts_pulse,])
    
    if print_results:
        np_soma = np.array(v_record['soma'])
        rise_soma, decay_soma, t50_soma, amp_soma = \
            analyse_ap(np_soma, h.dt, gF_c, gResample)
        np_bleb = np.array(v_record['bleb'])
        rise_bleb, decay_bleb, t50_bleb, amp_bleb = \
            analyse_ap(np_bleb, h.dt, gF_c, gResample)
        sys.stdout.write(
            "Max. rate of rise (soma): %f V/s\n" % rise_soma)
        sys.stdout.write(
            "Max. rate of decay (soma): %f V/s\n" % decay_soma)
        sys.stdout.write(
            "FWHM (soma): %f ms\n" % t50_soma)
        sys.stdout.write(
            "Max. rate of rise (bleb): %f V/s\n" % rise_bleb)
        sys.stdout.write(
            "Max. rate of decay (bleb): %f V/s\n" % decay_bleb)
        sys.stdout.write(
            "FWHM (bleb): %f ms\n" % t50_bleb)

    # Clean up in case we're repeatedly running from an interactive shell
    for vec in v_record['axon']:
        vec.play_remove()
        del vec
    v_record['soma'].play_remove()
    del v_record['soma']
    v_record['bleb'].play_remove()
    del v_record['bleb']

    return t50_x, t50_y

def plot_ap(gating = '8s', bleb=True, plot=True, print_results=False):

    cell = init_h(bleb = bleb)

    ic = h.IClamp(cell.somaLoc.secRef.sec(0.5))
    ic.delay = 0.5
    ic.amp = 2.0
    ic.dur = 0.5

    setting = \
        RunSettings(uniform_g = False, uniform_kin = False,
                    gating = gating)
    t50_x, t50_y = \
        run(cell, setting, ic=ic, plot=plot, print_results=print_results)
    del cell
    del ic

if __name__=="__main__":
    plot_ap(gating='8s', bleb=True, plot=False, print_results=True)
    plot_ap(gating='8s', bleb=False, plot=True, print_results=False)
    stfio_plot.plt.show()
