import nest
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pn

flattenlist = lambda lst: [item for sublist in lst for item in sublist] # this is a method to 'flatten' a list; i.e. to
# extract a sublist from a list

def uglyReshape(array, num_procs, N):
    '''
    Reshape output array from multimeter with multiprocessing
    to (N_neurons, N_timesteps)

    parameters
    array : ndarray
        array to be reshaped
    num_procs : int
        total_num_virtual_procs/number of multiprocessing threads
    N : int
        number of neurons
    '''
    return array.reshape((num_procs, -1))[::-1].flatten('F').reshape((-1, N))

# Neuron parameters from Destexhe

# aeIF parameters (shared)
S = 20000.   # um2, membrane surface area
cellParams = dict(
    C_m = 1. * S * 1E6 * 1E-8,   # uF/cm2 * um2 * 1E6 pF/uF * 1E-8 cm2/um2 -> pF
    g_L = 0.05 * S * 1E6 * 1E-8, # mS/cm2 * um2 * 1E6 nS/mS * 1E-8 cm2/um2 -> nS
    E_L = -60.,     # mV
    V_reset = -60., # mV, reset potential
    V_m = -60.,     # mV, initial potential
    Delta_T = 2.5,  # mV, steepness parameter
    V_th = -50.,    # mV, spike threshold
    t_ref = 2.5,    # ms, refractory period
    tau_w = 600.,   # ms, adaptation time constant
    a = 0.001*1E3,  # uS * 1E3 nS/uS -> nS, subthreshold adaptation
    b = 0.04*1E3,   # nA *1E3 pA/nA -> pA, spike-threshold adaptation
)

# Parameters
# Synapse parameters
tau_syn_ex = 5.  # ms, synaptic time constant for exc. connections
tau_syn_in = 10.  # ms, inh. time constant
ge = 6.  # nS, exc. synaptic conductance
gi = -67.  # nS, inh. syn. cond.
E_ex = 0.  # mV, exc. syn. reversal potential
E_in = -80.  # mV, inh. syn. rev. pot.

# To initiate activity, a number of randomly-chosen neurons
# (from 2% to 10% of the network) were stimulated by random
# excitatory inputs during the first 50 ms of the simulation.
# The mean frequency of this random activity was high enough (200–400 Hz)
# to evoke random firing in the recipient neurons. In cases
# where self- sustained activity appeared to be unstable,
# different parameters of this initial stimulation were tested.
# It is important to note that after this initial period of 50 ms,
# no input was given to the network and thus the activity states
# described here are self-sustained with no external input or added noise.
noiseParams = dict(start=0., stop=10000000., rate=50.)  # ms, ms, Hz
p_conn_noise = 0.05  # connection probability of 5%

# Population sizes
N = dict(
    TC=100,
    RE=100,
    PYplus=800,
    PYmin=800,
    INplus=200,
    INmin=200,
)

# AdEX parameters (neuron specific)
popParams = dict(
    PYplus=dict(
        a=0.001 * 1E3,  # uS * 1E3 nS/uS -> nS
        b=0.04 * 1E3,  # nA * 1E3 pA/nA -> pA
        tau_syn_ex=tau_syn_ex,  # ms
        tau_syn_in=tau_syn_in,  # ms
        E_ex=E_ex,  # mV
        E_in=E_in  # mV
    ),

    PYmin=dict(
        a=0.001 * 1E3,  # uS * 1E3 nS/uS -> nS
        b=0.04 * 1E3,  # nA * 1E3 pA/nA -> pA
        tau_syn_ex=tau_syn_ex,  # ms
        tau_syn_in=tau_syn_in,  # ms
        E_ex=E_ex,  # mV
        E_in=E_in  # mV
    ),
    INplus=dict(
        a=0.001 * 1E3,  # uS * 1E3 nS/uS -> nS
        b=0.0 * 1E3,  # nA * 1E3 pA/nA -> pA
        tau_syn_ex=tau_syn_ex,  # ms
        tau_syn_in=tau_syn_in,  # ms
        E_ex=E_ex,  # mV
        E_in=E_in  # mV
    ),
    INmin=dict(
        a=0.001 * 1E3,  # uS * 1E3 nS/uS -> nS
        b=0.0 * 1E3,  # nA * 1E3 pA/nA -> pA
        tau_syn_ex=tau_syn_ex,  # ms
        tau_syn_in=tau_syn_in,  # ms
        E_ex=E_ex,  # mV
        E_in=E_in  # mV
    ),
    TC=dict(
        a=0.04 * 1E3,  # uS * 1E3 nS/uS -> nS
        b=0. * 1E3,  # nA * 1E3 pA/nA -> pA
        tau_syn_ex=tau_syn_ex,  # ms
        tau_syn_in=tau_syn_in,  # ms
        E_ex=E_ex,  # mV
        E_in=E_in  # mV
    ),
    RE=dict(
        a=0.03 * 1E3,  # uS * 1E3 nS/uS -> nS
        b=0.08 * 1E3,  # nA * 1E3 pA/nA -> pA
        tau_syn_ex=tau_syn_ex,  # ms
        tau_syn_in=tau_syn_in,  # ms
        E_ex=E_ex,  # mV
        E_in=E_in  # mV
    ),
)

# Connection probabilites and parameters
#    pre   post  prob  g
delay = nest.GetKernelStatus(
    'resolution')  # Destexhe do not consider delay,so here it is instant (with the next simulation step)
C = [
    ['PYplus', 'PYplus', 0.035, ge],
    ['PYplus', 'PYmin', 0.005, ge],
    ['PYmin', 'PYplus', 0.005, ge],
    ['PYmin', 'PYmin', 0.035, ge],
    ['PYplus', 'INplus', 0.035, ge],
    ['PYplus', 'INmin', 0.005, ge],
    ['PYmin', 'INmin', 0.035, ge],
    ['PYmin', 'INplus', 0.005, ge],
    ['PYplus', 'TC', 0.035, ge],
    ['PYmin', 'TC', 0.005, ge],
    ['PYplus', 'RE', 0.035, ge],
    ['PYmin', 'RE', 0.005, ge],
    ['INplus', 'PYplus', 0.035, gi],
    ['INmin', 'PYplus', 0.005, gi],
    ['INplus', 'PYmin', 0.005, gi],
    ['INmin', 'PYmin', 0.035, gi],
    ['INplus', 'INplus', 0.035, gi],
    ['INplus', 'INmin', 0.005, gi],
    ['INmin', 'INplus', 0.005, gi],
    ['INmin', 'INmin', 0.035, gi],
    ['TC', 'RE', 0.02, ge],
    ['TC', 'PYplus', 0.035, ge],
    ['TC', 'PYmin', 0.005, ge],
    ['TC', 'INplus', 0.035, ge],
    ['TC', 'INmin', 0.005, ge],
    ['RE', 'TC', 0.08, gi],
    ['RE', 'RE', 0.08, gi],
]

# Reset NEST
nest.ResetKernel()
nest.SetKernelStatus(dict(total_num_virtual_procs=16))
##nest.SetKernelStatus(dict(total_num_virtual_procs=4))

# create populations
pops = dict()
for popName, popSize in N.items():
    params = popParams[popName].copy()
    params.update(cellParams)
    params.update(dict(V_m = cellParams['E_L'])) ## why is this necessary???
    pops.update({popName : nest.Create('aeif_cond_exp', popSize, params)})

# external noise
noise = nest.Create('poisson_generator', params=noiseParams)

# spike detectors
spikeDetectors = dict()
for popName in N.keys():
    spikeDetectors.update({popName : nest.Create('spike_detector')})

# voltmeter
voltMeters = dict()
for popName in N.keys():
    voltMeters.update({popName : nest.Create('voltmeter',
                                             params=dict(interval=1.0))})

# connect the network
for pre, post, p, g in C:
    nest.Connect(pops[pre], pops[post],
                 conn_spec=dict(rule = 'pairwise_bernoulli', p=p), ## a pairwise bernoulli connect two populations
                 ## with a specific probability p
                 syn_spec=dict(model='static_synapse', weight=g, delay=delay)
                )

# spike detectors
for popName, SD in spikeDetectors.items():
    nest.Connect(pops[popName], SD)

# connect voltmeters
for popName, VM in voltMeters.items():
    nest.Connect(VM, (pops[popName][0],))

# connect noise
nest.Connect(noise, flattenlist(pops.values()), ## this command I do not understand
             conn_spec=dict(rule='pairwise_bernoulli', p=p_conn_noise),
             syn_spec=dict(model='static_synapse', weight=ge, delay=nest.GetKernelStatus('resolution')))

## LFP recording
# Start by recording the synaptic conductances.

## By default, multimeters record values once per ms. Set the parameter /interval to change this.
# Reset NEST

nest.CopyModel('multimeter', 'RecordingNode',
               params={'interval': 1.0,
                       'record_from': ['g_ex', 'g_in', 'V_m'],
                       'record_to': ['memory'],
                       'withgid': True,
                       'withtime': True, })

multimeter_plus = nest.Create('RecordingNode')
multimeter_min = nest.Create('RecordingNode')

# connect
nest.Connect(multimeter_plus, pops['PYplus'])
nest.Connect(multimeter_min, pops['PYmin'])

nest.Simulate(10000.)

## Find g_e, g_in, and V_M and calculate the AMPA and GABA currents based on this
events_plus = nest.GetStatus(multimeter_plus)[0]['events']
events_min = nest.GetStatus(multimeter_min)[0]['events']
I_E_plus = events_plus['g_ex']*(events_plus['V_m']-E_ex) ## application of Ohm's law
I_E_min = events_min['g_ex']*(events_min['V_m']-E_ex)
I_I_plus = events_plus['g_in']*(events_plus['V_m']-E_in)
I_I_min = events_min['g_in']*(events_min['V_m']-E_in)
## shape into matrix with neuron on x axis and time on y axis
I_E_plus = uglyReshape(I_E_plus, 16, N['PYplus'])
I_E_min = uglyReshape(I_E_min, 16, N['PYmin'])
I_I_plus = uglyReshape(I_I_plus, 16, N['PYplus'])
I_I_min = uglyReshape(I_I_min, 16, N['PYmin'])
shifted_I_E_plus = -np.roll(I_E_plus, -60)
shifted_I_E_min = -np.roll(I_E_min, -60)
sumI_E_plus = np.sum(shifted_I_E_plus, axis=1)
sumI_E_min = np.sum(shifted_I_E_min, axis=1)
sumI_E = sumI_E_plus + sumI_E_min
sumI_I_plus = np.sum(I_I_plus, axis=1)
sumI_I_min = np.sum(I_I_min, axis=1)
sumI_I = sumI_I_plus + sumI_I_min
LFPtemp = sumI_E-(1.65*sumI_I)
LFPmean = -91368.9962786
LFP1 = (LFPtemp-LFPmean)/297321.584991
LFP = LFP1[:-60]
print(LFP)
print(LFP.shape)
xaxis = []
for ii, value in enumerate(LFP):
    xaxis.append((ii+1)/1000)
plt.plot(xaxis, LFP)
plt.ylabel('LFP')
plt.xlabel('s')
plt.axis([0.2,10.25, -10, 2])
plt.title('LFP in mosaic model')

## save the LFP as a .dat file
file = open('lfp.dat', 'wb')
file.write(LFP)
file.close()

import nest.raster_plot as raster
import nest.voltage_trace as vtrace
for popName, SD in spikeDetectors.items():
    try:
         raster.from_device(SD)
         fig = pylab.gcf()
         fig.suptitle(popName)
    except:
         continue


## find average firing rate up and down states
dt = 20.
simtime = 10000.
bins = np.arange(200., simtime, dt)
ratedict = dict()
for popName, SD in spikeDetectors.items():
    if popName in ['TC', 'RE', 'INmin', 'PYmin']:
        pass
    else:
        events = nest.GetStatus(SD)[0]['events']
        rate,_ = np.histogram(events['times'], bins)
        rate = ((rate/dt)*1000.)/N[popName]
        plt.plot(bins[:-1], rate, label = popName)
        ratedict.update({popName : rate})

upstatePYplus = []
upstateINplus = []
downstatePYplus = []
downstateINplus = []
for popName, f in ratedict.items():
    if popName is 'PYplus':
        for b in f:
            if b > 10:
                upstatePYplus.append(b)
            if b < 5:
                downstatePYplus.append(b)

for popName, f in ratedict.items():
    if popName is 'INplus':
        for b in f:
            if b > 10:
                upstateINplus.append(b)
            if b < 5:
                downstateINplus.append(b)

meanupIN = np.mean(upstateINplus)
meanupPY = np.mean(upstatePYplus)
meandownIN = np.mean(downstateINplus)
meandownPY = np.mean(downstatePYplus)

print(meanupIN)
print(meanupPY)
print(meandownIN)
print(meandownPY)

plt.show()