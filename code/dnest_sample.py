
import numpy as np
import glob
import shutil

#import burstmodel
#import parameters
#import word
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.cm as cm
import scipy.stats
import scipy.optimize



def move_posteriors():
    """
    Random function that moves files already run through DNest into a folder ./finished/.
    That folder had better exist!

    """
    pfiles = glob.glob("*posterior*")
    for p in pfiles:
        fsplit = p.split("_")
        froot = "%s_%s*"%(fsplit[0], fsplit[1])
        nfiles = glob.glob(froot)
        for n in nfiles:
            shutil.move(n, "./finished/")


    return


def plot_posterior_lightcurve(namestr, datadir="./", nsims=10,
                              nburst_column=9):
    """
    Plot the data and model light curves from the posterior sample, in one
    subplot, and the distribution of spikes in the other.

    Parameters
    ----------
    namestr : string
        Root of the data file. namestr+.txt should make the whole data file.
        Note that the posterior sample file must have the form
        `namestr_posterior_sample.txt`.

    datadir : string, optional, default "./"
        Directory where the data file and posterior samples are located.
        Default is current working directory.

    nsims : int, optional, default 10
        The number of simulations to plot along with the data

    nburst_column : int, optional, default 9
        The column in the `posterior_sample.txt` file that contains the
        distribution of the number of spikes.
    """

    data = np.loadtxt("%s%s.txt"%(datadir, namestr))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15,5))

    ax.errorbar(data[:,0], data[:,1], yerr=data[:,2], lw=2, fmt="o",
                color="black", linestyle="steps-mid")

    sample = np.atleast_2d(np.loadtxt("%s%s_posterior_sample.txt"%(datadir,
                                                                   namestr)))


    ind = np.random.choice(np.arange(len(sample)), replace=False, size=nsims)

    for i in ind:
        ax.plot(data[:,0], sample[i,-data.shape[0]:], lw=1, c="red",
                alpha=0.4, zorder=10)

    ax.set_xlabel("MJD [days]", fontsize=16)
    ax.set_ylabel("Counts per bin", fontsize=16)
    ax.set_xlim([data[0,0], data[-1,0]])

    ax.set_title("%s"%namestr, fontsize=16)

    nbursts = sample[:, nburst_column]

    ax2.hist(nbursts, bins=30, range=[np.min(nbursts), np.max(nbursts)],
             histtype='stepfilled')

    ax2.set_xlabel("Number of spikes per burst", fontsize=16)
    ax2.set_ylabel("N(samples)", fontsize=16)

    plt.savefig("%s%s_lc.png"%(datadir, namestr), format="png")
    plt.close()

    return


def read_dnest_results(namestr, datadir="./", par_ind=10):

    """
    Read output from magnetron2 run and return in a format more
    friendly to post-processing.

    Parameters
    ----------
    namestr : string
        Root of the data file. namestr+.txt should make the whole data file.
        Note that the posterior sample file must have the form
        `namestr_posterior_sample.txt`.

    datadir : string, optional, default "./"
        Directory where the data file and posterior samples are located.
        Default is current working directory.

    par_ind : int, optional, default 10
        Column index where the parameters for each flare component start
    """
    samples = np.loadtxt("%s%s_posterior_sample.txt"%(datadir, namestr))

    niter = samples.shape[0]
    print("There are %i samples in the posterior sample file."%niter)

    # background parameter
    bkg = samples[:,0]

    ou_noise_l = samples[:,1]
    ou_noise_sigma = samples[:,2]

    # dimensions of parameter space of individual  model components
    burst_dims =  samples[:,3]
    burst_dims = list(set(burst_dims))[0]

    # total number of model components permissible in the model
    compmax = samples[:,4]
    compmax = int(list(set(compmax))[0])

    # hyper-parameter (mean) of the exponential distribution used
    # as prior for the spike amplitudes
    # NOTE: IN LINEAR SPACE, NOT LOG
    hyper_mean_amplitude = samples[:,5]

    # hyper-parameter (mean) for the exponential distribution used
    # as prior for the spike rise time
    # NOTE: IN LINEAR SPACE, NOT LOG
    hyper_mean_risetime = samples[:,6]

    # hyper-parameters for the lower and upper limits of the uniform
    # distribution osed as a prior for the skew
    hyper_mean_skew = samples[:,7]
    hyper_width_skew = samples[:,8]

    # distribution over number of model components
    nbursts = samples[:, 9]

    # peak positions for all model components, some will be zero
    pos_all = np.array(samples[:, par_ind:par_ind+compmax])

    # amplitudes for all model components, some will be zero
    amp_all = samples[:, par_ind+compmax:par_ind+2*compmax]

    # rise times for all model components, some will be zero
    scale_all = samples[:, par_ind+2*compmax:par_ind+3*compmax]

    # skew parameters for all model components, some will be zero
    skew_all = samples[:, par_ind+3*compmax:par_ind+4*compmax]

    # pull out the ones that are not zero
    paras_real = []


    for p,a,sc,sk in zip(pos_all, amp_all, scale_all, skew_all):
        paras_real.append([(pos,amp,scale,skew) for pos,amp,scale,skew
                           in zip(p,a,sc,sk) if pos != 0.0])


    sample_dict = {"bkg":bkg, "cdim":burst_dims, "nbursts":nbursts,
                   "ou_noise_l": ou_noise_l, "ou_noise_sigma": ou_noise_sigma,
                   "cmax":compmax,
                   "parameters":paras_real,
                   "hyper_mean_amp":hyper_mean_amplitude,
                   "hyper_mean_rise": hyper_mean_risetime,
                   "hyper_mean_skew": hyper_mean_skew,
                   "hyper_width_skew": hyper_width_skew}

    return sample_dict

def plot_hyper_parameters(namestr, datadir="./", ouprocess=True, par_ind=10):

    sd = read_dnest_results(namestr, datadir, par_ind=par_ind)

    bkg = sd["bkg"]
    ou_noise_l = sd["ou_noise_l"]
    ou_noise_sigma = sd["ou_noise_sigma"]

    amp_mean = sd["hyper_mean_amp"]
    rise_mean = sd["hyper_mean_rise"]

    skew_mean = sd["hyper_mean_skew"]
    skew_width = sd["hyper_width_skew"]

    return

def plot_flare_parameters(namestr, datadir="./", par_ind=10):

    data = np.loadtxt("%s%s.txt"%(datadir, namestr))

    sd = read_dnest_results(namestr, datadir, par_ind=par_ind)
    pars = sd["parameters"]
    pos_all, amp_all, scale_all, skew_all = [], [], [], []

    for pa in pars:
        for p in pa:
            pos_all.append(p[0])
            amp_all.append(p[1])
            scale_all.append(p[2])
            skew_all.append(p[3])

    pos_all = np.array(pos_all)
    amp_all = np.array(amp_all)
    scale_all = np.array(scale_all)
    skew_all = np.array(skew_all)

    # compute fall time
    fall_all = scale_all * skew_all


    # plot rise time, fall time, skew and ratio
    fig, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, figsize=(12, 12))
    ax11.hist(np.log10(scale_all), bins=50)
    ax11.set_xlabel("log10(rise time) [days]")
    ax11.set_ylabel("Number of simulated flares")
    ax11.set_title(namestr)

    ax12.hist(np.log10(fall_all), bins=50)
    ax12.set_xlabel("log10(fall time) [days]")
    ax12.set_ylabel("Number of simulated flares")

    ax13.hist(skew_all, bins=50)
    ax13.set_xlabel("fall time/rise time")
    ax13.set_ylabel("Number of simulated flares")

    ax14.hist(np.log10(amp_all), bins=50)
    ax14.set_xlabel("log10(flare amplitude) [Jy]")
    ax14.set_ylabel("Number of simulated flares")

    plt.tight_layout()
    plt.savefig("%s%s_flare_pars.png"%(datadir, namestr), format="png")
    plt.close()

    # plot light curve with position rug plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(8,6))
    ax2.errorbar(data[:,0], data[:,1], yerr=data[:,2], fmt="o",
                 color="black", lw=2)

    ax2.set_xlim(data[0,0], data[-1, 0])
    sns.rugplot(pos_all, ax=ax2, color="red", lw=2, alpha=0.5)
    ax2.set_title(namestr)

    plt.savefig("%s%s_positions.png"%(datadir, namestr), format="png")
    plt.close()

    return sd


def flare(times, t0, amplitude, scale, skew):
    fall = scale*skew
    y = np.zeros_like(times)
    y[t0 <= times] = amplitude*np.exp((t0 - times[t0 <= times])/fall)
    y[times < t0] = amplitude*np.exp((times[times < t0] - t0)/scale)

    return y

def plot_components(namestr, datadir="./", nsamples=1, idx=None, par_ind=10):

    data = np.loadtxt("%s%s.txt"%(datadir, namestr))

    sd = read_dnest_results(namestr, datadir, par_ind=par_ind)
    pars = sd["parameters"]
    bkg = sd["bkg"]

    nsamples = len(pars)

    # if no index specifically given, pick one at random
    if idx is None:
        idx = int(np.random.choice(np.arange(nsamples)))

    print("index: " + str(idx))
    samp = pars[idx]
    bb = bkg[idx]

    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    ax.errorbar(data[:,0], data[:,1], yerr=data[:,2], fmt="o", color="black",
                lw=2)

    for s in samp:
        flare_model = flare(data[:,0], s[0], s[1], s[2], s[3])
        ax.plot(data[:,0], flare_model + bb, lw=2, color="red")

    ax.plot(data[:,0], np.ones_like(data[:,0])*bb, lw=3, linestyle="dashed",
            color="blue")

    return fig, ax

def plot_doppler_factors():
    pass

def plot_fastest_risetime():
    pass