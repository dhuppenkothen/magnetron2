import os
import shutil
import subprocess
import time as tsys
import numpy as np
import copy
import glob
import argparse

from dnest4.classic import logsumexp, logdiffexp

def rewrite_main(filename, dnest_dir = "./"):

    mfile = open(dnest_dir+"main.cpp", "r")
    mdata = mfile.readlines()
    mfile.close()

    ## replace filename in appropriate line:
    mdata[-6] = '\tData::get_instance().load("%s");\n'%filename

    mfile.close()

    mwrite_file = open(dnest_dir+"main.cpp.tmp", "w")

    for l in mdata:
        mwrite_file.write(l)

    mwrite_file.close()

    shutil.move(dnest_dir+"main.cpp.tmp", dnest_dir+"main.cpp")

    return


def rewrite_options(nlevels=1000, dnest_dir="./"):

    mfile = open(dnest_dir+"OPTIONS", "r")
    mdata = mfile.readlines()
    mfile.close()

    print(mdata[6])
    mdata[6] = '%i\t# maximum number of levels\n'%nlevels
    print(mdata[6])

    mwrite_file = open(dnest_dir+"OPTIONS.tmp", "w")

    for l in mdata:
        mwrite_file.write(l)

    mwrite_file.close()

    shutil.move(dnest_dir+"OPTIONS.tmp", dnest_dir+"OPTIONS")

    return


def rewrite_display(filename, dnest_dir="./"):

    mfile = open(dnest_dir+"display.py", "r")
    mdata = mfile.readlines()
    mfile.close()

    mdata[2] = "data = np.loadtxt('%s')\n"%filename

    mwrite_file = open(dnest_dir+"display.py.tmp", "w")

    for l in mdata:
        mwrite_file.write(l)

    mwrite_file.close()

    shutil.move(dnest_dir+"display.py.tmp", dnest_dir+"display.py")

    return


def remake_model(dnest_dir="./"):

    tstart = tsys.clock()
    subprocess.call(["make", "-C", dnest_dir])
    tsys.sleep(15)
    tend = tsys.clock()

    return


def extract_nlevels(filename):

    fsplit = filename.split("_")

    sdata = np.loadtxt("%s_%s_samples.txt"%(fsplit[0], fsplit[1]))

    nlevels = np.shape(sdata)[0]

    return nlevels




def postprocess_new(temperature=1., numResampleLogX=1, plot=False,
                    save_posterior=False):

    cut = 0

    try:
        levels = np.atleast_2d(np.loadtxt("levels.txt"))
        sample_info = np.atleast_2d(np.loadtxt("sample_info.txt"))
        sample = np.atleast_2d(np.loadtxt("sample.txt"))
    except IOError:
        return None, None

    sample = sample[int(cut*sample.shape[0]):, :]
    sample_info = sample_info[int(cut*sample_info.shape[0]):, :]

    if sample.shape[0] != sample_info.shape[0]:
        print('# Size mismatch. Truncating...')
        lowest = np.min([sample.shape[0], sample_info.shape[0]])
        sample = sample[0:lowest, :]
        sample_info = sample_info[0:lowest, :]

    # Convert to lists of tuples
    logl_levels = [(levels[i,1], levels[i, 2]) for i in range(0,
                                                              levels.shape[0])] # logl, tiebreaker
    logl_samples = [(sample_info[i, 1], sample_info[i, 2], i) for
                    i in range(0, sample.shape[0])] # logl, tiebreaker, id

    logx_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    logp_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    logP_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    P_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    logz_estimates = np.zeros((numResampleLogX, 1))
    H_estimates = np.zeros((numResampleLogX, 1))

    # Find sandwiching level for each sample
    sandwich = sample_info[:,0].copy().astype('int')
    for i in range(0, sample.shape[0]):
        while sandwich[i] < levels.shape[0]-1 and \
                        logl_samples[i] > logl_levels[sandwich[i] + 1]:
            sandwich[i] += 1


    for z in range(0, numResampleLogX):
        # For each level
        for i in range(0, levels.shape[0]):
            # Find the samples sandwiched by this level
            which = np.nonzero(sandwich == i)[0]
            logl_samples_thisLevel = [] # (logl, tieBreaker, ID)
            for j in range(0, len(which)):
                logl_samples_thisLevel.append(copy.deepcopy(logl_samples[which[j]]))
            logl_samples_thisLevel = sorted(logl_samples_thisLevel)
            N = len(logl_samples_thisLevel)

            # Generate intermediate logx values
            logx_max = levels[i, 0]
            if i == levels.shape[0]-1:
                logx_min = -1E300
            else:
                logx_min = levels[i+1, 0]
            Umin = np.exp(logx_min - logx_max)

            if N == 0 or numResampleLogX > 1:
                U = Umin + (1. - Umin)*np.random.rand(len(which))
            else:
                U = Umin + (1. - Umin)*np.linspace(1./(N+1), 1. - 1./(N+1), N)
            logx_samples_thisLevel = np.sort(logx_max + np.log(U))[::-1]

            for j in range(0, which.size):
                logx_samples[logl_samples_thisLevel[j][2]][z] = \
                    logx_samples_thisLevel[j]

                if j != which.size - 1:
                    left = logx_samples_thisLevel[j+1]
                elif i == levels.shape[0]-1:
                    left = -1E300
                else:
                    left = levels[i+1][0]

                if j != 0:
                    right = logx_samples_thisLevel[j-1]
                else:
                    right = levels[i][0]

                logp_samples[logl_samples_thisLevel[j][2]][z] = \
                    np.log(0.5) + logdiffexp(right, left)

        logl = sample_info[:,1]/temperature

        logp_samples[:,z] = logp_samples[:,z] - \
                            logsumexp(logp_samples[:,z])
        logP_samples[:,z] = logp_samples[:,z] + logl
        logz_estimates[z] = logsumexp(logP_samples[:,z])
        logP_samples[:,z] -= logz_estimates[z]
        P_samples[:,z] = np.exp(logP_samples[:,z])
        H_estimates[z] = -logz_estimates[z] + np.sum(P_samples[:,z]*logl)

    if save_posterior:

        P_samples = np.mean(P_samples, 1)
        P_samples = P_samples/np.sum(P_samples)
        logz_estimate = np.mean(logz_estimates)
        logz_error = np.std(logz_estimates)
        H_estimate = np.mean(H_estimates)
        H_error = np.std(H_estimates)
        ESS = np.exp(-np.sum(P_samples*np.log(P_samples+1E-300)))

        print("log(Z) = " + str(logz_estimate) + " +- " + str(logz_error))
        print("Information = " + str(H_estimate) + " +- " +
              str(H_error) + " nats.")
        print("Effective sample size = " + str(ESS))

        # Resample to uniform weight
        N = int(ESS)
        posterior_sample = np.zeros((N, sample.shape[1]))
        w = P_samples
        w = w/np.max(w)
        np.savetxt('weights.txt', w) # Save weights
        for i in range(0, N):
            while True:
                which = np.random.randint(sample.shape[0])
                if np.random.rand() <= w[which]:
                    break
            posterior_sample[i,:] = sample[which,:]
        np.savetxt("posterior_sample.txt", posterior_sample)

    return logx_samples, P_samples


def find_weights(p_samples):

    print("max(p_samples): %f" %np.max(p_samples[-10:]))

    # NOTE: logx_samples runs from 0 to -120, but I'm interested
    # in the values of p_samples near the
    # smallest values of X, so I need to look at the end of the list
    if np.max(p_samples[-10:]) < 1.0e-5:
        print("Returning True")
        return True
    else:
        print("Returning False")
        return False


def run_burst(filename, dnest_dir = "./", levelfilename=None, nsims=100,
              ncores=8, min_levels=100):

    assert isinstance(ncores, int), "Number of cores must be an integer number!"

    ### first run: set levels to 200
    print("Rewriting DNest run file")
    rewrite_main(filename, dnest_dir)
    # rewrite_options(nlevels=1000, dnest_dir=dnest_dir)
    rewrite_options(nlevels=1000, dnest_dir=dnest_dir)
    remake_model(dnest_dir)

    fdir = filename.split("/")
    fname = fdir[-1]
    fdir = filename[:-len(fname)]
    print("directory: %s" %fdir)
    print("filename: %s" %fname)

    fsplit = fname.split(".")
    froot = "%s%s" %(fdir, fsplit[0])
    print("froot: " + str(froot))


    ### printing DNest display script
    rewrite_display(filename, dnest_dir)


    print("First run of DNest: Find number of levels")
    print("I am running on %i cores."%ncores)
    ## run DNest
    dnest_process = subprocess.Popen(["./main", "-t", "%i"%ncores])

    endflag = False
    while endflag is False:
        try:
            tsys.sleep(360)
            levels = np.loadtxt("%slevels.txt" %dnest_dir)
            if len(levels)-1 <= min_levels:
                endflag = False
            else:
                logx_samples, p_samples = postprocess_new(save_posterior=False)
                if p_samples is None:
                    endflag = False
                else:
                    endflag = find_weights(p_samples)
                    print("Endflag: " + str(endflag))


        except KeyboardInterrupt:
            break


    print("endflag: " + str(endflag))

    dnest_process.kill()
    dnest_data = np.loadtxt("%slevels.txt" %dnest_dir)
    nlevels = len(dnest_data)+100

    nsamples = len(np.loadtxt("%ssample.txt"%dnest_dir))

    ### save levels to file
    if not levelfilename is None:
        levelfile = open(levelfilename, "a")
        levelfile.write("%s \t %i \n" %(filename, nlevels))
        levelfile.close()

    rewrite_options(nlevels=nlevels, dnest_dir=dnest_dir)
    remake_model(dnest_dir)

    dnest_process = subprocess.Popen(["./main", "-t", "%i"%ncores])

    endflag = False
    while endflag is False:
        try:
            tsys.sleep(300)
            logx_samples, p_samples = postprocess_new(save_posterior=True)
            post_samples = np.loadtxt("%sposterior_sample.txt"%dnest_dir)
            print("samples file: %ssample.txt" %dnest_dir)
            print("Endflag: " + str(endflag))
            if len(post_samples) >= nsims and len(np.shape(post_samples)) > 1:
                endflag = True
            else:
                if len(post_samples) <= 10 or len(np.shape(post_samples)) == 1:
                    print("I have made it here!")
                    levels = np.loadtxt("%slevels.txt" % dnest_dir)
                    raw_samples = len(np.loadtxt("%ssample.txt" % dnest_dir))

                    if len(levels) >= nlevels-1 and raw_samples > nsamples+1000:
                        print("I have almost made it to the right spot!")
                        endflag = True
                        broken_file = "%sfailed.txt"%data_dir
                        if os.path.exists(broken_file):
                            append_write = 'a'  # append if already exists
                        else:
                            append_write = 'w'  # make a new file if not
                        with open(broken_file, append_write) as f:
                            f.write(filename + "\n")

                else:
                    endflag = False
        except KeyboardInterrupt:
            break

    print("Endflag: " + str(endflag))

    dnest_process.kill()
     
    logx_samples, p_samples = postprocess_new(save_posterior=True)    

    print("froot: " + str(froot))

    shutil.move("sample.txt", "%s_sample.txt" %froot)
    try:
        shutil.move("posterior_sample.txt", "%s_posterior_sample.txt" %froot)
        shutil.move("levels.txt", "%s_levels.txt" %froot)
        shutil.move("sample_info.txt", "%s_sample_info.txt" %froot)
        shutil.move("weights.txt", "%s_weights.txt" %froot)
    except IOError:
        print("No file posterior_sample.txt")

    return


def run_all_bursts(data_dir="./", dnest_dir="./", levelfilename="test_levels.dat",
                   ncores=8, nsims=100, match_string="*.csv", min_levels=10):

    print("I am in run_all_bursts")
    filenames = glob.glob("%s%s"%(data_dir, match_string))
    print(filenames)

    levelfilename = data_dir+levelfilename
    print("Saving levels in file %s"%levelfilename)

    levelfile = open(levelfilename, "w")
    levelfile.write("# data filename \t number of levels \n")
    levelfile.close()

    for f in filenames:
        print("Running on burst %s" %f)
        run_burst(f, dnest_dir=dnest_dir, levelfilename=levelfilename,
                  ncores=ncores, nsims=nsims, min_levels=min_levels)

    return


def main():
    print("I am in main")
    run_all_bursts(data_dir, dnest_dir, levelfilename, ncores=ncores,
                   nsims=nsamples, match_string=match_string,
                   min_levels=min_levels)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running DNest on a number "
                                                 "of bursts")

    parser.add_argument("-d", "--datadir", action="store", required=False,
                        dest="data_dir",
                        default="./", help="Specify directory with data "
                                           "files (default: current directory)")
    parser.add_argument("-n", "--dnestdir", action="store", required=False,
                        dest="dnest_dir", default="./",
                        help="Specify directory with DNest model "
                             "implementation (default: current directory")

    parser.add_argument("-f", "--filename", action="store", required=False,
                        dest="filename", default="test_levels.dat",
                        help="Define filename for file that saves the number "
                             "of levels to use")

    parser.add_argument("-c", "--cores", action="store", required=False, type=int,
                        dest="ncores", default=8, help="Number of cores "
                                                       "DNest4 should use.")

    parser.add_argument("--samples", action="store", required=False, type=int,
                        dest="nsamples", default=100, help="Numer of posterior "
                                                        "samples to store.")

    parser.add_argument("-s", "--string", action="store", required=False,
                        dest="match_string", default="*.csv",
                        help="The string to use to search for data files to "
                             "run.")

    parser.add_argument("--min-levels", action="store", required=False,
                        dest="min_levels", default=200, type=int,
                        help="The minimum number of levels to run.")

    clargs = parser.parse_args()

    data_dir = clargs.data_dir
    dnest_dir = clargs.dnest_dir
    levelfilename = clargs.filename
    ncores = clargs.ncores
    match_string = clargs.match_string
    nsamples = clargs.nsamples
    min_levels = clargs.min_levels

    main()
