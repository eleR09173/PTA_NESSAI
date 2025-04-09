########################### IMPORT ####################################################

# Standard library
import os
# homedir = os.path.abspath("PTA_NESSAI")

# Scientific computing and linear algebra
import numpy as np
import scipy.linalg as sl
from scipy.stats import norm

# libstempo library
import libstempo as LT
import libstempo.toasim as LTsim
from libstempo.toasim import (
    add_efac, 
    add_equad, 
    add_gwb, 
    add_cgw
)

# enterprise and extensions
from enterprise.pulsar import Pulsar
from enterprise.signals.utils import powerlaw, createfourierdesignmatrix_dm
from enterprise.signals import white_signals, gp_signals, parameter
from enterprise.signals.signal_base import PTA
from enterprise_extensions.sampler import JumpProposal

# PTMCMCSampler
# from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

# Plotting libraries
import matplotlib.pyplot as plt
import corner

# Import pickle
import pickle

# NESSAI
from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger

from multiprocessing import Pool #PARALLELIZATION
from nessai.utils.multiprocessing import initialise_pool_variables #PARALLELIZATION


# import argparse to set the following Nessai parameters in every run
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nlive", type=int, required=True)
parser.add_argument("--n_blocks", type=int, required=True)
parser.add_argument("--n_layers", type=int, required=True)
parser.add_argument("--n_neurons", type=int, required=True)
parser.add_argument("--patience", type=int, required=True)
parser.add_argument("--pytorch_threads", type=int, required=True)
parser.add_argument("--n_pool", type=int, required=True)
# min iteration is optional: if not set for the run, Nessai uses its autoiteration criterium
parser.add_argument("--min_iteration", type=int, default=None)



parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--homedir", type=str, default=os.path.abspath("PTA_NESSAI"))


args = parser.parse_args()



########################### PULSAR DEFINITIONS ADAPTED FROM libstempo - SAME FOR SINGLE AND MULTI PULSAR ####################################################



# Constants
DAY = 24 * 3600
YEAR = 365.25 * DAY

def add_efac(psr, efac=1.0, flagid=None, flags=None, seed=None):
    """
    Adapted from libstempo.
    Add nominal TOA errors, multiplied by the `efac` factor.
    Optionally, use a pseudorandom-number-generator seed.
    """
    if seed is not None:
        np.random.seed(seed)

    # Default efacvec
    efacvec = np.ones(psr.nobs)

    if flags is None:
        if not np.isscalar(efac):
            raise ValueError("If flags is None, efac must be a scalar.")
        efacvec = np.full(psr.nobs, efac)
    elif flagid is not None and not np.isscalar(efac):
        if len(efac) == len(flags):
            flagvals = np.array(psr.flagvals(flagid))
            for ct, flag in enumerate(flags):
                efacvec[flagvals == flag] = efac[ct]

    # Add TOA errors with random noise
    psr.stoas[:] += efacvec * psr.toaerrs * (1e-6 / DAY) * np.random.randn(psr.nobs)

def add_time_corr_signal(psr, A, gamma, components=10, tspan=None, seed=None, idx=0, factor=1.0):
    """
    Taken from libstempo.
    Add DM variations with P(f) = A^2 / (12 pi^2) (f year)^-gamma using Fourier bases.
    Optionally use a pseudorandom-number-generator seed.
    """
    if seed is not None:
        np.random.seed(seed)

    t = psr.toas()
    fref = 1400
    v = (fref / psr.freqs) ** idx

    minx, maxx = np.min(t), np.max(t)
    if tspan is None:
        x = (t - minx) / (maxx - minx)
        T = (DAY / YEAR) * (maxx - minx)
    else:
        x = (t - minx) / tspan
        T = (DAY / YEAR) * tspan

    size = 2 * components
    F = np.zeros((psr.nobs, size))
    f = np.zeros(size)

    # Vectorized computation of Fourier components
    i_vals = np.arange(1, components + 1)
    i_vals = i_vals[:, np.newaxis]  # Shape (30, 1)
    x = np.asarray(x)  # Ensure x is a 1D array
    F[:, ::2] = np.cos(2 * np.pi * i_vals * x).T  # Cosine terms
    F[:, 1::2] = np.sin(2 * np.pi * i_vals * x).T  # Sine terms

    f[::2] = f[1::2] = np.squeeze(i_vals / T)

    norm = A**2 * YEAR**2 / (12 * np.pi**2 * T)
    prior = norm * f ** (-gamma)

    y = np.sqrt(prior) * np.random.randn(size)
    psr.stoas[:] += (1.0 / DAY) * np.dot(F, y) * v * factor
    
    
    
############################################################# SIMULATE MULTI PULSAR DATA ########################################################################################
    
    
######################################################################### THE FOLLOWING SIMULATE ALL 25 PULSARS IN DR2new
# data dir
# data_dir = os.path.join(homedir, "data", "EPTA_DR2", "DR2new")
data_dir=args.data_dir

# for to span DR2new
pulsars = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]


cadence = 5.0  # days
start_mjd, end_mjd = 52000, 59000
num_toas = int((end_mjd - start_mjd) / cadence)

toas = np.linspace(start_mjd, end_mjd, num_toas)  # MJD
toaerrs = 1  # microseconds
freqs = np.random.choice([500, 900, 1400], num_toas)  # MHz

# Create pulsars
pulsar_objects = {}
for psrname in pulsars:
    parfile = os.path.abspath(os.path.join(data_dir, psrname, f"{psrname}.par"))
    pulsar_objects[psrname] = LTsim.fakepulsar(parfile, obstimes=toas, toaerr=toaerrs, freq=freqs)

# Add gravitational wave background (GWB) and save residuals
for psrname, psr_obj in pulsar_objects.items():
    add_gwb(psr_obj, gwAmp=2e-15, flow=1e-9, fhigh=1e-5)
    pickle.dump(psr_obj.residuals() * 1e6, open(f'GWB_res_{psrname}.pkl', 'wb'))

# Compute injected power spectrum
Tspan = 25.4 * 365.25 * 24 * 3600

G = 6.6743e-8
M_solar = 1.98847e33
c = 2.99792458e10

df = 1 / Tspan
N = 30
f_yr = 1 / (365.25 * 24 * 3600)

def compute_PSD(f, gamma, log10_A):
    return (10**(log10_A))**2/12/np.pi**2*(f/f_yr)**(-gamma)*(np.pi*10**7)**3

def hc_to_logrho(f_bin, hc, Tspan):
    return 0.5 * np.log10(hc**2 / 12 / np.pi**2 / f_bin**3 / Tspan)

def PSD_to_logrho(f_bin, PSD, Tspan):
    return  0.5 * np.log10(PSD / Tspan)

f = np.linspace(1e-9, 1e-7, 1000)
PSD = compute_PSD(f, 13/3, np.log10(2e-15))
log_rho = PSD_to_logrho(f, PSD, Tspan)

# Add white noise (EFAC) to pulsars
efac = 1.0
for psrname, psr_obj in pulsar_objects.items():
    res_i = np.copy(psr_obj.residuals())
    add_efac(psr_obj, efac=efac, seed=9827)
    res_wn = np.copy(psr_obj.residuals()) - res_i

# Add red noise (Time-Correlated Signal) to pulsars
A_rn = 7e-14  # np.random.uniform(1e-15, 1e-13)
gamma_rn = 3  # np.random.uniform(1, 5)

for psrname, psr_obj in pulsar_objects.items():
    toa_tmp = np.copy(psr_obj.residuals())
    add_time_corr_signal(psr_obj, A=A_rn, gamma=gamma_rn, components=30, seed=4105)
    res_rn = np.copy(psr_obj.residuals()) - toa_tmp

# Add dispersion measure (DM) variations to pulsars
A_dm = 5e-14
gamma_dm = 2.3

for psrname, psr_obj in pulsar_objects.items():
    toa_tmp = np.copy(psr_obj.residuals())
    add_time_corr_signal(psr_obj, A=A_dm, gamma=gamma_dm, components=30, idx=2, seed=413405)
    res_dm = np.copy(psr_obj.residuals()) - toa_tmp

# Print red noise parameters
def print_parameters(param_name, value):
    print(f"{param_name:<10} | {value:<18.5e}")

# print(f"{'Parameter':<10} | {'Value':<18}")
# print("-" * 30)
# print_parameters("A_rn", A_rn)
# print_parameters("gamma_rn", gamma_rn)
# print_parameters("A_dm", A_dm)
# print_parameters("gamma_dm", gamma_dm)


################################################################# CREATE MODEL WITH ENTERPRISE ################################################################################



multi_pulsars = {psrname: Pulsar(psr_obj) for psrname, psr_obj in pulsar_objects.items()}

# Define psr_multi
psr_multi = []
for psrname, psr in multi_pulsars.items():
    # include TM para marginalized
    s = gp_signals.TimingModel()
    
    # include WN EFAC
    efac_prior = parameter.Uniform(0.1, 5)
    s += white_signals.MeasurementNoise(efac=efac_prior)
    
    # include red noise
    log10_A_rn_prior = parameter.Uniform(-18, -10)
    gamma_rn_prior = parameter.Uniform(0, 7)
    pl = powerlaw(log10_A=log10_A_rn_prior, gamma=gamma_rn_prior)
    s += gp_signals.FourierBasisGP(pl, components=30)
    
    # include DM variations
    log10_A_dm_prior = parameter.Uniform(-18, -10)
    gamma_dm_prior = parameter.Uniform(0, 7)
    pl_dm = powerlaw(log10_A=log10_A_dm_prior, gamma=gamma_dm_prior)
    dm_basis = createfourierdesignmatrix_dm(nmodes=30)
    s += gp_signals.BasisGP(pl_dm, dm_basis, name="dm_gp")
    
    # create 
    psr_multi.append(s(psr))  

# Crete PTA object from psr_multi
ptaM = PTA(psr_multi) 




####################################################### SAMPLING WITH NESSAI ###################################################################

homedir=args.homedir
# Set up the outdir name for every run
run_name = f"SimDR2newAllPSR-{args.nlive}-{args.n_blocks}-{args.n_layers}-{args.n_neurons}-{args.patience}-{args.pytorch_threads}-{args.n_pool}"
# Modify outdir name if min_iteration is set or not
if args.min_iteration is not None:
    run_name += f"-{args.min_iteration}"
else:
    run_name += "-autoiter"
output = os.path.join(homedir, "SimulatedPSR", "outdir", "DR2newAllPSR", run_name)
os.makedirs(output, exist_ok=True)
print(f"Output directory: {output}")

# put the log file in the correct outdir for every run
logger = setup_logger(output=output)


class BaseNessaiModel(Model):

    def __init__(self):
        # Names of parameters to sample
        self.names = [p.name for p in ptaM.params]
        # Prior bounds for each parameter
        temp_bounds = {}
        for p in ptaM.params:
            temp_bounds[p.name] = [
                p.prior.func_kwargs["pmin"],
                p.prior.func_kwargs["pmax"],
            ]
        self.bounds = temp_bounds.copy() # THIS HAS TO BE A dict[str, list[int]] 
            

    def log_prior(self, x):
        # Check if values are in bounds, returns True/False
        # Then take the log to get 0/-inf and make sure the dtype is float
        log_p = np.log(self.in_bounds(x), dtype="float")
        
    

# since the live points are a structured array we can
# get each value using just the name
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
 # Use a Gaussian logpdf and iterate through the parameters
        return ptaM.get_lnlikelihood([x[n] for n in self.names]) 
   

############################################ NO IMPORTANCE SAMPLING #################################################################
# flow_config = dict(n_blocks=8, n_neurons=10, n_layers=8, ftype="RealNVP")

# #Configure the sampler.
# fs = FlowSampler(
#     BaseNessaiModel(),
#     output=output,
#     flow_config=flow_config,
#     resume=False,
#     seed=1234,
# )

# And go!


########################################### IMPORTANCE NESTED SAMPLIING AND PARALLELIZATION #############################################################

    def to_unit_hypercube(self, x):
        """Map to the unit hyper-cube"""
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (x[n] - self.bounds[n][0]) / (
                self.bounds[n][1] - self.bounds[n][0]
            )
        return x_out

    def from_unit_hypercube(self, x):
        """Map from the unit hyper-cube"""
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (self.bounds[n][1] - self.bounds[n][0]) * x[
                n
            ] + self.bounds[n][0]
        return x_out


flow_config = dict(
                       n_blocks=args.n_blocks, 
                       n_layers=args.n_layers, 
                       n_neurons=args.n_neurons,
                       ftype="RealNVP"
                    )
#QUESTO L'HO PRESO DALLA CONFIGURAZIONE DI EGGBOX  
training_config = dict(
    patience=args.patience # is the number of iterations with no improvement in the validation loss to wait before stopping training early
)
  
    
##################################################################### PARALLELIZATION ################################################################
    # Using n_pool
logger.warning("Running nessai with n_pool")
# Configure the sampler with n total threads, k of which are used for
# evaluating the likelihood
###########################################


# The FlowSampler object is used to managed the sampling as has more configuration options
# Note the importance nested sampler has different settings to the standard sampler
# Define the dictionary containing the keyword arguments of FlowSampler
fs_args = {
    "model": BaseNessaiModel(),
    "nlive": args.nlive,
    "output": output,
    "resume": False,
    "seed": 1451,
    "importance_nested_sampler": True,
    # draw_constant=True,  # Draw a constant number of samples (2000) QUESTO L'HO TOLTO PERCHE' A PARITA' DEGLI ALTRI PARAMETRI FA TROPPO POCHI POSTERIOR
    ###################PARALLELIZATION########################
    "pytorch_threads": args.pytorch_threads,
    "n_pool": args.n_pool,
    ###########################################################
}

# If args.min_iteration is set, add the "min_iteration" element to the  dictionary
if args.min_iteration is not None:
    fs_args["min_iteration"] = args.min_iteration
    
# Pass to FlowSampler the keywords arguments **fs_args
fs = FlowSampler(**fs_args)



# and go!









fs.run()


 


    