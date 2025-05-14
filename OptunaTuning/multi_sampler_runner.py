
import os
import numpy as np
import libstempo as LT
import libstempo.toasim as LTsim
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter, white_signals, gp_signals
from enterprise.signals.signal_base import PTA
from enterprise.signals.utils import powerlaw, createfourierdesignmatrix_dm
from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
import multiprocessing
import time
import random

# Constants
DAY = 24 * 3600
YEAR = 365.25 * DAY

def add_efac(psr, efac=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    psr.stoas[:] += efac * psr.toaerrs * (1e-6 / DAY) * np.random.randn(psr.nobs)

def add_time_corr_signal(psr, A, gamma, components=10, tspan=None, seed=None, idx=0, factor=1.0):
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

    i_vals = np.arange(1, components + 1)[:, np.newaxis]
    x = np.asarray(x)
    F[:, ::2] = np.cos(2 * np.pi * i_vals * x).T
    F[:, 1::2] = np.sin(2 * np.pi * i_vals * x).T

    f[::2] = f[1::2] = np.squeeze(i_vals / T)
    norm = A**2 * YEAR**2 / (12 * np.pi**2 * T)
    prior = norm * f ** (-gamma)

    y = np.sqrt(prior) * np.random.randn(size)
    psr.stoas[:] += (1.0 / DAY) * np.dot(F, y) * v * factor

def run_multi_pulsar_sampler(config, output_path, seed=None):
    homedir = os.path.abspath("PTA_NESSAI")
    pulsars = ["J0030+0451", "J1744-1134", "J1909-3744"]
    cadence = 5.0
    start_mjd, end_mjd = 52000, 59000
    num_toas = int((end_mjd - start_mjd) / cadence)

    toas = np.linspace(start_mjd, end_mjd, num_toas)
    toaerrs = 1
    freqs = np.random.choice([500, 900, 1400], num_toas)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    pulsar_objects = {}
    for psrname in pulsars:
        parfile = os.path.join(homedir, "data", "EPTA_DR2", "DR2new", psrname, f"{psrname}.par")
        pulsar_objects[psrname] = LTsim.fakepulsar(parfile, obstimes=toas, toaerr=toaerrs, freq=freqs)

    for psr in pulsar_objects.values():
        add_efac(psr, efac=1.0, seed=9827)
        add_time_corr_signal(psr, A=7e-14, gamma=3, components=30, seed=4105)
        add_time_corr_signal(psr, A=5e-14, gamma=2.3, components=30, idx=2, seed=413405)

    multi_pulsars = {psrname: Pulsar(psr_obj) for psrname, psr_obj in pulsar_objects.items()}
    psr_multi = []
    for psr in multi_pulsars.values():
        s = gp_signals.TimingModel()
        efac_prior = parameter.Uniform(0.1, 5)
        s += white_signals.MeasurementNoise(efac=efac_prior)
        log10_A_rn = parameter.Uniform(-18, -10)
        gamma_rn = parameter.Uniform(0, 7)
        s += gp_signals.FourierBasisGP(powerlaw(log10_A=log10_A_rn, gamma=gamma_rn), components=30)
        log10_A_dm = parameter.Uniform(-18, -10)
        gamma_dm = parameter.Uniform(0, 7)
        s += gp_signals.BasisGP(powerlaw(log10_A=log10_A_dm, gamma=gamma_dm), createfourierdesignmatrix_dm(nmodes=30), name="dm_gp")
        psr_multi.append(s(psr))

    pta = PTA(psr_multi)
    names = [p.name for p in pta.params]
    bounds = {p.name: [p.prior.func_kwargs["pmin"], p.prior.func_kwargs["pmax"]] for p in pta.params}

    class MultiPulsarModel(Model):
        def __init__(self):
            self.names = names
            self.bounds = bounds

        def log_prior(self, x):
            if not np.isfinite(x).all():
                return -np.inf
            lp = np.log(self.in_bounds(x), dtype="float")
            for n in self.names:
                lp -= np.log(self.bounds[n][1] - self.bounds[n][0])
            return lp

        def log_likelihood(self, x):
            if not np.isfinite(x).all():
                return -np.inf
            return pta.get_lnlikelihood([x[n] for n in self.names])

        def to_unit_hypercube(self, x):
            x_out = x.copy()
            for n in self.names:
                x_out[n] = (x[n] - self.bounds[n][0]) / (self.bounds[n][1] - self.bounds[n][0])
            return x_out

        def from_unit_hypercube(self, x):
            x_out = x.copy()
            for n in self.names:
                x_out[n] = x[n] * (self.bounds[n][1] - self.bounds[n][0]) + self.bounds[n][0]
            return x_out

    model = MultiPulsarModel()
    model.configure()

    flow_config = {
        "n_blocks": config["n_blocks"],
        "n_layers": config["n_layers"],
        "n_neurons": config["n_neurons"],
        "ftype": "RealNVP"
    }

    torch_threads = config.get("pytorch_threads", 1)
    n_pool = config.get("n_pool", 1)

    os.makedirs(output_path, exist_ok=True)
    logger = setup_logger(output=output_path)

    with multiprocessing.get_context("fork").Pool(n_pool) as pool:
        fs = FlowSampler(
            model,
            flow_config=flow_config,
            output=output_path,
            pool=pool,
            resume=False,
            seed=seed,
            importance_nested_sampler=True,
            pytorch_threads=torch_threads
        )

        t0 = time.time()
        fs.run()
        t1 = time.time()

    return t1 - t0
