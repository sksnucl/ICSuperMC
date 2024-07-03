import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.interpolate import interp1d
from scipy.misc import derivative
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
import sys

icparams = {
    "xmin"      :    -15.0,     # minimum x
    "xmax"      :    15.0,      # maximum x
    "ymin"      :    -15.0,     # minimum y
    "ymax"      :    15.0,      # maximumy
    "zmin"      :    -10.0,     # minimum eta. Must be <= -beam rapidity
    "zmax"      :    10.0,      # maximum eta. Must be >= beam rapidity
    "dx"        :    0.1,
    "dy"        :    0.1,
    "dz"        :    0.1,
    "sNN"       :    200.0,
    "eta0"      :    2.2,
    "sigmaeta"  :    0.9,
    "w"         :    0.8,
    "eff"       :    0.15,
    "etaB"      :    3.5,
    "sigmaIN"   :    2.0,
    "sigmaOUT"  :    0.1,
    "mN"        :    0.939
}

eoslaine = np.loadtxt("./Laine_nf3.dat")
ttab = eoslaine[:,2]*1000
etab = eoslaine[:,0]
ptab = eoslaine[:,1]

eosp = interp1d(etab, ptab, kind='cubic')

# cs2 using numerical derivative
def cs2(e):
    return derivative(eosp,e,dx=1e-2)

# Function to convert conserved variables to primitive variables, taken from vhlle
def transformPV(Q):
    MAXIT = 100
    dpe = 1.0 / 3.0
    corrf = 0.9999

    M = np.sqrt(Q[1] * Q[1] + Q[2] * Q[2] + Q[3] * Q[3])

    if Q[0] <= 0.0:
        e = 0.0
        p= 0.0
        vx = 0.0
        vy = 0.0
        vz = 0.0
        nb = 0.0
        return e, p, nb, vx, vy, vz

    if M == 0.0:
        e = Q[0]
        vx = 0.0
        vy = 0.0
        vz = 0.0
        nb = Q[4]
        p = eosp(e)
        return e, p, nb, vx, vy, vz

    if M > Q[0]:
        Q[1] *= corrf * Q[0] / M
        Q[2] *= corrf * Q[0] / M
        Q[3] *= corrf * Q[0] / M
        M = Q[0] * corrf

    vl, vh = 0.0, 1.0
    v = 0.5 * (vl + vh)

    e = Q[0] - M * v
    if e < 0.0:
        e = 0.0

    nb = Q[4] * np.sqrt(1 - v * v)
    p = eosp(e)

    f = (Q[0] + p) * v - M
    df = (Q[0] + p) - M * v * dpe
    dvold = vh - vl
    dv = dvold

    for i in range(MAXIT):
        if ((f + df * (vh - v)) * (f + df * (vl - v)) >= 0.0 or abs(2.0 * f) > abs(dvold * df)):
            dvold = dv
            dv = 0.5 * (vh - vl)
            v = vl + dv
        else:
            dvold = dv
            dv = f / df
            v -= dv

        if abs(dv) < 0.00001:
            break

        e = Q[0] - M * v
        if e < 0.0:
            e = 0.0

        nb = Q[4] * np.sqrt(1 - v * v)

        p = eosp(e)

        f = (Q[0] + p) * v - M
        df = (Q[0] + p) - M * v * dpe

        if f > 0.0:
            vh = v
        else:
            vl = v

    vx = v * Q[1] / M
    vy = v * Q[2] / M
    vz = v * Q[3] / M

    e = Q[0] - M * v
    p = eosp(e)
    nb = Q[4] * np.sqrt(1 - vx**2 - vy**2- vz**2)

    return e, p, nb, vx, vy, vz
    
def yCM(TA,TB,ybeam):
    return np.arctanh((TA-TB)/(TA+TB+1e-8) *np.tanh(ybeam))

def Mass(TA,TB,mN,ybeam):
    return mN*np.sqrt(TA*TA+TB*TB+2*TA*TB*np.cosh(2*ybeam))

def normalization(M,icdict):
    eta0 = icdict["eta0"]
    sigmaeta = icdict["sigmaeta"]
    C = np.exp(eta0) *special.erfc(-np.sqrt(0.5)*sigmaeta) + np.exp(-eta0) *special.erfc(np.sqrt(0.5)*sigmaeta)
    return M/(2*np.sinh(eta0) + np.sqrt(0.5*np.pi) *sigmaeta *np.exp(sigmaeta*sigmaeta*0.5)*C )

def energy(N, etas, ycombo, ybeam, icdict):
    eta0 = icdict["eta0"]
    sigmaeta = icdict["sigmaeta"]
    eta0_new = np.minimum(eta0, ybeam - ycombo)
    absarg = np.abs(etas - ycombo) - eta0_new
    exp_fac = np.exp(-1. / (2 * sigmaeta * sigmaeta) * absarg * absarg)
    result = np.where(absarg >= 0, N*exp_fac,N)
    return result

def baryon_density_profile(etas, id, sigmaIN, sigmaOUT, etaB):
    norm = 1. / (np.sqrt(2 * np.pi) * (sigmaIN + sigmaOUT))
    if id == 1:
        variable = etas - etaB
        if variable >= 0:
            result = norm * np.exp(-1. / (2 * sigmaOUT * sigmaOUT) * variable * variable)
        else:
            result = norm * np.exp(-1. / (2 * sigmaIN * sigmaIN) * variable * variable)
    elif id == 2:
        variable = etas + etaB
        if variable >= 0:
            result = norm * np.exp(-1. / (2 * sigmaIN * sigmaIN) * variable * variable)
        else:
            result = norm * np.exp(-1. / (2 * sigmaOUT * sigmaOUT) * variable * variable)
    else:
        print("Wrong participant id!")
        exit(1)
        return 0
    return result

def process_event(i, event_loc, partdata, xg, yg, w):
    if i == 0:
        event_start = 0
        event_end = event_loc[0]
    else:
        event_start = event_loc[i-1]
        event_end = event_loc[i]

    ta = np.zeros_like(xg)
    tb = np.zeros_like(xg)

    maskA = partdata[event_start:event_end, 2] == 1
    partA_x = partdata[event_start:event_end][maskA, 0]
    partA_y = partdata[event_start:event_end][maskA, 1]
    xcm = np.sum(partA_x)/len(partA_x)
    ycm = np.sum(partA_y)/len(partA_y)
    r_max_A = np.max(np.sqrt((partA_x-xcm)**2 + (partA_y-ycm)**2)) if partA_x.size > 0 else 0
    rg = np.sqrt((xg-xcm)**2 + (yg-ycm)**2)
    
    gridpts_within_rmax = rg < r_max_A + 5*w
    indices_within_rmax = np.where(gridpts_within_rmax)
    
    for idx in zip(*indices_within_rmax):
        x = xg[idx]
        y = yg[idx]
        if r_max_A > 0:
            expA = np.exp(-1. / (2. * w * w) * ((x - partA_x)**2 + (y - partA_y)**2))
            ta[idx] = np.sum(expA)

    maskB = partdata[event_start:event_end, 2] == 2
    partB_x = partdata[event_start:event_end][maskB, 0]
    partB_y = partdata[event_start:event_end][maskB, 1]
    xcm = np.sum(partB_x)/len(partB_x)
    ycm = np.sum(partB_y)/len(partB_y)
    r_max_B = np.max(np.sqrt((partB_x-xcm)**2 + (partB_y-ycm)**2)) if partB_x.size > 0 else 0
    rg = np.sqrt((xg-xcm)**2 + (yg-ycm)**2)
    
    gridpts_within_rmax = rg < r_max_B + 5*w
    indices_within_rmax = np.where(gridpts_within_rmax)
    
    for idx in zip(*indices_within_rmax):
        x = xg[idx]
        y = yg[idx]
        
        if r_max_B > 0:
            expB = np.exp(-1. / (2. * w * w) * ((x - partB_x)**2 + (y - partB_y)**2))
            tb[idx] = np.sum(expB)
    
    return ta, tb

def sum_chunk_results(ta, tb, results):
    for ta_event, tb_event in results:
        ta += ta_event
        tb += tb_event
    return ta, tb

def process_chunk(start_idx, end_idx, event_loc, partdata, xg, yg, w):
    results = Parallel(n_jobs=-1)(delayed(process_event)(i, event_loc, partdata, xg, yg, w) for i in range(start_idx, end_idx))
    return sum_chunk_results(np.zeros_like(xg), np.zeros_like(xg), results)  

def computeconservedvariables(idx, nx, ny, nz, xpts, ypts, zpts, ta, tb, norm, y_CM, ybeam, icdict):
    sigmaIN = icdict["sigmaIN"]
    sigmaOUT = icdict["sigmaOUT"]
    etaB = icdict["etaB"]
    eff = icdict["eff"]
    #nx, ny, nz = ta.shape
    iz = idx % nz
    idx //= nz
    iy = idx % ny
    ix = idx // ny
    x = xpts[ix]
    y = ypts[iy]
    z = zpts[iz]
    if ta[ix, iy] == 0 and tb[ix, iy] == 0:
        return x,y,z,0, 0, 0
    Q = np.zeros(5)
    En = energy(norm[ix, iy], z, (1 - eff) * y_CM[ix, iy], ybeam, icdict)
    Nb_ = ta[ix, iy] * baryon_density_profile(z, 1,sigmaIN, sigmaOUT, etaB) + tb[ix, iy] * baryon_density_profile(z, 2,sigmaIN, sigmaOUT, etaB)
    Q[0] = En * np.cosh(eff * y_CM[ix, iy])
    Q[3] = En * np.sinh(eff * y_CM[ix, iy])
    Q[1] = 0.0
    Q[2] = 0.0
    Q[4] = 0.0
    e, p, nb, vx, vy, vz = transformPV(Q)
    gamma = 1.0 / np.sqrt(1 - vx * vx - vy * vy - vz * vz)
    Q[4] = gamma * Nb_
    e, p, nb, vx, vy, vz = transformPV(Q)
    if e < 0.3:
        e = 0
        nb = 0
        vz = 0
    return x,y,z,nb,e,vz

def write_ic_to_file(results):
    with open("supermcic.dat", "w") as out_file:
        for result in results:
            x, y, z, nb, e, vz = result
            out_file.write("%8.3f %8.3f %8.3f %16.8f %16.8f %16.8f\n" % (x, y, z, nb, e, vz))

def setIC(icdict):
    # Define the ranges and resolutions
    xmin = icdict["xmin"]
    xmax = icdict["xmax"]
    ymin = icdict["ymin"]
    ymax = icdict["ymax"]
    zmin = icdict["zmin"]
    zmax = icdict["zmax"]
    dx = icdict["dx"]
    dy = icdict["dy"]
    dz = icdict["dz"]

    nx = int((xmax - xmin) / dx) + 1
    ny = int((ymax - ymin) / dy) + 1
    nz = int((zmax - zmin) / dz) + 1

    # Create arrays for each coordinate
    xpoints = np.linspace(xmin, xmax, nx)
    ypoints = np.linspace(ymin, ymax, ny)
    zpoints = np.linspace(zmin, zmax, nz)

    # Meshgrid for transverse plane
    Xt_x, Xt_y = np.meshgrid(xpoints, ypoints, indexing='ij')

    sNN = icdict["sNN"]
    mN = icdict["mN"]
    w = icdict["w"]
    ybeam = np.arccosh(sNN/(2.*mN))

    # Load the datafile with positions of participants from supermc code
    data = np.loadtxt("./rhic200-20-60%_20kevents.data")

    next = data[1:, 2]
    prev = data[:-1, 2]
    Nevents = 5000#np.sum((next == 1) & (prev == 2)) + 1
    event_location_in_data = np.where((next == 1) & (prev == 2))[0] + 1
    
    ta = np.zeros(Xt_x.shape)
    tb = np.zeros(Xt_x.shape)
    y_CM = np.zeros(Xt_x.shape)
    mass = np.zeros(Xt_x.shape)
    norm = np.zeros(Xt_x.shape)
    
    total_jobs = 5000#len(event_location_in_data)
    chunk_size = 100
    
    num_jobs = (total_jobs + chunk_size - 1) // chunk_size
    
    results = Parallel(n_jobs=-1)(delayed(process_chunk)(i*chunk_size,min((i+1)*chunk_size,total_jobs),event_location_in_data, data, Xt_x, Xt_y, w) for i in range(num_jobs))
    
    ta, tb = sum_chunk_results(ta, tb, results)

    ta *= 1. / (2. * np.pi * w * w * Nevents)
    tb *= 1. / (2. * np.pi * w * w * Nevents)

    y_CM = yCM(ta, tb, ybeam)
    mass = Mass(ta, tb,mN,ybeam)
    norm = normalization(mass,icdict)
    
    total_grid_points = nx * ny * nz
    results = Parallel(n_jobs=-1)(delayed(computeconservedvariables)(idx, nx, ny, nz, xpoints, ypoints, zpoints, ta, tb, norm, y_CM, ybeam, icdict) for idx in range(total_grid_points))
    write_ic_to_file(results)
    
setIC(icparams)
