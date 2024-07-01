import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.interpolate import interp1d
from scipy.misc import derivative
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
import sys

eoslaine = np.loadtxt("./Laine_nf3.dat")
ttab = eoslaine[:,2]*1000
etab = eoslaine[:,0]
ptab = eoslaine[:,1]

eosp = interp1d(etab, ptab, kind='cubic')

# cs2 from lattice qcd
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
    
# Define the ranges and resolutions
xmin, xmax, dx = -15, 15, 0.2
ymin, ymax, dy = -15, 15, 0.2
zmin, zmax, dz = -10, 10, 0.2

# Create arrays for each coordinate
nx = int((xmax - xmin) / dx) + 1
ny = int((ymax - ymin) / dy) + 1
nz = int((zmax - zmin) / dz) + 1

xpoints = np.linspace(xmin, xmax, nx)
ypoints = np.linspace(ymin, ymax, ny)
zpoints = np.linspace(zmin, zmax, nz)

# Meshgrid for transverse plane
Xt_x, Xt_y = np.meshgrid(xpoints, ypoints, indexing='ij')

# IC Parameters
sNN = 200
eta0 = 1.5
sigmaeta = 1.4
w = 0.4
eff = 0.15
etaB = 3.5
sigmaIN = 2.0
sigmaOUT = 0.1
mN = 0.939
ZoverA = 0.4
ybeam = np.arccosh(sNN/(2.*mN))

# Load the datafile with positions of participants from supermc code
data = np.loadtxt("./rhic200-20-60%_20kevents.data")
r = np.sqrt(data[:,0]**2+data[:,1]**2)
rcut = np.max(r) + 5*w
next = data[1:, 2]
prev = data[:-1, 2]
Nevents = np.sum((next == 1) & (prev == 2)) + 1
event_location_in_data = np.where((next == 1) & (prev == 2))[0] + 1

def yCM(TA,TB):
    return np.arctanh((TA-TB)/(TA+TB+1e-8) *np.tanh(ybeam))

def Mass(TA,TB):
    return mN*np.sqrt(TA*TA+TB*TB+2*TA*TB*np.cosh(2*ybeam))

def normalization(M):
    C = np.exp(eta0) *special.erfc(-np.sqrt(0.5)*sigmaeta) + np.exp(-eta0) *special.erfc(np.sqrt(0.5)*sigmaeta)
    return M/(2*np.sinh(eta0) + np.sqrt(0.5*np.pi) *sigmaeta *np.exp(sigmaeta*sigmaeta*0.5)*C )

def energy(N, etas, ycombo):
    eta0_new = np.minimum(eta0, ybeam - ycombo)
    absarg = np.abs(etas - ycombo) - eta0_new
    exp_fac = np.exp(-1. / (2 * sigmaeta * sigmaeta) * absarg * absarg)
    result = np.where(absarg >= 0, N*exp_fac,N)
    return result

def baryon_density_profile(etas, id):
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
        print("Problem with particle ID!")
        exit(1)
        return 0
    return result

def process_event(i, event_loc, partdata, xg, yg):
    if(i == 0):
        event_start = 0
        event_end = event_loc[0]
    else:
        event_start = event_loc[i-1] + 1
        event_end = event_loc[i]
                    
    mask = partdata[event_start:event_end,2] == 1
    partA_x = partdata[event_start:event_end][mask,0]
    partA_y = partdata[event_start:event_end][mask,1]
                    
    mask = partdata[event_start:event_end,2] == 2
    partB_x = partdata[event_start:event_end][mask,0]
    partB_y = partdata[event_start:event_end][mask,1]
        
    ta =  np.sum(np.exp(-1. / (2. * w * w) * ((xg[..., np.newaxis] - partA_x)**2 + (yg[..., np.newaxis] - partA_y)**2)), axis=-1)
    tb =  np.sum(np.exp(-1. / (2. * w * w) * ((xg[..., np.newaxis] - partB_x)**2 + (yg[..., np.newaxis] - partB_y)**2)), axis=-1)
    
    return ta,tb

def computeconservedvariables(ix,iy,iz,ta,tb,norm,y_CM): 
    x = xpoints[ix]
    y = ypoints[iy]
    z = zpoints[iz]
    Q = np.zeros(5)
    En = energy(norm[ix,iy], z, (1-eff) * y_CM[ix,iy])
    Nb_ = ta[ix,iy] * baryon_density_profile(z, 1) + tb[ix,iy] * baryon_density_profile(z, 2)
    Q[0] = En * np.cosh(eff * y_CM[ix,iy])
    Q[3] = En * np.sinh(eff * y_CM[ix,iy])
    Q[1] = 0.0
    Q[2] = 0.0
    Q[4] = 0.0
    e,p,nb,vx,vy,vz = transformPV(Q)
    gamma = 1.0 / np.sqrt(1 - vx*vx - vy*vy - vz*vz)
    Q[4] = gamma * Nb_
    e,p,nb,vx,vy,vz = transformPV(Q)
    if (e < 0.3):
        e = 0
        nb = 0
        vz = 0
    return nb,e,vz
                
def setIC():
    next = data[1:, 2]
    prev = data[:-1, 2]
    Nevents = np.sum((next == 1) & (prev == 2)) + 1
    event_location_in_data = np.where((next == 1) & (prev == 2))[0]
    
    ta = np.zeros(Xt_x.shape)
    tb = np.zeros(Xt_x.shape)
    y_CM = np.zeros(Xt_x.shape)
    mass = np.zeros(Xt_x.shape)
    norm = np.zeros(Xt_x.shape)
    
    num_jobs = len(event_location_in_data)
    
    results = Parallel(n_jobs=-1)(delayed(process_event)(i, event_location_in_data, data, Xt_x, Xt_y) for i in range(num_jobs))
    
    for ta_event, tb_event in results:
        ta += ta_event
        tb += tb_event
    
    ta *= 1. / (2. * np.pi * w * w * Nevents)
    tb *= 1. / (2. * np.pi * w * w * Nevents)
                    
    y_CM = yCM(ta, tb)
    mass = Mass(ta, tb)
    norm = normalization(mass)
                                
    results = Parallel(n_jobs=-1)(delayed(computeconservedvariables)(i, j, k,ta,tb,norm,y_CM) for i in range(nx) for j in range(ny) for k in range(nz))
    
    with open("supermcic.dat","w") as out_file:
        for idx, (ix, iy, iz) in enumerate(np.ndindex((nx, ny, nz))):
            nb, e, vz = results[idx]
            x = xpoints[ix]
            y = ypoints[iy]
            z = zpoints[iz]
            out_file.write("%8.3f %8.3f %8.3f %16.8f %16.8f %16.8f\n" % (x,y,z,e,nb,vz))

setIC()
