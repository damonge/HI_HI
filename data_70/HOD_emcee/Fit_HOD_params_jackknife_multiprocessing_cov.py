import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate
import scipy.integrate as si
import scipy.special as spec
import pyfftlog

def FFTLOGtransform_Xi(ks,Pks,n=512,mu=0.,q=0,kr=1.,kropt=1,tdir=1):
    # Range of periodic interval
    logkmin = np.log10(ks[0])
    logkmax = np.log10(ks[-1])
#     print logkmin, logkmax
    logkc = (logkmin + logkmax)/2.
    nc = (n + 1)/2.0
    # Log-spacing of points
    dlogk = (logkmax - logkmin)/n
    dlnk = dlogk*np.log(10.0)
    # Calculation related to the logarithmic spacing
    kfft = 10**(logkc + (np.arange(1, n+1) - nc)*dlogk)
    ak = np.interp(kfft,ks,Pks)*kfft
    # Initialize FFTLog transform
    kr, xsave = pyfftlog.fhti(n, mu, dlnk, q, kr, kropt)
    logrc = np.log10(kr) - logkc
    rk = 10**(logkc - logrc)
    # Transform
    ar = pyfftlog.fht(ak.copy(), xsave, tdir)
    rfft = 10**(logrc + (np.arange(1, n+1) - nc)*dlogk)
    final_xi = ar*(1./(2.*np.pi))/rfft
    return [rfft,final_xi]

Omega_bh2 = 0.02230; Omega_bh21s = 0.00014
Omega_ch2 = 0.1188 ; Omega_ch21s = 0.0010
h = 0.6774; h1s = 0.0046
Omega_m = (Omega_ch2+Omega_bh2)/h**2
Omega_l = 1.-Omega_m
A_s1010 = 2.142*10.; A_s10101s = 0.049*10
n_s = 0.9667; n_s1s = 0.0040
sigma_8 = 0.8159; sigma_81s = 0.0086
H0 = 100.; c = 3e5; 
M_nu = 0.06; 
# Neff = 3.04; Neff1s=0.33

print(Omega_m, Omega_l, h, np.log(A_s1010), A_s1010/1e10)
# print(u"Mean acceptance fraction: ", np.mean(sampler.acceptance_fraction))


#we use this function to compute the comoving distance to a given redshift
def func(y,x,Omega_m,Omega_L):
    return [1.0/np.sqrt(Omega_m*(1.0+x)**3+Omega_L)]
##############################################################################

##############################################################################
#This functions computes the comoving distance to redshift z, in Mpc/h
#As input it needs z, Omega_m and Omega_L. It assumes a flat cosmology
def comoving_distance(z,Omega_m,Omega_L):
    H0=100.0 #km/s/(Mpc/h)
    c=3e5    #km/s
    #compute the comoving distance to redshift z
    yinit=[0.0]
    z_limits=[0.0,z]
    I=si.odeint(func,yinit,z_limits,args=(Omega_m,Omega_L),
                rtol=1e-8,atol=1e-8,mxstep=100000,h0=1e-6)[1][0]
    r=c/H0*I
    return r


z = np.linspace(0,6,1000)
Dc = np.zeros(z.size)
for i,r in enumerate(z):
    Dc[i] = comoving_distance(r,Omega_m,Omega_l)    
Dcom = scipy.interpolate.interp1d(z, Dc, kind='linear')

# Here we also define E(z) since we will need it to compute the comoving volumes:
def E(z,Omega_m,Omega_l):
    return np.sqrt(Omega_l + Omega_m*(1+z)**3)

# Here we compute the linear growth factor using Pacos code and interpolate it:
##############################################################################
def func_lgf(y,x,Omega_m,Omega_L):
    #print x, 1.0/(x**3 * (np.sqrt(Omega_m/x**3 + Omega_L))**3)
    return 1.0/(x*np.sqrt(Omega_m/x**3 + Omega_L))**3

#This function computes the linear growth factor. See Eq. 1 of 0006089
#Notice that in that formula H(a) = (Omega_m/a^3+Omega_L)^1/2 and that the 
#growth is D(a), not g(a). We normalize it such as D(a=1)=1
def linear_growth_factor(z,Omega_m,Omega_L):
    
    # compute linear growth factor at z and z=0
    yinit = [0.0];  a_limits = [1e-30, 1.0/(1.0+z), 1.0/(1.0+0.0)]
    I = si.odeint(func_lgf,yinit,a_limits,args=(Omega_m,Omega_L),
                  rtol=1e-10,atol=1e-10,mxstep=100000,h0=1e-20)[1:]
    redshifts = np.array([ [z], [0.0] ])
    Ha = np.sqrt(Omega_m*(1.0+redshifts)**3 + Omega_L)
    D = (5.0*Omega_m/2.0)*Ha*I

    return D[0]/D[1]
##############################################################################

l = np.zeros(z.size)
for i,r in enumerate(z):
    l[i]=linear_growth_factor(r,Omega_m,Omega_l)
lgf = scipy.interpolate.interp1d(z, l, kind='quadratic', bounds_error=False, fill_value=0.)


path = '/home/aobuljen/hibias/HI_HI/data_70/HOD_Fitting_jackknife_full/'
k, Pk = np.loadtxt(path + "k_Pk_fiducial_z_0_kmax_100_kext_1e4.txt", unpack=True)
Pkz0 = scipy.interpolate.interp1d(k, Pk)


# Constants
pi=np.pi
delta_crit = 1.686
rho_crit   = 2.77536627e11 #h^2 Msun/Mpc^3

#Define a redshift array:
z_nu = np.arange(0,6,0.1)
# path = ''
Mtable, nutable = np.loadtxt(path + "M_nu_tables/Planck_M_nu_%.2f.txt"%z_nu[0], unpack=True)
print("Mbins = ", Mtable.size, "Min Mass", np.log10(np.min(Mtable)), "Max Mass", np.log10(np.max(Mtable)))


# Tinker halo bias 2010
def Tinker_halo_bias(nu):
    Delta=200.0
    y=np.log10(Delta)

    A=1.0+0.24*y*np.exp(-(4.0/y)**4);          a=0.44*y-0.88
    B=0.183;                                   b=1.5
    C=0.019+0.107*y+0.19*np.exp(-(4.0/y)**4);  c=2.4

    return 1.0-A*nu**a/(nu**a+delta_crit**a)+B*nu**b+C*nu**c

# Tinker 2010 halo mass function
def f_nu(nus,z):
    alpha  = 0.368
    beta0  = 0.589;       beta  = beta0*(1.0+z)**0.20
    gamma0 = 0.864;       gamma = gamma0*(1.0+z)**(-0.01)
    phi0   = -0.729;      phi   = phi0*(1.0+z)**(-0.08)
    eta0   = -0.243;      eta   = eta0*(1.0+z)**0.27

    fnu = alpha*(1.0+(beta*nus)**(-2.0*phi))*nus**(2.0*eta)*\
            np.exp(-gamma*nus**2/2.0)
    return fnu

Ms = Mtable[:]
nus = nutable[:]
print(np.log10(Ms.min()), np.log10(Ms.max()))

#fiducial parameters:
cHI0f = 25
gammaf = 0.1
alphaf = 0.59
M0f = 7e10
logM0f = np.log10(M0f)
fiducial_OmegaHI0 = 4.00*1e-4

def OmegaM(z):
    return (Omega_m*(1+z)**3/(1.-Omega_m + Omega_m*(1.+z)**3))

def Ptotfast_allk(k,z,alpha=alphaf,cHI0=cHI0f,logM0=logM0f):
    OmM = OmegaM(z)
    MHI = Ms**alpha*np.exp(-10**logM0/Ms)
    OmegaHI_unnorm = OmM*np.trapz(f_nu(nus,z)*MHI/Ms,nus)
    Kf = fiducial_OmegaHI0/OmegaHI_unnorm
    MHI *=Kf
    rho_m = rho_crit*OmM
    rhoHIbar = OmegaHI_unnorm*Kf*rho_crit
    concHI = cHI0*(Ms/1e11/h)**(-0.109)*4./(1.+z)**gammaf
    _x = OmM-1.
    Delta_v = (18.*np.pi**2 + 82.*_x - 39.*_x**2)/(1. + _x)
    Rv = (3.0*Ms/(4.0*np.pi*Delta_v*rho_m))**(1.0/3.0)
    rs = Rv/concHI
    Parray = np.zeros_like(k)
    for ik,ki in enumerate(k):
        Parray[ik] = 1./rhoHIbar**2*np.trapz(f_nu(nus,0)*rho_m/Ms\
                    *MHI**2*(1./(1.+(ki*rs)**2)**2)**2,nus)+\
                Pkz0(ki)*1./rhoHIbar**2*(np.trapz(f_nu(nus,0)*rho_m/Ms\
                    *MHI*Tinker_halo_bias(nus)*(1./(1.+(ki*rs)**2)**2),nus))**2
    return Parray

# sdata, Xidata, Xierr = np.loadtxt(path + 'sigma_Xi_sigma_MHI_70_pimax_30_error_jackknife_113.0_40deg2.txt', unpack=True)
sdata_t, Xidata_t, Xierr_t = np.loadtxt(path + 'sigma_Xi_sigma_MHI_70_pimax_30_error_jackknife_113.0_40deg2.txt', unpack=True)
sdata = sdata_t[:-2]
Xidata= Xidata_t[:-2]
Xierr = Xierr_t[:-2]

iCovt = np.loadtxt(path+'icov_jackknife_40.txt', unpack=True)
iCov = iCovt[:-2,:-2]

stotfftlog, Xitotfftlog = FFTLOGtransform_Xi(k,Ptotfast_allk(k,0.))

plt.plot(stotfftlog,Xitotfftlog,label = 'FFTlog tot')
plt.errorbar(sdata,Xidata,yerr=Xierr, fmt='.')
plt.xlim(1e-1,1e2)
plt.ylim(1,5e2)
plt.loglog()
plt.legend(loc=0)
plt.savefig('test_cut_cov.pdf', bbox_inches='tight')
plt.close()


def HOD_Xi(s,p):
    Ptottest = Ptotfast_allk(k,0,p[0],p[1],p[2])
    stemp,Xitottemp = FFTLOGtransform_Xi(k,Ptottest,kr=1,n=100)
    return np.interp(s,stemp,Xitottemp)
    
print(HOD_Xi(sdata,[alphaf,cHI0f,logM0f]))

import emcee

def lnlike(theta, x, y, icov):
    model = HOD_Xi(x, theta)
    diff  = (y-model)
    return -np.dot(diff,np.dot(icov,diff))/2.0

def lnprior(theta):
    ## flat priors:
    alpha_prior, cHI0_prior, logM0_prior = theta
    if 10.<cHI0_prior<400. and 0.<alpha_prior<2. and 8.<logM0_prior<13.:
        return 0.0
    return -np.inf 

def lnprob(theta, x, y, icov):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, icov)

# Initial parameters:
p0=[alphaf,cHI0f,logM0f]


ndim, nwalkers = 3, 20
pos = [p0 + 1e-3 * np.random.randn(ndim) for i in range (nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(sdata, Xidata, iCov), threads = 40)
sampler.run_mcmc(pos, 10000)

#what did sampler do? It drew a lot of nicely arranged points in the parameter space. Density of poins follows probability of k and n:
samples = sampler.chain[:, :, :].reshape((-1, ndim))

samples.shape


#what to pick from the samples? 
nll = lambda *args: -lnlike(*args)
alphaemcee, cHI0emcee, logM0emcee = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))

# print ,gamm
best_fit = [alphaemcee[0], cHI0emcee[0],logM0emcee[0]]
print(alphaemcee, cHI0emcee, logM0emcee)
print(best_fit)



plt.plot(stotfftlog,Xitotfftlog,label = 'FFTlog tot')
plt.plot(sdata,HOD_Xi(sdata,best_fit),label = 'FFTlog best fit')
plt.errorbar(sdata,Xidata,yerr=Xierr, fmt='.')
plt.xlim(1e-1,1e2)
plt.ylim(1,5e2)
plt.loglog()
plt.legend(loc=0)
plt.savefig('best_fit_cut_cov.pdf', bbox_inches='tight')
plt.close()

np.savetxt('samples_cut_cov.txt',np.transpose([samples]))

import corner

fig = corner.corner(samples[:,[0,1,2]], labels=['$\\alpha$', "$c_{HI0}$", "$log_{10}M_0$"], show_titles=True, quantiles=[0.16,0.5,0.84])
fig.savefig('corner_best_fit_cut_cov.pdf', bbox_inches='tight')
# fig.close()

print(best_fit)
