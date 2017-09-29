import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os

fname_data="../../DATA_FULL/data_alpha_70_DATA_FULL.txt"
fname_rand="../../DATA_FULL/random_alpha_DATA_FULL_70_MHI_10bins.txt"
dtor=np.pi/180
clight=299792.458
plot_stuff=False
weight_mhi=False
weight_fkp=True

h0=0.000333564095 

#Read data
data_dfull=np.genfromtxt(fname_data,
                     dtype=(int,float,float,float,float,float),
                     names=('agc','ra','dec','v50','vcmb','logMHI'))
data_rfull=np.genfromtxt(fname_rand,
                     dtype=(float,float,float,float,float),names=('ra','dec','v50','vcmb','logMHI'))

## data
print 'data:'
print data_dfull['logMHI'].shape
print data_dfull['logMHI'].max()
print '\n'

# #### Here I make a threshold in M_HI < 9.63 in data
# data_d = data_dfull[np.where(data_dfull['logMHI']>8.)]
# print data_d['logMHI'].shape
# print data_d['logMHI'].max()

## randoms
print '\nrandoms'
print data_rfull['logMHI'].shape
print data_rfull['logMHI'].max()
print '\n'

# #### Here I make a threshold in M_HI < 9.63
# data_r = data_rfull[np.where(data_rfull['logMHI']>8.)]
# print data_r['logMHI'].shape
# print data_r['logMHI'].max()


#### Here I remove one patch from one box

I = 12 ; J = 7 

boxDEClim, boxRAlim = np.loadtxt('../boxes_limits/box'+str(I)+'.txt', unpack=True)
print 'boxes limits', boxDEClim, boxRAlim

rbox = data_rfull[np.where((data_rfull['dec']>=boxDEClim[0]) & (data_rfull['dec']<boxDEClim[1]) &\
                             (data_rfull['ra']>=boxRAlim[0]) & (data_rfull['ra']<boxRAlim[1]))]


plt.scatter(data_rfull['ra'][::100],data_rfull['dec'][::100], marker='.', s=1)
plt.scatter(rbox['ra'][::100], rbox['dec'][::100], marker='.', s=1)
plt.savefig('test_box_'+str(I)+'_'+str(J)+'.pdf')
plt.close()

def surface(box):
    return (box['dec'].max()-box['dec'].min())*\
            np.absolute((box['ra'].max()-box['ra'].min()))*np.cos(np.mean(box['dec'])*np.pi/180.)
def max_min_dec(box):
    return (box['dec'].max()-box['dec'].min())
def max_min_ra(box):
    return (box['ra'].max()-box['ra'].min())

def F(x,deltaalpha,x0):
    return (x-x0)*deltaalpha*np.cos((x+x0)/2.*np.pi/180.) - S
def limits(box):
    x0 = np.min(box['dec'])
    deltaalpha = max_min_ra(box)
    lims = []
    lims.append(x0)
    i=0
    while optimize.fsolve(F,lims[i],args=(deltaalpha,lims[i])) < np.max(box['dec']):
        lims.append(optimize.fsolve(F,lims[i],args=(deltaalpha,lims[i])))
        i +=1
    return np.array(lims)

# # lets do box3s3
# boxDEClim3s3, boxRAlim3s3 = np.loadtxt('../boxes_limits/box3s3.txt', unpack=True)
# print 'box3s3_limits', boxDEClim3s3, boxRAlim3s3
# rbox3s3 = data_rfull[np.where((data_rfull['dec']>=boxDEClim3s3[0]) & (data_rfull['dec']<boxDEClim3s3[1]) &\
#                              (data_rfull['ra']>=boxRAlim3s3[0]) & (data_rfull['ra']<boxRAlim3s3[1]))]
# print 'S=', surface(rbox3s3)
S = 40.

lims = limits(rbox)
print 'lims', lims

data_r = data_rfull[np.where((data_rfull['ra']<boxRAlim[0]) | (data_rfull['ra']>boxRAlim[1])\
                              | (data_rfull['dec']<lims[J]) | (data_rfull['dec']>lims[J+1]))]

data_d = data_dfull[np.where((data_dfull['ra']<boxRAlim[0]) | (data_dfull['ra']>boxRAlim[1])\
                              | (data_dfull['dec']<lims[J]) | (data_dfull['dec']>lims[J+1]))]

plt.scatter(data_r['ra'][::100],data_r['dec'][::100], marker='.', s=1, color='k')
plt.scatter(data_d['ra'], data_d['dec'], marker='.', s=1, color='m')
plt.savefig('test_box_'+str(I)+'_'+str(J)+'_cut.pdf')
plt.close()

####
#Compute solar system dipole
th_d=(90-data_d['dec'])*dtor; phi_d=data_d['ra']*dtor;
pos_d=np.array([np.sin(th_d)*np.cos(phi_d),np.sin(th_d)*np.sin(phi_d),np.cos(th_d)])
th_r=(90-data_r['dec'])*dtor; phi_r=data_r['ra']*dtor;
pos_r=np.array([np.sin(th_r)*np.cos(phi_r),np.sin(th_r)*np.sin(phi_r),np.cos(th_r)])
vsol=np.dot(np.linalg.inv(np.dot(pos_d,np.transpose(pos_d))),
            np.dot(pos_d,data_d['v50']-data_d['vcmb']))
print 'V_sol = ',vsol,' km/s, |V_sol|=',np.sqrt(np.sum(vsol**2)),' km/s'
print 'Sanity check: ',[np.std(data_d['v50']-np.dot(vsol,pos_d)-data_d['vcmb']), 
                        np.std(data_r['v50']-np.dot(vsol,pos_r)-data_r['vcmb'])]

#Compute redshift
z_d=data_d['vcmb']/clight; r_d=z_d/h0
z_r=data_r['vcmb']/clight; r_r=z_r/h0

print r_d.max(), r_r.max()
print data_d['logMHI'][0:10]
h_z,z_b=np.histogram(z_r,bins=100,range=[600/clight,15100/clight])
hd_z,dum1=np.histogram(z_d,bins=100,range=[600/clight,15100/clight])

rb=z_b/h0
dcth=np.amax(np.cos(th_r))-np.amin(np.cos(th_r)); dphi=np.amax(phi_r)-np.amin(phi_r);
fsky=0.11028; nfrac=(len(data_d)+0.0)/len(data_r);
vol=4*np.pi*fsky*np.array([(rb[i+1]**3-rb[i]**3)/3 for i in np.arange(len(h_z))])


zarr = (z_b[1:]+z_b[:-1])/2
rarr = (rb[1:]+rb[:-1])/2.

# n0=0.23/5.; gamma=0.99; z0=0.0104;
def nz_fit_1(z) :
    # return n0*np.exp(-(z/z0)**gamma)
    return np.poly1d(np.polyfit(z, np.log(nfrac*h_z/vol),1))(zarr)
def nz_fit_2(z) :
    # return n0*np.exp(-(z/z0)**gamma)
    return np.poly1d(np.polyfit(z, np.log(nfrac*h_z/vol),2))(zarr)
# print 'Look', nz_fit(zarr)
# print nz_fit(zarr)
# print 'Typical number density (Mpc/h)^-3 :',nz_fit(0.02)
print 'Average number density (Mpc/h)^-3 :',(len(data_d)+0.)/(4*np.pi*fsky*(np.amax(r_d)**3-
                                                                                np.amin(r_d)**3)/3)
#### Weighting part

w_d = np.ones_like(z_d)
w_d = 10.**data_d['logMHI']

w_r = np.ones_like(z_r)
w_r = 10.**data_r['logMHI']

# z_bins = np.linspace(z_d.min(),z_r.max(),6)
# z_b_c = (z_bins[1:]+z_bins[:-1])/2

# w_random = np.zeros_like(z_r)
# w_r      = np.zeros_like(z_r)
# delta_zbc = np.diff(z_b_c)[0]
# for i in xrange(0,z_b_c.size):
#     print z_b_c[i], z_bins[i], z_bins[i+1]
#     hist, bins = np.histogram(HImass[np.where((z_d>z_bins[i])&(z_d<=z_bins[i+1]))], bins=20)
#     bin_midpoints = bins[:-1] + np.diff(bins)/2
#     cdf = np.cumsum(hist)
#     cdf = cdf / float(cdf[-1])
#     random_index = np.where((z_r>z_bins[i])&(z_r<=z_bins[i+1]))[0]
#     values = np.random.rand(random_index.size)
#     value_bins = np.searchsorted(cdf, values)
#     random_from_cdf = bin_midpoints[value_bins]
#     w_random[random_index] = random_from_cdf

# w_r[np.where(w_random>0.)[0]] = 10.**w_random

print 'Minimum and maximum data weights: ', w_d.min(), w_d.max()
print 'Minimum and maximum random weights: ', w_r.min(), w_r.max()

print len(w_d), len(z_d)
print len(w_r), len(z_r)

plt.plot(zarr,nfrac*h_z/vol, label = 'random')
plt.plot(zarr,hd_z/vol, label = 'data')
plt.plot(zarr,np.exp(nz_fit_1(zarr)),label = 'fit 1', alpha=0.5)
plt.plot(zarr,np.exp(nz_fit_2(zarr)),label = 'fit 2', color='k')
plt.yscale('log')
plt.xlabel('$r\,[h^{-1}\,{\\rm Mpc}]$',fontsize=20)
plt.ylabel('$n(r)\\,[({\\rm Mpc/h})^{-3}]$',fontsize=20)
plt.legend(loc=0)
plt.savefig('test.pdf')
plt.close()

plt.plot(zarr,nfrac*h_z/vol, label = 'random')
plt.plot(zarr,hd_z/vol, label = 'data')
plt.plot(zarr,np.exp(nz_fit_1(zarr)),label = 'fit 1', alpha=0.5)
plt.plot(zarr,np.exp(nz_fit_2(zarr)),label = 'fit 2', color='k')
plt.xlabel('$r\,[h^{-1}\,{\\rm Mpc}]$',fontsize=20)
plt.ylabel('$n(r)\\,[({\\rm Mpc/h})^{-3}]$',fontsize=20)
plt.legend(loc=0)
plt.savefig('test_lin.pdf')
plt.close()

# plt.plot(zarr,hd_z/vol, label = 'data')
# plt.plot(zarr,nfrac*h_z/vol, label = 'random')
# # plt.plot(zarr,nz_fit(zarr),label = 'fit')
# plt.plot(zarr,nz_fit(zarr),label = 'fit')
# # plt.yscale('log')
# plt.xlabel('$r\,[h^{-1}\,{\\rm Mpc}]$',fontsize=20)
# plt.ylabel('$n(r)\\,[({\\rm Mpc/h})^{-3}]$',fontsize=20)
# plt.legend(loc=0)
# plt.savefig('nr_10bins.pdf')
# plt.close()


plt.hist(z_d,normed=True,histtype='step',label='data')
plt.hist(z_r,normed=True,histtype='step',label='random')
plt.legend(loc=0)
plt.savefig('hist_z.pdf')
plt.close()

plt.hist(z_d,normed=True,histtype='step',label='data',weights=w_d)
plt.hist(z_r,normed=True,histtype='step',label='random', weights=w_r)
plt.legend(loc=0)
plt.savefig('hist_z_weighted.pdf')
plt.close()
# #####


np.savetxt("data.txt",np.transpose([z_d,np.cos(th_d),phi_d,w_d]))
np.savetxt("random.txt",np.transpose([z_r,np.cos(th_r),phi_r,w_r]))

# np.savetxt('nr_data.txt',np.transpose([rarr,hd_z/vol]))
# np.savetxt('nr_random.txt',np.transpose([rarr,nfrac*h_z/vol]))
# np.savetxt('nr_fit.txt',np.transpose([rarr,nz_fit(zarr)]))
np.savetxt('nr_fit.txt',np.transpose([rarr,np.exp(nz_fit_2(zarr))]))



#Run CUTE
stout= "data_filename= data.txt\n"
stout+="random_filename= random.txt\n"
stout+="input_format= 0\n"
stout+="output_filename= ../all/corr_3D_ps_jackknife_box"+str(I)+"_"+str(J)+".txt\n"
stout+="omega_M= 0.315\n"
stout+="omega_L= 0.685\n"
stout+="w= -1.\n"
stout+="corr_type= 3D_ps\n"
stout+="log_bin= 1\n"
stout+="dim1_min_logbin= 0.1\n"
stout+="dim1_max= 60.\n"
stout+="dim1_nbin= 24\n"
stout+="dim2_max= 60.\n"
stout+="dim2_nbin= 60\n"
stout+="dim3_max= 20.\n"
stout+="dim3_min= 0.\n"
stout+="dim3_nbin= 20\n"
stout+="do_j3= 1\n"
stout+="j3_gamma= 1.51\n"
stout+="j3_r0= 3.3\n"
stout+="j3_ndens_file= nr_fit.txt\n"
f=open("param.ini","w")
f.write(stout)
f.close()
os.system("../../../CUTEdir/CUTE/CUTE param.ini > log.txt")
os.system("rm data.txt random.txt")
os.system("cat log.txt")
