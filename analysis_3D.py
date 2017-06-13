import numpy as np
#import healpy as hp
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
#import py_cosmo_mad as csm

fname_data="data/data_cat_mhithresh7.5.csv"
fname_rand="data/rand_cat_mhithresh7.5.csv"
dtor=np.pi/180
clight=299792.458
plot_stuff=False
weight_mhi=False
weight_fkp=True

#pcs=csm.PcsPar()
#pcs.set_verbosity(1)
#pcs.background_set(0.315,0.685,0.049,-1.,0.,0.67,2.725)
h0=0.000333564095 #pcs.hubble(1.)

#Read data
data_d=np.genfromtxt(fname_data,skiprows=1,delimiter=',',
                     dtype=(int,float,float,float,float,float,float),
                     names=('agc','ra','dec','v50','verr','vcmb','logMHI'))
data_r=np.genfromtxt(fname_rand,skiprows=1,delimiter=',',
                     dtype=(float,float,float,float),names=('ra','dec','v50','vcmb'))

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
if plot_stuff :
    fig=plt.figure(); ax=fig.add_subplot(111,projection='3d');
    ax.scatter(data_d['vcmb']*pos_d[0],
               data_d['vcmb']*pos_d[1],
               data_d['vcmb']*pos_d[2],
               marker='.',s=1);
    plt.show()

#Compute redshift
z_d=data_d['vcmb']/clight; r_d=z_d/h0
z_r=data_r['vcmb']/clight; r_r=z_r/h0
h_z,z_b=np.histogram(z_r,bins=100,range=[600/clight,15100/clight])
hd_z,dum1=np.histogram(z_d,bins=100,range=[600/clight,15100/clight])

#Compute FKP weights
if weight_fkp :
    rb=z_b/h0
    dcth=np.amax(np.cos(th_r))-np.amin(np.cos(th_r)); dphi=np.amax(phi_r)-np.amin(phi_r);
    fsky=dcth*dphi/(4*np.pi); nfrac=(len(data_d)+0.0)/len(data_r);
    vol=4*np.pi*fsky*np.array([(rb[i+1]**3-rb[i]**3)/3 for i in np.arange(len(h_z))])

    n0=0.23; gamma=0.99; z0=0.0104;
    def nz_fit(z) :
        return n0*np.exp(-(z/z0)**gamma)
    print 'Typical number density (Mpc/h)^-3 :',nz_fit(0.02)
    print 'Average number density (Mpc/h)^-3 :',(len(data_d)+0.)/(4*np.pi*fsky*(np.amax(r_d)**3-
                                                                                np.amin(r_d)**3)/3)
    zarr=(z_b[1:]+z_b[:-1])/2
    if plot_stuff :
        plt.plot(zarr,hd_z/vol);
        plt.plot(zarr,nfrac*h_z/vol);
        plt.plot(zarr,nz_fit(zarr))
        plt.yscale('log')
        plt.xlim('$z$',fontsize=16)
        plt.ylim('$n(z)\\,[({\\rm Mpc/h})^{-3}]$',fontsize=16)
        plt.show()

    j30=2962*0.7**3
    w_fkp_d=1./(1+4*np.pi*nz_fit(z_d)*j30)
    w_fkp_r=1./(1+4*np.pi*nz_fit(z_r)*j30)
    print 'J(r<30 Mpc/h) : ',j30
    if plot_stuff :
        plt.plot(zarr,1./(1+4*np.pi*nz_fit(zarr)*j30));
        plt.yscale('log')
        plt.xlabel('$z$',fontsize=16)
        plt.ylabel('$w(z)$',fontsize=16)
        plt.show()
else :
    w_fkp_d=np.ones_like(z_d)
    w_fkp_r=np.ones_like(z_r)
print 'Typical FKP weights :',[np.mean(w_fkp_d), np.mean(w_fkp_r)]

#Compute mass weights
if weight_mhi :
    w_mhi_d=10.**(data_d['logMHI']-np.mean(data_d['logMHI']))
else :
    w_mhi_d=np.ones_like(w_fkp_d)
w_mhi_r=np.ones_like(w_fkp_r)
print 'Typical mass weights :',[np.mean(w_mhi_d), np.mean(w_mhi_r)]

w_d=w_fkp_d*w_mhi_d
w_r=w_fkp_r*w_mhi_r
print 'Typical total weights :',[np.mean(w_d), np.mean(w_r)]

#Run CUTE
np.savetxt("data.txt",np.transpose([z_d,np.cos(th_d),phi_d,w_d]))
np.savetxt("random.txt",np.transpose([z_r,np.cos(th_r),phi_r,w_r]))
stout= "data_filename= data.txt\n"
stout+="random_filename= random.txt\n"
stout+="input_format= 0\n"
stout+="output_filename= corr_3D_ps_highres_nbin60_fkp.txt\n"
stout+="corr_type= 3D_ps\n"
stout+="omega_M= 0.315\n"
stout+="omega_L= 0.685\n"
stout+="w= -1.\n"
stout+="log_bin= 1\n"
stout+="dim1_min_logbin= 0.1\n"
stout+="dim1_max= 20.\n"
stout+="dim1_nbin= 20\n"
stout+="dim2_max= 60.\n"
stout+="dim2_nbin= 60\n"
stout+="dim3_max= 20.\n"
stout+="dim3_min= 0.\n"
stout+="dim3_nbin= 20\n"
f=open("param_cute_highres_nbin60_fkp.ini","w")
f.write(stout)
f.close()
os.system("./CUTEdir/CUTE/CUTE param_cute_highres_nbin60_fkp.ini > log_cute_highres_nbin60_fkp.txt")
os.system("rm data.txt random.txt")
os.system("cat log_cute_highres_nbin60_fkp.txt")

plt.figure()
r,xi,exi,dd,dr,rr=np.loadtxt("corr.txt",unpack=True)
plt.plot(r,xi,'k-')
plt.xlabel('$r$',fontsize=16);
plt.ylabel('$\\xi(r)$',fontsize=16);
plt.xscale('log'); plt.yscale('log');
plt.show()

exit(1)

xarr=(bins[1:]+bins[:-1])/2
hh=h.copy()
indgap=np.where((xarr>0.028) & (xarr<0.031))[0]
hh[indgap]=27.5-400.*(xarr-0.028)[indgap]
print np.shape(bins),np.shape(h),np.shape(p)
plt.plot(xarr,h)
plt.plot(xarr,hh)
plt.show()
plt.plot(xarr,hh/xarr**2)
plt.yscale('log')
plt.show()
