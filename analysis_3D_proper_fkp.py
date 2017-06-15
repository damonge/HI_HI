import numpy as np
import matplotlib.pyplot as plt
import os

fname_data="data/data_cat_mhithresh7.5.csv"
fname_rand="data/rand_cat_mhithresh7.5.csv"
dtor=np.pi/180
clight=299792.458
plot_stuff=False
weight_mhi=False
weight_fkp=True

h0=0.000333564095 

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

#Compute redshift
z_d=data_d['vcmb']/clight; r_d=z_d/h0
z_r=data_r['vcmb']/clight; r_r=z_r/h0
h_z,z_b=np.histogram(z_r,bins=100,range=[600/clight,15100/clight])
hd_z,dum1=np.histogram(z_d,bins=100,range=[600/clight,15100/clight])

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
zarr = (z_b[1:]+z_b[:-1])/2
rarr = (rb[1:]+rb[:-1])/2.
w_d = np.ones_like(z_d)
w_r = np.ones_like(z_r)

np.savetxt("data.txt",np.transpose([z_d,np.cos(th_d),phi_d,w_d]))
np.savetxt("random.txt",np.transpose([z_r,np.cos(th_r),phi_r,w_r]))

# np.savetxt('nr_data.txt',np.transpose([rarr,hd_z/vol]))
# np.savetxt('nr_random.txt',np.transpose([rarr,nfrac*h_z/vol]))
np.savetxt('nr_fit.txt',np.transpose([rarr,nz_fit(zarr)]))

# plt.plot(rarr,hd_z/vol, label = 'data')
# plt.plot(rarr,nfrac*h_z/vol, label = 'random')
# plt.plot(rarr,nz_fit(zarr),label = 'fit')
# plt.yscale('log')
# plt.xlabel('$r\,[h^{-1}\,{\\rm Mpc}]$',fontsize=20)
# plt.ylabel('$n(r)\\,[({\\rm Mpc/h})^{-3}]$',fontsize=20)
# plt.legend(loc=0)
# plt.savefig('nr.pdf')

#Run CUTE
stout= "data_filename= data.txt\n"
stout+="random_filename= random.txt\n"
stout+="input_format= 0\n"
stout+="output_filename= corr_3D_ps_nbin60_proper_fkp_nfit_1p51_3p3.txt\n"
stout+="omega_M= 0.315\n"
stout+="omega_L= 0.685\n"
stout+="w= -1.\n"
stout+="corr_type= 3D_ps\n"
stout+="log_bin= 1\n"
stout+="dim1_min_logbin= 0.1\n"
stout+="dim1_max= 20.\n"
stout+="dim1_nbin= 20\n"
stout+="dim2_max= 60.\n"
stout+="dim2_nbin= 60\n"
stout+="dim3_max= 20.\n"
stout+="dim3_min= 0.\n"
stout+="dim3_nbin= 20\n"
stout+="do_j3= 1\n"
stout+="j3_gamma= 1.51\n"
stout+="j3_r0= 3.3\n"
stout+="j3_ndens_file= nr_fit.txt\n"
f=open("param_cute_highres_nbin60_proper_fkp_nfit_1p51_3p3.ini","w")
f.write(stout)
f.close()
os.system("./CUTEdir/CUTE/CUTE param_cute_highres_nbin60_proper_fkp_nfit_1p51_3p3.ini > log_cute_highres_nbin60_proper_fkp_nfit_1p51_3p3.txt")
os.system("rm data.txt random.txt")
os.system("cat log_cute_highres_nbin60_proper_fkp_nfit_1p51_3p3.txt")
