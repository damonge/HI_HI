import numpy as np
import pyfits as pf

data_d=np.genfromtxt('data/data_cat_mhithresh7.5.csv',skiprows=1,delimiter=',',
                     dtype=(int,float,float,float,float,float,float),
                     names=('agc','ra','dec','v50','verr','vcmb','logMHI'))
data_r=np.genfromtxt('data/rand_cat_mhithresh7.5.csv',skiprows=1,delimiter=',',
                     dtype=(float,float,float,float),names=('ra','dec','v50','vcmb'))

th=(90-data_d['dec'])*np.pi/180
phi=data_d['ra']*np.pi/180
u=np.array([np.sin(th)*np.cos(phi),np.sin(th)*np.sin(phi),np.cos(th)])
vsol=np.dot(np.linalg.inv(np.dot(u,np.transpose(u))),np.dot(u,data_d['v50']-data_d['vcmb']))
vcmb=data_d['v50']-np.dot(np.transpose(u),vsol)

data_d70=np.genfromtxt('data_70/a70_160624.csv',skiprows=1,delimiter=',',
                       dtype=None,
                       names=('AGCNr','Name','HIra','HIdec','OCra','OCdec','v21','w50','werr','flux','fluxerr','snratio','rms','dist','loghimass','detcode'))
th=(90-data_d70['HIdec'])*np.pi/180
phi=data_d70['HIra']*np.pi/180
u=np.array([np.sin(th)*np.cos(phi),np.sin(th)*np.sin(phi),np.cos(th)])
vcmb=data_d70['v21']-np.dot(np.transpose(u),vsol)
tbhdu=pf.new_table([pf.Column(name='AGCNr'    ,format='K',array=data_d70['AGCNr']),
                    pf.Column(name='HIra'     ,format='D',array=data_d70['HIra']),
                    pf.Column(name='HIdec'    ,format='D',array=data_d70['HIdec']),
                    pf.Column(name='v21'      ,format='D',array=data_d70['v21']),
                    pf.Column(name='vcmb'     ,format='D',array=vcmb),
                    pf.Column(name='w50'      ,format='D',array=data_d70['w50']),
                    pf.Column(name='werr'     ,format='D',array=data_d70['werr']),
                    pf.Column(name='flux'     ,format='D',array=data_d70['flux']),
                    pf.Column(name='fluxerr'  ,format='D',array=data_d70['fluxerr']),
                    pf.Column(name='snratio'  ,format='D',array=data_d70['snratio']),
                    pf.Column(name='rms'      ,format='D',array=data_d70['rms']),
                    pf.Column(name='dist'     ,format='D',array=data_d70['dist']),
                    pf.Column(name='loghimass',format='D',array=data_d70['loghimass']),
                    pf.Column(name='detcode'  ,format='K',array=data_d70['detcode'])])
tbhdu.writeto('data_70/data_a70.fits')


tbhdu=pf.new_table([pf.Column(name='agc',format='K',array=data_d['agc']),
                    pf.Column(name='ra',format='D',array=data_d['ra']),
                    pf.Column(name='dec',format='D',array=data_d['dec']),
                    pf.Column(name='v50',format='D',array=data_d['v50']),
                    pf.Column(name='verr',format='D',array=data_d['verr']),
                    pf.Column(name='vcmb',format='D',array=data_d['vcmb']),
                    pf.Column(name='logMHI',format='D',array=data_d['logMHI'])])
tbhdu.writeto('data/data_cat_mhithresh7.5.fits')

tbhdu=pf.new_table([pf.Column(name='ra',format='D',array=data_r['ra']),
                    pf.Column(name='dec',format='D',array=data_r['dec']),
                    pf.Column(name='v50',format='D',array=data_r['v50']),
                    pf.Column(name='vcmb',format='D',array=data_r['vcmb'])])
tbhdu.writeto('data/rand_cat_mhithresh7.5.fits')


'''
names_a40=['AGCNr','Name','RAdeg_HI','Decdeg_HI','RAdeg_OC','DECdeg_OC','Vhelio','W50','errW50','HIflux','errflux','SNR','RMS','Dist','logMsun','HIcode','OCcode','NoteFlag']
dtype_a40=[int,'|S10',float,float,float,float,float,float,float,float,float,float,float,float,float,int,'|S10','|S10']
data_a40=np.genfromtxt("data/a40.datafile1.csv",skiprows=1,delimiter=',',names=names_a40,dtype=dtype_a40)
print data_a40['Name']
print data_a40['OCcode']
print data_a40['NoteFlag']

names_a70=['AGCNr','Name','HIra','HIdec','OCra','OCdec','v21','w50','werr','flux','fluxerr','snratio','rms','dist','loghimass','detcode']
dtype_a70=[int,'|S10',float,float,float,float,float,float,float,float,float,float,float,float,float,int]
data_a70=np.genfromtxt("data/a70_151111.csv",skiprows=1,delimiter=',',names=names_a70,dtype=dtype_a70)
print data_a70['Name']
print data_a70['detcode']
print data_a70['dist']

names_cor=['agc','ra','dec','v50','verr','vcmb','logMHI']
dtype_cor=[int,float,float,float,float,float,float]
data_cor=np.genfromtxt("data/data_cat_mhithresh7.5.csv",skiprows=1,delimiter=',',names=names_cor,dtype=dtype_cor)
print data_cor['agc']
print data_cor['v50']
print data_cor['vcmb']
'''
