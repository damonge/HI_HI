import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from swml import mfunc_analysis,det_test_code1,det_test_code12
import os

np.random.seed(6789)

#Boxes defining each region in the a.100 sample (first interval is dec, second is RA)
boxes_spring=np.array([[[ 0.,4.],[7.7,16.5]],
                       [[ 4.,8.],[7.7,16.5]],
                       [[ 8.,12.],[7.7,16.5]],
                       [[ 12.,16.],[7.7,16.5]],
                       [[16.,18.],[7.7,16. ]],
                       [[18.,20.],[8.7,15.4]],
                       [[20.,24.],[9.4,15.4]],
                       [[24.,30.],[7.6,16.5]],
                       [[30.,32.],[8.5,16.0]],
                       [[32.,36.],[9.5,15.5]]])
boxes_fall=np.array([[[ 0., 2.],[22.0,3.0]],
                     [[ 2., 6.],[22.5,3.0]],
                     [[ 6.,10.],[22.0,3.0]],
                     [[10.,14.],[22.0,2.5]],
                     [[14.,20.],[22.0,3.0]],
                     [[20.,26.],[22.0,3.0]],
                     [[26.,32.],[22.0,3.0]],
                     [[32.,36.],[22.0,3.0]]])
#Boxes for group jackknives
boxes_jk_grp=np.array([[[-1., 8.],[ 7.5,12.1]],
                       [[-1., 8.],[12.1,17.0]],
                       [[ 8.,16.],[ 7.5,12.1]],
                       [[ 8.,16.],[12.1,17.0]],
                       [[16.,20.],[ 7.5,17.0]],
                       [[20.,24.],[ 7.5,17.0]],
                       [[24.,31.],[ 7.5,12.1]],
                       [[24.,31.],[12.1,17.0]],
                       [[-1., 2.],[-2.5, 3.5]],
                       [[12.,18.],[-2.5, 3.5]]])

def get_box_area(b) :
    '''
    Returns box area in sterad
    '''
    decint=b[0]
    raint=b[1]*15.
    if raint[0]>raint[1] :
        raint[0]-=360
    phi0=raint[0]*np.pi/180
    phif=raint[1]*np.pi/180
    cth0=np.cos((90-decint[0])*np.pi/180)
    cthf=np.cos((90-decint[1])*np.pi/180)
    return decint,raint,np.fabs((phif-phi0)*(cthf-cth0))
    
def survey_area(boxes_all=[boxes_spring,boxes_fall]) :
    '''
    Returns total area for a set of boxes
    '''
    v=0
    for b in np.concatenate(boxes_all) :
        dint,rint,area_here=get_box_area(b)
        v+=area_here

    return v

def get_sub_boxes(n_subboxes,boxes_all=[boxes_spring,boxes_fall]) :
    '''
    Divide a set of boxes into a larger set of boxes by subsampling
    n_subboxes = Number of subboxes to generate
    '''
    area=survey_area(boxes_all)
    area_subbox=area/n_subboxes
    subboxes=[]
    for b in np.concatenate(boxes_all) :
        decint,raint,area_here=get_box_area(b)
        nboxes_here=max(int(area_here/area_subbox),1)
        dra=(raint[1]-raint[0])/nboxes_here
        for i in np.arange(nboxes_here) :
            subboxes.append([[b[0,0],b[0,1]],[(raint[0]+i*dra)/15.,(raint[0]+(i+1)*dra)/15.]])
    return np.array(subboxes)

def in_boxes(data_ra,data_dec,boxes_all=[boxes_spring,boxes_fall]) :
    '''
    Determine whether a given point is inside a set of boxes.

    data_ra = array of right ascensions
    data_dec = array of declinations
    boxes_all = set of boxes determining the survey footprint

    return -> boolean array selecting all objects within footprint
    '''
    inrange=np.zeros(len(data_ra),dtype=bool)
    for b in np.concatenate(boxes_all) :
        decint=b[0]
        raint=b[1]*15.
        if raint[0]>raint[1] :
            raint[0]-=360
        indec=(decint[0]<=data_dec) &(data_dec<=decint[1])
        inra=(raint[0]<=data_ra) &(data_ra<=raint[1])
        inrange+=inra*indec
    return inrange

def mf_group(grps,dcode=2,verbose=0) :
    '''
    Generate mass function measurements for galaxies in a set of groups.

    grps = groups. Each group contains group information and information about all galaxies in it.
    dcode = detection code for the sources you want to take into account
    verbose = output level (0-nothing, 1-main steps, >1-detailed)
    '''

    #Generate array with all sources
    d_full=np.concatenate([g['gals'] for g in grps])

    print ' %d groups, %d HI sources'%(len(grps),len(d_full))
    
    #Select by detection code
    good_det=d_full['detcode']<=dcode
    #High redshift cut
    good_vel=d_full['vcmb']<=15000.
    #Full set of cuts
    good_gals=good_vel*good_det
    #Total survey volume (sum of group volumes) including volume correction
    vsurvey=np.sum([g['area']*4*np.pi*(g['r180L']/0.7)**3/3 for g in grps])
    #Total volume occupied by the groups (without volume correction)
    vgroups=np.sum([4*np.pi*(g['r180L']/0.7)**3/3 for g in grps])

    #Edges for the mass function calculation
    wmin=np.log10(np.amin(d_full['w50'])); wmax=np.log10(np.amax(d_full['w50'])); dw=0.0*(wmax-wmin)
    w_range=np.array([wmin-dw,wmax+dw]); w_nbins=5
    mmin=np.amin(d_full['loghimass']); mmax=np.amax(d_full['loghimass']); dm=0.0*(mmax-mmin)
    m_range=np.array([mmin-dm,mmax+dm]); m_nbins=5
    #Avoid bins with just one galaxy
    found=False
    while not found :
        h,b=np.histogram(d_full['loghimass'],range=m_range,bins=m_nbins)
        if h[0]>1 :
            found=True
        else :
            m_range[0]=b[1]

    #Generate jackknives
    n_jk=len(boxes_jk_grp)
    good_gals_jk=[]; vsurvey_jk=[];
    for i in np.arange(n_jk) :
        mask=np.arange(n_jk)!=i
        good_range_jk=in_boxes(d_full['HIra'],d_full['HIdec'],[boxes_jk_grp[mask]])
        good_gals_jk.append(good_range_jk*good_vel*good_det)
        vsurvey_jk.append(vsurvey*sum(good_range_jk)/(len(d_full)+0.))

    #Get mass function
    res_grps=mfunc_analysis(d_full['loghimass'],d_full['w50'],d_full['dist'],
                            d_full['flux'],d_full['fluxerr'],0*d_full['werr'],d_full['disterr'],
                            good_gals,vsurvey,good_gals_jk,vsurvey_jk,nsims_ms=100,dcode=dcode,
                            m_range=m_range,m_nbins=m_nbins,m_rs=10,
                            w_range=w_range,w_nbins=w_nbins,w_rs=10,
                            verbose=verbose)

    return res_grps

def mf_all(d,dcode=1,verbose=1) :
    '''
    Generate mass function measurements for the alfa.100 sample

    d = list of sources
    dcode = detection code for the sources you want to take into account
    verbose = output level (0-nothing, 1-main steps, >1-detailed)
    '''

    #Select by detection code
    good_det=d['detcode']<=dcode
    #High redshift cut
    good_vel=d['vcmb']<=15000.

    #Get sub-boxes for the jackknives
    sbb=get_sub_boxes(53)
    n_jk=len(sbb)

    #Select galaxies in footprint
    good_range=in_boxes(d['HIra'],d['HIdec'])
    #Full set of cuts
    good_gals=good_vel*good_det*good_range
    ngals_good=np.sum(good_gals)
    rmin=np.amin(d['dist'][good_gals]); rmax=np.amax(d['dist'][good_gals]);
    #Total survey volume
    v_survey=survey_area()*(rmax**3-rmin**3)/3.

    #Generate jackknives
    good_gals_jk=[]; v_survey_jk=[];
    if verbose>1 :
        print sum(good_gals),v_survey
    for i in np.arange(n_jk) :
        mask=np.arange(n_jk)!=i
        good_range_jk=in_boxes(d['HIra'],d['HIdec'],[sbb[mask]])
        good_gals_jk.append(good_vel*good_det*good_range_jk)
        rmin=np.amin(d['dist'][good_gals_jk[i]]); rmax=np.amax(d['dist'][good_gals_jk[i]]);
        v_survey_jk.append(survey_area([sbb[mask]])*(rmax**3-rmin**3)/3.)
        if verbose>1 :
            print sum(good_gals_jk[i]),v_survey_jk[i]

    #Get mass function
    res_a100=mfunc_analysis(d['loghimass'],d['w50'],d['dist'],
                            d['flux'],d['fluxerr'],0*d['werr'],d['disterr'],
                            good_gals,v_survey,good_gals_jk,
                            v_survey_jk,nsims_ms=100,m_rs=10,w_rs=10,
                            dcode=dcode,verbose=verbose)

    return res_a100

#Run parameters:
nbins_Mh=7 #Number of halo mass bins
mh_type='L' #Halo mass estimate (L or M)
bin_type='lin' #Use linear M_h bins ('lin')? Or equal-number M_h bins ('eq')?
compute_full_mf=False #Compute also full sample HIMF?

#Read full dataset
data=(fits.open("data_70/data_a100.fits"))[1].data
data['HIra'][data['HIra']>300]-=360. #Wrap right ascension around longitude 0 to avoid cut
goodid=np.where(data['flux']>0)[0]; data=data[goodid]

#Read group data
data_groups=np.genfromtxt('data_70/a70_HI_group_mh12.5_ngal2.dat',names=True,dtype=None)

#Read group area corrections
group_areas=np.genfromtxt('data_70/area_correct_a70_group.dat',names=True,dtype=None)
group_areas['area'][group_areas['area']==0]=1.

#Plot sky distributions
sbb=get_sub_boxes(53)
plt.figure()
plt.title('Sky distribution of sources and jackknives')
raarr=data['HIra'].copy(); raarr[raarr>300]-=360; raarr/=15.; decarr=data['HIdec'].copy();
plt.plot(raarr,decarr,'.',markersize=2,c='#AAAAAA',label='All $\\alpha$.100 sources')
for b in sbb :
    plt.plot([b[1,0],b[1,1]],[b[0,0],b[0,0]],'k-')
    plt.plot([b[1,1],b[1,1]],[b[0,0],b[0,1]],'k-')
    plt.plot([b[1,1],b[1,0]],[b[0,1],b[0,1]],'k-')
    plt.plot([b[1,0],b[1,0]],[b[0,1],b[0,0]],'k-')
for b in boxes_jk_grp :
    plt.plot([b[1,0],b[1,1]],[b[0,0],b[0,0]],'r--')
    plt.plot([b[1,1],b[1,1]],[b[0,0],b[0,1]],'r--')
    plt.plot([b[1,1],b[1,0]],[b[0,1],b[0,1]],'r--')
    plt.plot([b[1,0],b[1,0]],[b[0,1],b[0,0]],'r--')
raarr=data_groups['HIra'].copy(); raarr[raarr>300]-=360; raarr/=15.; decarr=data_groups['HIdec'].copy();
plt.plot(raarr,decarr,'.',markersize=2,c='b',label='Sources in groups')
plt.plot([-100,-100],[-100,-100],'k-',label='JK regions in full survey')
plt.plot([-100,-100],[-100,-100],'r--',label='JK regions in groups')
plt.ylim([-1,37]); plt.ylabel('dec. [deg]',fontsize=14)
plt.xlim([-5,21]); plt.xlabel('R.A. [hr]',fontsize=14)
plt.legend(loc='upper left')

groups=[] #Will contain details for all groups
mMarr=[] #Will contain group stellar-mass-based masses
mLarr=[] #Will contain group luminosity-based masses
ngarr=[] #Will contain number of sources in each group
r180Larr=[] #Will contain group radii
vcmbarr=[] #Will contain maximum velocity (CMB) of galaxies in group
for n,i_d in enumerate(np.unique(data_groups['groupid'])) :
    index_groups=np.where(data_groups['groupid']==i_d)[0]
    if len(np.unique(data_groups['Mhalo_M'][index_groups]))!=1 : #Check mass is unique for each group
        print "SHIT"
        exit(1)
    id_area=np.where(group_areas['group_id']==i_d)[0]
    if len(id_area)!=1 :
        print "SHIT"
        exit(1)
    area=group_areas['area'][id_area[0]]
    mL=data_groups['Mhalo_L'][index_groups[0]]
    mM=data_groups['Mhalo_M'][index_groups[0]]
    r180L=data_groups['r180L'][index_groups[0]]
    #Find galaxies in this group by matching their AGCNr
    hiids=data_groups['AGCNr'][index_groups]
    id_gals=np.in1d(data['AGCNr'],hiids)
    vcmb=np.amax(data['vcmb'][id_gals])
    groups.append({'mL':mL,'mM':mM,'r180L':r180L,'ng':len(hiids),'area':area,
                   'vcmb':vcmb,'gals':data[id_gals]})
    mMarr.append(mM)
    mLarr.append(mL)
    r180Larr.append(r180L)
    vcmbarr.append(vcmb)
    ngarr.append(len(hiids))
mMarr=np.array(mMarr)
mLarr=np.array(mLarr)
vcmbarr=np.array(vcmbarr)
r180Larr=np.array(r180Larr)
groups=np.array(groups)
ngarr=np.array(ngarr)

#Retain only halos with galaxies within velocity limit
ids_goodv=np.where(vcmbarr<15000.)[0] 
mMarr=mMarr[ids_goodv]
mLarr=mLarr[ids_goodv]
vcmbarr=vcmbarr[ids_goodv]
r180Larr=r180Larr[ids_goodv]
groups=groups[ids_goodv]
ngarr=ngarr[ids_goodv]

#Use mL or mM?
if mh_type=='L' :
    m_choice=mLarr
elif mh_type=='M' :
    m_choice=mMarr
else :
    raise ValueError('Unknown halo mass type '+m_choice)

def get_bin_edges(marr,weight,nbins) :
    '''
    Splits array marr into nbins bins with roughly equal weight
    '''
    ntot=np.sum(weight)
    n_perbin=(ntot+0.)/nbins
    mmin=np.min(marr)
    mmax=np.max(marr)
    hist,mbins=np.histogram(marr,range=[mmin,mmax],bins=1000,weights=weight)
    cumhist=np.cumsum(hist)
    medges=[mbins[0]]
    for i in np.arange(nbins) :
        medges.append(mbins[np.where(cumhist<(i+1)*n_perbin)[0][-1]+1])
    medges=np.array(medges)
    medges[-1]=mmax
    return medges

if bin_type=='lin' :
    #Option 1: use pre-set bins (linear in log10 M)
    mh_bin_edges=np.linspace(np.min(m_choice),np.max(m_choice),nbins_Mh+1)
elif bin_type=='eq' :
    #Option 2: use constant-weight bins
    mh_bin_edges=get_bin_edges(m_choice,ngarr,nbins_Mh)
else :
    raise ValueError('Unknown binning type '+bin_type)
print 'Mass bins: ',mh_bin_edges

#Split groups into mass bins and compute mass functions
grps_mf=[]
for i in range(nbins_Mh) :
    group_ids=np.where((m_choice>mh_bin_edges[i]) & (m_choice<=mh_bin_edges[i+1]))[0]
    print "Computing mass function in %d-th Mh bin"%(i+1)
    grps_mf.append(mf_group(groups[group_ids]))

#Save results
os.system('mkdir -p results_himf')
for i in range(nbins_Mh) :
    np.savez('results_himf/himf_%dMh'%nbins_Mh+mh_type+'_bin'+bin_type+'_bin%d'%(i+1),
             mh_range=mh_bin_edges[i:i+2],
             m_bins=grps_mf[i]['m_bins'],m_cent=grps_mf[i]['m_cent'],phi1d=grps_mf[i]['phi1d'],
             err1d_tot=grps_mf[i]['err1d_tot'],err1d_ps=grps_mf[i]['err1d_ps'],
             err1d_jk=grps_mf[i]['err1d_jk'],err1d_ms=grps_mf[i]['err1d_ms'])

#Plot results
plt.figure()
cols=['r','g','b','y','c','m','k']
for i in range(nbins_Mh) :
    #plt.plot(grps_mf[i]['m_cent'],grps_mf[i]['err1d_ps'],cols[i]+'-')
    #plt.plot(grps_mf[i]['m_cent'],grps_mf[i]['err1d_jk'],cols[i]+'--')
    plt.errorbar(grps_mf[i]['m_cent'],grps_mf[i]['phi1d'],yerr=grps_mf[i]['err1d_tot'],fmt='.',
                 label='$%.1lf<\\log\,M_h<%.1lf$'%(mh_bin_edges[i],mh_bin_edges[i+1]))
plt.yscale('log',nonposy='clip')
plt.ylim([3E-4,8]); plt.ylabel('$\\phi(M_{\\rm HI})\\,[{\\rm Mpc}^{-3}\\,{\\rm dex}^{-1}]$',fontsize=14)
plt.xlim([8.5,11]); plt.xlabel('$\\log_{10}\\,M_{\\rm HI}/M_{\\odot}$',fontsize=14)
plt.legend(loc='lower left',ncol=2)

if compute_full_mf :
    #Compute HIMF in full survey
    print 'Computing HIMF for full alpha sample'
    print ' det_code=1'
    a100_1=mf_all(data,dcode=1,verbose=1) #detcode=1
    print ' det_code=1,2'
    a100_2=mf_all(data,dcode=2,verbose=1) #detcode=1,2
    
    #Save results
    np.savez('results_himf/himf_a100_dc1',
             m_bins=a100_1['m_bins'],m_cent=a100_1['m_cent'],phi1d=a100_1['phi1d'],
             err1d_tot=a100_1['err1d_tot'],err1d_ps=a100_1['err1d_ps'],
             err1d_jk=a100_1['err1d_jk'],err1d_ms=a100_1['err1d_ms'])
    np.savez('results_himf/himf_a100_dc2',
             m_bins=a100_2['m_bins'],m_cent=a100_2['m_cent'],phi1d=a100_2['phi1d'],
             err1d_tot=a100_2['err1d_tot'],err1d_ps=a100_2['err1d_ps'],
             err1d_jk=a100_2['err1d_jk'],err1d_ms=a100_2['err1d_ms'])
    
    plt.figure()
    plt.errorbar(a100_1['m_cent'],a100_1['phi1d'],yerr=a100_1['err1d_tot'],fmt='ro',label='det_code=1')
    plt.errorbar(a100_2['m_cent'],a100_2['phi1d'],yerr=a100_2['err1d_tot'],fmt='bo',label='det_code=1,2')
    plt.yscale('log',nonposy='clip')
    plt.ylabel('$\\phi(M_{\\rm HI})\\,[{\\rm Mpc}^{-3}\\,{\\rm dex}^{-1}]$',fontsize=14)
    plt.xlabel('$\\log_{10}\\,M_{\\rm HI}/M_{\\odot}$',fontsize=14)
    plt.legend(loc='lower left')
plt.show()
