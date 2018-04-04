import numpy as np
import matplotlib.pyplot as plt

def det_test_code1(m,w,D):
    '''
    Returns array of 1 or 0 for when the galaxies given are above or below the detection threshold.
    
    m = log10 HI mass in Msol (array)
    w = log10 W50 in km/s (array)
    D = distance in Mpc (array)
    '''
    
    return np.array(np.greater(m,np.maximum(0.5*w-1.207,w-2.457)+np.log10((D**2.)*235600.)),dtype='float')

def det_test_code12(m,w,D):
    '''
    Returns array of 1 or 0 for when the galaxies given are above or below the detection threshold.
    
    m = log10 HI mass in Msol (array)
    w = log10 W50 in km/s (array)
    D = distance in Mpc (array)
    '''
    
    return np.array(np.greater(m,np.maximum(0.5*w-1.240,w-2.490)+np.log10((D**2.)*235600.)),dtype='float')

def get_mfunc(m_arr,w_arr,d_arr,v_survey,
              m_range=[6.,11.],m_nbins=25,m_rs=10,
              w_range=[1.2,3.],w_nbins=18,w_rs=10,
              niter=100,tol=0.01,dcode=1) :
    '''
    Computes 2DSWML function for a set of objects.

    m_arr = log10 HI mass for each object in Msol (array)
    w_arr = log10 W50 in km/s for each object (array)
    d_arr = distance for each object in Mpc (array)
    v_survey = survey volume (in Mpc^3) to use when normalizing the mass function
    m_range = range in m over which to compute the 2DSWML
    m_nbins = number of bins in m to use
    m_rs = number of point to resample each cell in the m direction when computing H_ijk
    w_range = range in w over which to compute the 2DSWML
    w_nbins = number of bins in w to use
    w_rs = number of point to resample each cell in the w direction when computing H_ijk
    niter = maximum number of iterations to use in the 2DSWML
    tol = tolerance for convergence of the 2DSWML iterative process
    dcode = detection code of the objects used (needed to determine completeness function)
    '''
    
    warning=None
    if dcode==1 :
        fcomp=det_test_code1
    else :
        fcomp=det_test_code12

    #Use only sources above completeness limit
    good_comp=fcomp(m_arr,w_arr,d_arr)>0
    m_arr_use=m_arr[good_comp]
    w_arr_use=w_arr[good_comp]
    d_arr_use=d_arr[good_comp]

    #Number counts and bin edges
    ncounts2d,m_bins,w_bins=np.histogram2d(m_arr_use,w_arr_use,range=[m_range,w_range],
                                           bins=[m_nbins,w_nbins])
    ncounts1d=np.sum(ncounts2d,axis=1)+0.
    ncounts2d=ncounts2d.flatten()+0.
    m_delt=m_bins[1]-m_bins[0]
    w_delt=w_bins[1]-w_bins[0]

    #Compute H_ijk. To compute completeness integral within each cell we resample them by factors
    # w_rs and m_rs
    w_res=w_bins[0]+(w_bins[-1]-w_bins[0])*(np.arange(w_nbins*w_rs)+0.5)/(w_nbins*w_rs)
    m_res=m_bins[0]+(m_bins[-1]-m_bins[0])*(np.arange(m_nbins*m_rs)+0.5)/(m_nbins*m_rs)
    h3d_rs=fcomp(m_res[:,None,None]*np.ones(w_nbins*w_rs)[None,:,None],
                 w_res[None,:,None]*np.ones(m_nbins*m_rs)[:,None,None],d_arr_use[None,:])
    h3d=np.mean(h3d_rs.reshape([m_nbins,m_rs,w_nbins,w_rs,len(d_arr_use)]),
                axis=(1,3)).reshape([m_nbins*w_nbins,len(d_arr_use)])

    #Start iterative process
    resid=1.; i_iter=0;
    phi2d=np.ones([m_nbins,w_nbins]).flatten()
    while (resid>tol) and (i_iter<niter) :
        denom0=np.sum(h3d[:,:]*phi2d[:,None],axis=0) #Denominator's denominator
        denom=np.sum(h3d/denom0[None,:],axis=1) #Denominator
        denom=np.where(denom==0,np.inf,denom) #Protect against division by zero
        res=ncounts2d/denom-phi2d
        resid=np.sqrt(np.mean((res/np.mean(phi2d))**2)) #Residual to compare with tolerance
        phi2d+=res #Update phi2d and i_iter
        i_iter+=1

    if i_iter==niter :
        warning="Exceeded number of iterations"

    #Normalize
    norm_denom=np.sum(h3d[:,:]*phi2d[:,None],axis=0)*m_delt*w_delt
    norm=np.sum(1./norm_denom)/v_survey
    phi2d*=norm
    phi2d=phi2d.reshape([m_nbins,w_nbins])
    phi1d=np.sum(phi2d,axis=1)*w_delt #Marginalize over W50

    #Compute poisson errors
    err2d=np.zeros_like(phi2d).flatten();
    err2d[ncounts2d>0]=phi2d.flatten()[ncounts2d>0]/np.sqrt(ncounts2d[ncounts2d>0]);
    err2d=err2d.reshape([m_nbins,w_nbins])
    err1d=np.zeros_like(phi1d);
    err1d[ncounts1d>0]=phi1d[ncounts1d>0]/np.sqrt(ncounts1d[ncounts1d>0])

    return m_bins,phi1d,err1d,w_bins,phi2d,err2d,warning

def mfunc_analysis(lm_arr,w5_arr,ds_arr,fl_arr,fle_arr,w5e_arr,dse_arr,
                   mask_all,v_survey_all,masks_jk,v_survey_jk,nsims_ms=100,
                   m_range=[6.,11.],m_nbins=25,m_rs=10,
                   w_range=[1.2,3.],w_nbins=18,w_rs=10,
                   niter_mf=100,tol_mf=0.01,dcode=1,verbose=1) :
    '''
    Computes 2DSWML function and its error bars for a set of objects.
    
    lm_arr = log10 HI mass for each object in Msol (array)
    w5_arr = W50 in km/s for each object (array)
    ds_arr = distance for each object in Mpc (array)
    mask_all = boolean array determining which objects to use
    v_survey_all = survey volume (in Mpc^3) to use when normalizing the mass function
    masks_jk = array of boolean arrays determining which objects to use in each jackknife region
    v_survey_jk = array of survey volumes (in Mpc^3) to use when normalizing the mass function for each jackknife region
    nsims_ms = number of simulations of measurement errors to use when computing the associated errors
    m_range = range in m over which to compute the 2DSWML
    m_nbins = number of bins in m to use
    m_rs = number of point to resample each cell in the m direction when computing H_ijk
    w_range = range in w over which to compute the 2DSWML
    w_nbins = number of bins in w to use
    w_rs = number of point to resample each cell in the w direction when computing H_ijk
    niter = maximum number of iterations to use in the 2DSWML
    tol = tolerance for convergence of the 2DSWML iterative process
    dcode = detection code of the objects used (needed to determine completeness function)
    verbose = output level (0-nothing, 1-main steps, >1-detailed
    '''

    if verbose>0 :
        print "  Computing fiducial"
    ngals_good=sum(mask_all)
    mbns,phi1d,e1d_ps,wbns,ph2d,e2d_ps,wrn=get_mfunc(lm_arr[mask_all],
                                                     np.log10(w5_arr[mask_all]),
                                                     ds_arr[mask_all],
                                                     v_survey_all,
                                                     m_range=m_range,m_nbins=m_nbins,m_rs=m_rs,
                                                     w_range=w_range,w_nbins=w_nbins,w_rs=w_rs,
                                                     niter=niter_mf,tol=tol_mf,dcode=dcode)
    m_cent=0.5*(mbns[1:]+mbns[:-1])

    if verbose>0 :
        print "  Computing measurement errors"
    phi1d_ms=np.zeros([nsims_ms,len(phi1d)])
    lm_rc=np.log10(2.356E5*ds_arr**2*fl_arr)
    for i in np.arange(nsims_ms) :
        if verbose>1 :
            print i
        fluxb=fl_arr[mask_all]+np.random.randn(ngals_good)*fle_arr[mask_all]
        distb=ds_arr[mask_all]+np.random.randn(ngals_good)*dse_arr[mask_all]
        lmhib=lm_arr[mask_all]+np.log10(np.fabs(2.356E5*distb**2*fluxb))-lm_rc[mask_all]
        lw50b=np.log10(np.fabs(w5_arr[mask_all]+np.random.randn(ngals_good)*w5e_arr[mask_all]))
        mb,p1d,e1d,wb,p2d,e2d,wrn=get_mfunc(lmhib,lw50b,distb,v_survey_all,
                                            m_range=m_range,m_nbins=m_nbins,m_rs=m_rs,
                                            w_range=w_range,w_nbins=w_nbins,w_rs=w_rs,
                                            niter=niter_mf,tol=tol_mf)

        phi1d_ms[i,:]=p1d
    e1d_ms=np.std(phi1d_ms,axis=0)

    if verbose>0 :
        print "  Computing JK"
    n_jk=len(masks_jk)
    phi1d_jk=np.zeros([n_jk,len(phi1d)])
    for i in np.arange(n_jk) :
        if verbose>1 :
            print i
        mb,p1d,e1d,wb,p2d,e2d,wrn=get_mfunc(lm_arr[masks_jk[i]],
                                            np.log10(w5_arr[masks_jk[i]]),
                                            ds_arr[masks_jk[i]],v_survey_jk[i],
                                            m_range=m_range,m_nbins=m_nbins,m_rs=m_rs,
                                            w_range=w_range,w_nbins=w_nbins,w_rs=w_rs,
                                            niter=niter_mf,tol=tol_mf)
        phi1d_jk[i,:]=p1d
    e1d_jk=np.std(phi1d_jk,axis=0)*np.sqrt(n_jk-1.)

    #Add all uncertainties in quadrature
    e1d_tot=np.sqrt(e1d_ps**2+e1d_jk**2+e1d_ms**2)

    dict_out={'m_bins':mbns,       #Bin edges
              'm_cent':m_cent,     #Center of each bin
              'phi1d':phi1d,       #Mass function
              'err1d_tot':e1d_tot, #Total errors
              'err1d_ps':e1d_ps,   #Poisson errors
              'err1d_jk':e1d_jk,   #Jackknife errors
              'err1d_ms':e1d_ms}   #Measurement errors

    return dict_out
