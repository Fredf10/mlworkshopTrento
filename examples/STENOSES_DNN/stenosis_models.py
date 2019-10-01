from __future__ import division
from builtins import range
import logging
import numpy as np
import scipy as sp
import scipy.optimize
from scipy.integrate import trapz

logger = logging.getLogger(__name__)
machine_eps = np.finfo(float).eps

## BEGIN HUO STENOSIS 
def p_diffusive_int_fcn(a, log=np.log, sqrt=np.sqrt,atan=np.arctan):
    """ Generated using Sympy
    poly = (1 + 4*x + 9*x**2 + 4*x**3)/(x*(3 + 2*x)*(3 + 2*x + x**2)**2)
    a = sympy.symbols("a")
    poly_int = sympy.integrate(poly, (x, a, 1))
    """
    return (5*a + 34)/(27*a**2 + 54*a + 81) - log(a)/27 + 28*log(a + 3/2)/243 - 19*log(a**2 + 2*a + 3)/486 + 34*sqrt(2)*atan(sqrt(2)*a/2 + sqrt(2)/2)/243 - 13/54 - 34*sqrt(2)*atan(sqrt(2))/243 - 28*log(5)/243 + 19*log(6)/486 + 28*log(2)/243

def alpha_int_fct(a, log=np.log, sqrt=np.sqrt,atan=np.arctan):
    """ Generated using sympy
        domain: (0, inf)
    alpha_poly = ((1-x)*(6+x)*(1 + 4*x + 9*x**2 + 4*x**3))/(5*x*(3 + 2*x)*(3 + 2*x + x**2)**2)
    a = sympy.symbols("a")
    alpha_int = sympy.integrate(alpha_poly,(x,a,1))
    """
    return -(a - 13)/(5*a**2 + 10*a + 15) - 2*log(a)/45 + 7*log(a + 3/2)/27 + 5*log(a**2 + 2*a + 3)/54 + 29*sqrt(2)*atan(sqrt(2)*a/2 + sqrt(2)/2)/54 - 29*sqrt(2)*atan(sqrt(2))/54 - 7*log(5)/27 - 2/5 - 5*log(6)/54 + 7*log(2)/27

def calc_alpha_function(left_side):
    def solve_function(x):
        value = alpha_int_fct(x)
        return (left_side-value) 
    
    if alpha_int_fct(machine_eps)<left_side:
        res_x = 0.0 #pass
    elif left_side > 0:
        try:
            res_x = scipy.optimize.brentq(solve_function,machine_eps,1.0) 
        except ValueError:
            logger.warning("Inviscid core estimate invalid: %f, %s", (left_side, map(solve_function,(machine_eps,1.0))))
            print("Inviscid core estimate invalid: %f, %s", (left_side, map(solve_function,(machine_eps,1.0))))
            res_x = np.NaN
    else: 
        res_x = 1 # L->0
    if res_x>1:
        print(left_side, res_x)
    elif res_x<0:
        print(left_side, res_x)
    return res_x

def calc_entrance_function(left_side):
    # Find entrance length
    res_x =  alpha_int_fct(0.05)/left_side 
    return res_x 
def huo_model(Q, A_0, A_in, A_out, A_s, L_vessel, L_stenosis, dyn_visc, rho):
    """
    Calculate the pressure over a stenosis with the model developed by Huo et al.
    """
    left_side = ((np.pi*dyn_visc*L_stenosis)/(rho*Q))
    alpha = calc_alpha_function(left_side)
    p_diffusive = 0
    p_expansion = 0
    if alpha>0.05:
        p_diffusive = ((rho*Q**2)/(2*A_s**2)) * (96/5.) * p_diffusive_int_fcn(alpha) + (8*np.pi*dyn_visc/A_0**2)*Q*(L_vessel-L_stenosis)
        p_expansion = ((rho*Q**2)/2.)*((1/A_s - 1/A_0)**2  + ( 2 * (1/A_s - 1/A_0)*(1/A_s - (1/3.) * 1/A_0) - (1/A_s - 1/A_0)**2)*(1-alpha)**2)
    else:
        left_side = ((np.pi*dyn_visc)/(rho*Q))
        L_entrance = calc_entrance_function(left_side)
        p_diffusive = ((rho*Q**2)/(2*A_s**2)) * (96/5.) *  p_diffusive_int_fcn(0.05) + (8*np.pi*dyn_visc/A_s**2)*Q*(L_vessel-L_entrance)
        p_expansion = ((rho*Q**2)/2.)*(1/A_s - 1/A_0)*(1/A_s - (1/3.)*1/A_0)
    
    p_convective = (rho*Q**2)*(1./A_out**2 - 1/A_in**2)/2.
    delta_p = p_diffusive + p_expansion + p_convective
    return delta_p


## End HUO STENOSIS


## Begin Young and Tsai STENOSIS

def young_tsai_model(Q, A_0, A_in, A_out, A_s, L_vessel, L_stenosis, dyn_visc, rho, Kt=1.52):
    """
    Calculate the pressure over a stenosis with the model developed by Young et al.
    """
    #print(Q, A_0, A_in, A_out, A_s, L_vessel, L_stenosis, dyn_visc, rho)
    D0 = 2.*np.sqrt(A_0/np.pi)
    Ds = 2.*np.sqrt(A_s/np.pi)
    Kv = 32.*(0.83*L_stenosis + 1.64*Ds)*((A_0/A_s)**2.)/D0

    a = Kv*dyn_visc/(A_0*D0)
    b = (1./(2.*A_0**2.))*Kt*rho*(A_0/A_s - 1.)**2.
    
    p_young_tsai = a*Q + b*abs(Q)*Q

    delta_p = p_young_tsai 
    
    return delta_p



def young_tsai_model_coeffs(A_0, A_in, A_out, A_s, L_vessel, L_stenosis, dyn_visc, rho, Kt=1.52, A_max_sphere_s=None):
    """
    Calculate the pressure over a stenosis with the model developed by Young et al.
    """
    D0 = 2.*np.sqrt(A_0/np.pi)
    Ds = 2.*np.sqrt(A_s/np.pi)
    Kv = 32.*(0.83*L_stenosis + 1.64*Ds)*((A_0/A_s)**2.)/D0

    a = Kv*dyn_visc/(A_0*D0)
    
    b = (1./(2.*A_0**2.))*Kt*rho*(A_0/A_s - 1.)**2.

    return a, b

def young_tsai_model_coeffs_exp(A_0, A_in, A_out, A_s, L_vessel, L_stenosis, dyn_visc, rho, Kt=1.52):
    """
    Calculate the pressure over a stenosis with the model developed by Young et al.
    """
    D0 = 2.*np.sqrt(A_0/np.pi)
    Ds = 2.*np.sqrt(A_s/np.pi)
    Kv = 32.*(0.83*L_stenosis + 1.64*Ds)*((A_0/A_s)**2.)/D0

    a = Kv*dyn_visc/(A_0*D0)
    

    b = (1./(2.*A_out**2.))*Kt*rho*(A_out/A_s - 1.)**2.
    
    return a, b

def young_tsai_model_coeffs_convective(A_0, A_in, A_out, A_s, L_vessel, L_stenosis, dyn_visc, rho, Kt=1.52, A_max_sphere_s=None):
    """
    Calculate the pressure over a stenosis with the model developed by Young et al.
    """
    D0 = 2.*np.sqrt(A_0/np.pi)
    Ds = 2.*np.sqrt(A_s/np.pi)
    Kv = 32.*(0.83*L_stenosis + 1.64*Ds)*((A_0/A_s)**2.)/D0

    a = Kv*dyn_visc/(A_0*D0)
    
    b = (1./(2.*A_out**2.))*Kt*rho*(A_out/A_s - 1.)**2.
    
    b_convective = rho*(1./A_out**2 - 1/A_in**2)/2.
    
    if A_out < A_in:
        
        b += b_convective
    
    return a, b

def young_tsai_model_visc_corrected_coeffs_convective(A_0, A_in, A_out, A_s, L_vessel, L_stenosis, dyn_visc, rho, Kt=1.52, A_max_sphere_s=None):
    """
    Calculate the pressure over a stenosis with the model developed by Young et al.
    """
    D0 = 2.*np.sqrt(A_0/np.pi)
    Ds = 2.*np.sqrt(A_s/np.pi)
    
    A_0_A_s_corrected = 0.75*A_0/A_s + 0.25
    Kv = 32.*(L_stenosis)*((A_0_A_s_corrected)**2.)/D0

    a = Kv*dyn_visc/(A_0*D0)
    b = (1./(2.*A_out**2.))*Kt*rho*(A_out/A_s - 1.)**2.
    
    b_convective = rho*(1./A_out**2 - 1/A_in**2)/2.
    
    if A_out < A_in:
        
        b += b_convective
    
    return a, b

def young_tsai_model_visc_corrected_coeffs(A_0, A_in, A_out, A_s, L_vessel, L_stenosis, dyn_visc, rho, Kt=1.52, A_max_sphere_s=None):
    """
    Calculate the pressure over a stenosis with the model developed by Young et al.
    """
    D0 = 2.*np.sqrt(A_0/np.pi)
    Ds = 2.*np.sqrt(A_s/np.pi)
    
    A_0_A_s_corrected = 0.75*A_0/A_s + 0.25
    Kv = 32.*(L_stenosis)*((A_0_A_s_corrected)**2.)/D0

    a = Kv*dyn_visc/(A_0*D0)
    
    b = (1./(2.*A_0**2.))*Kt*rho*(A_0/A_s - 1.)**2.
    
    return a, b

def itu_etal_model_coeffs(x_steno, r_steno, A_0, A_in, A_out, A_s, L_vessel, L_stenosis, dyn_visc, rho, Kt=1.52, zeta=2, exp=False, conv=False, A_max_sphere_s=None):
    """
    Calculate the pressure over a stenosis with the model developed by Young et al.
    """

    a = calcVesselResistance(x_steno, r_steno, zeta=zeta, mu=dyn_visc)
    
    if A_max_sphere_s != None:
        A_s = A_max_sphere_s
    if exp:
        b = (1./(2.*A_out**2.))*Kt*rho*(A_out/A_s - 1.)**2.
    else:
        b = (1./(2.*A_0**2.))*Kt*rho*(A_0/A_s - 1.)**2.
        
    if conv:
        b_convective = rho*(1./A_out**2 - 1/A_in**2)/2.
    
        if A_out < A_in:
        
            b += b_convective
    
    return a, b

def oneD_model_coeffs(x_steno, r_steno, A_0, A_in, A_out, A_s, L_vessel, L_stenosis, dyn_visc, rho, zeta=2):
    """
    Calculate the pressure over a stenosis with the model developed by Young et al.
    """

    a = calcVesselResistance(x_steno, r_steno, zeta=zeta, mu=dyn_visc)

    b_convective = rho*(1./A_out**2 - 1/A_in**2)/2.
    
    
    return a, b_convective

def simpleFit(x_steno, r_steno, A_0, A_in, A_out, A_s, L_vessel, L_stenosis, dyn_visc, rho, Kt=1.52, zeta=2):
    """
    Calculate the pressure over a stenosis with the model developed by Young et al.
    """

    a = calcVesselResistance(x_steno, r_steno, zeta=zeta, mu=dyn_visc)
    
    b = (1./(2.*A_0**2.))*Kt*rho*(A_0/A_s - 1.)**2.
    
    a *= 1.46 #1.473
    b *= 1.46 #1.473
    
    return a, b

def simpleFit_A_B(x_steno, r_steno, A_0, A_in, A_out, A_s, L_vessel, L_stenosis, dyn_visc, rho, Kt=1.52, zeta=2):
    """
    Calculate the pressure over a stenosis with the model developed by Young et al.
    """

    a = calcVesselResistance(x_steno, r_steno, zeta=zeta, mu=dyn_visc)
    
    b = (1./(2.*A_0**2.))*Kt*rho*(A_0/A_s - 1.)**2.
    
    a *= 1. #0.774
    b *= 1.664 #1.609
    
    return a, b

def ItuEtAlMLCorrection_b_Q(MLscaler, MLmodel, vessel, DP0D, Q, MLmaskDict, FeaturesML, convertUnits=True):

    #===================================================
    # conversion factors
    #===================================================
    ml_to_m3 = 1e-6
    mmHg_to_Pa = 133.322
    
    if convertUnits:
        DP0D /= mmHg_to_Pa
        Q /= ml_to_m3
    vesselType = vessel["vesselType"]
    
    if vesselType != "steno":
        print("Warning! trying to predict delta correction for stenosis, but vessel/segment is healthy")
        exit(-1)
    
    Ids = vessel["Ids"]
    #MLmaskLabelsToRetrieve = ["rProx", "rMin", "rDist", "length", "curvature", "shapeS", "rStd"]
#     rProx = max(MLmaskDict["rProx"][Ids][:,0])
#     rMin = max(MLmaskDict["rMin"][Ids][:,0])
#     #rMinSphere = max(MLmaskDict["rMinSphere"][Ids][:,0])
#     rDist = max(MLmaskDict["rDist"][Ids][:,0])
#     length = max(MLmaskDict["length"][Ids][:,0])
#     
    
    
    
    
    #gather 1D feature arrays
    
    featuresArray = []
    for feature in FeaturesML:
        if feature == "Q":
            featureLoc = Q
        elif feature =="DP0D":
            featureLoc = DP0D
        else:
            featureLoc = max(MLmaskDict[feature][Ids][:,0])
          
        featuresArray.append(featureLoc)
         
    #X = np.array([[rProx, rMin, rDist, length, Q, DP0D]])
    X = np.array([featuresArray])
    X = MLscaler.transform(X)
    
    pred_y = 1.*MLmodel.predict(X)
    
    pred_y *= mmHg_to_Pa
    
    return pred_y[0, 0]

def ItuEtAlMLCorrectionLinearQuadratic(MLscaler_A, MLmodel_A, MLscaler_B, MLmodel_B, 
                                       Ids, A_0D, B_0D, MLmaskDict, FeaturesML,convertUnits=True):

    #===================================================
    # conversion factors
    #===================================================
    ml_to_m3 = 1e-6
    m3_to_ml = 1e6
    mmHg_to_Pa = 133.322
    Pa_to_mmHg = 1./133.322
    #print("A_0D_SI: {0}, B_0D_SI: {1}".format(A_0D, B_0D))
    if convertUnits:
        A_0D *= Pa_to_mmHg/m3_to_ml
        B_0D *= Pa_to_mmHg/(m3_to_ml**2)

    # gather 1D feature arrays
    
    featuresArray = []
    for feature in FeaturesML:
        featureLoc = max(MLmaskDict[feature][Ids][:,0])
          
        featuresArray.append(featureLoc)
    featuresArray[-1] *= 1.664
    #X = np.array([[rProx, rMin, rDist, length, Q, DP0D]])
    X_B = np.array([featuresArray])
    #X_B = np.array([[rProx, rMin, rDist, length, B_0D, curvature, shapeS,exentric,rMinSphere]])
    #X_A = 0.*MLscaler_A.transform(X_A)
    X_B = MLscaler_B.transform(X_B)
    
    delta_A = 0. #MLmodel_A.predict(X_A)[0, 0]# - A_0D
    delta_B = 1.*MLmodel_B.predict(X_B)[0, 0]

    #print("delta_A_M: {0}, delta_B_M: {1}".format(delta_A, delta_B))
    A_ML = A_0D + delta_A
    B_ML = B_0D + delta_B
    
    #print("A_ML_M: {0}, B_ML_M: {1}".format(A_ML, B_ML))
    print("A_ML/A_0D: {0}, B_ML/B_0D: {1}".format(A_ML/A_0D, B_ML/B_0D))
    A_ML *= mmHg_to_Pa/ml_to_m3
    
    B_ML *= mmHg_to_Pa/(ml_to_m3**2)
    
    #print("A_ML_SI: {0}, B_ML_SI: {1}".format(A_ML, B_ML))
    
    return A_ML, B_ML

def calcVesselResistance(x, r, zeta=2., mu=3.5e-3):
    """ calculate the vessel resistance:
        Rv = 2(zeta + 2)*pi*my*K3, where
        K3 = int(1/Ad**2)dx
        
        """
    

    #N = len(x)
    
    #x_new = np.linspace(x[0], x[-1], N)
    #r_new = np.interp(x_new, x, r)
        
    Ad = np.pi*r**2
    
    f = 1./(Ad**2)
    K3 = trapz(f, x)
    Rv = 2*(zeta + 2)*np.pi*mu*K3
    
    return Rv

