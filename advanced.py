"""Supplementary calculations for the educational pipe-flow application (SI units)."""
import numpy as np
from scipy.optimize import brentq, least_squares
from physics import solve, viscosity, shear_rate


def pressure_for_flow(model, p, target_q, R, L, n=301):
    if target_q < 0: raise ValueError('Target flow rate must be nonnegative')
    if target_q == 0: return 0.0
    def residual(dp): return solve(model,p,dp,R,L,n)['Q']-target_q
    low=0.0
    high=1000.0
    for _ in range(90):
        if residual(high)>=0: break
        high*=2
    else: raise ValueError('Target flow rate could not be bracketed')
    return brentq(residual,low,high,xtol=1e-8,rtol=1e-10)


def flow_diagnostics(model,p,dp,R,L,density=1000.0,n=301):
    z=solve(model,p,dp,R,L,n)
    D=2*R; U=z['vmean']; gamma_wall=float(z['gamma'][-1]); tau_wall=z['tau_w']
    if model=='Bingham':
        eta_wall=p['mu_p']+p['yield']/max(gamma_wall,1e-30) if gamma_wall>0 else np.inf
    else: eta_wall=float(viscosity(np.array([gamma_wall]),model,p)[0])
    re_wall=density*U*D/eta_wall if np.isfinite(eta_wall) and eta_wall>0 else 0.
    re_newton=density*U*D/p['mu'] if model=='Newtonian' else None
    # This is a screening ratio, not a validated generalized Reynolds number.
    return {'U':U,'eta_wall':eta_wall,'Re_wall_screen':re_wall,'Re_newton':re_newton,
            'gamma_wall':gamma_wall,'tau_wall':tau_wall,
            'entrance_newton_estimate_m':0.05*re_newton*D if re_newton is not None else None}


def fit_carreau(gamma, eta, eta_inf_fixed=None):
    """Log-space bounded least squares. Parameters may be nonidentifiable without both plateaus."""
    x=np.asarray(gamma,float); y=np.asarray(eta,float)
    mask=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
    x=x[mask];y=y[mask]
    if len(x)<8 or len(np.unique(x))<6: raise ValueError('At least eight positive points with six distinct shear rates are required')
    order=np.argsort(x);x=x[order];y=y[order]
    e0=max(float(np.max(y)),float(np.median(y)))
    einf=min(float(np.min(y)),e0*0.9)
    if eta_inf_fixed is not None:
        if not 0<=eta_inf_fixed<e0: raise ValueError('Fixed η∞ must be nonnegative and below η₀ estimate')
        einf=eta_inf_fixed
    scale=float(np.median(x))
    def decode(z):
        e0_=np.exp(z[0]); einf_=e0_/(1+np.exp(-z[1])) if eta_inf_fixed is None else einf
        return {'eta0':e0_,'eta_inf':einf_,'lam':np.exp(z[2]),'n':z[3],'a':np.exp(z[4])}
    # Fit eta_inf / eta0 with a logistic transform; preserve eta_inf < eta0.
    ratio=np.clip(einf/e0,1e-5,0.95)
    z0=[np.log(e0),np.log(ratio/(1-ratio)),np.log(1/scale),0.5,np.log(2.)]
    lb=[np.log(max(min(y)*0.1,1e-8)),-15,np.log(1e-9),0.05,np.log(.2)]
    ub=[np.log(max(y)*1e4),8,np.log(1e9),1.,np.log(10.)]
    def err(z):
        p=decode(z)
        return np.log(np.maximum(viscosity(x,'Carreau-Yasuda',p),1e-30))-np.log(y)
    opt=least_squares(err,z0,bounds=(lb,ub),max_nfev=3000)
    p=decode(opt.x);pred=viscosity(x,'Carreau-Yasuda',p)
    log_res=np.log(pred)-np.log(y)
    sst=np.sum((np.log(y)-np.mean(np.log(y)))**2)
    r2=1-np.sum(log_res**2)/sst if sst>0 else np.nan
    return {'params':p,'gamma':x,'observed':y,'predicted':pred,'log_rmse':float(np.sqrt(np.mean(log_res**2))),
            'log_r2':float(r2),'success':bool(opt.success),'message':opt.message}


def stress_residual(model,p,z):
    if model=='Bingham':
        tau_rec=p['yield']+p['mu_p']*z['gamma']
        mask=z['gamma']>0
        return float(np.max(np.abs(tau_rec[mask]-z['tau'][mask]))) if np.any(mask) else 0.
    tau_rec=viscosity(z['gamma'],model,p)*z['gamma']
    return float(np.max(np.abs(tau_rec-z['tau'])))
