"""Steady fully-developed generalized-Newtonian circular-pipe flow.
All quantities in SI. Positive dp denotes pressure drop along the pipe.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid, trapezoid


def viscosity(gamma, model, p):
    g=np.asarray(gamma,dtype=float)
    if model=='Newtonian': return np.full_like(g,p['mu'])
    if model=='Power-law': return p['K']*np.maximum(g,1e-300)**(p['n']-1)
    if model=='Carreau-Yasuda':
        return p['eta_inf']+(p['eta0']-p['eta_inf'])*(1+(p['lam']*g)**p['a'])**((p['n']-1)/p['a'])
    raise ValueError(model)


def shear_rate(tau, model, p):
    t=np.asarray(tau,dtype=float)
    if model=='Newtonian': return t/p['mu']
    if model=='Power-law': return (t/p['K'])**(1/p['n'])
    if model=='Bingham': return np.maximum(t-p['yield'],0)/p['mu_p']
    if model=='Carreau-Yasuda':
        out=np.zeros_like(t)
        for i,tt in enumerate(t):
            if tt<=0: continue
            # eta(gamma)*gamma strictly increases for n>0, eta_inf>=0
            hi=max(tt/max(p['eta_inf'],1e-9),tt/max(p['eta0'],1e-9),1e-12)
            while viscosity(hi,model,p)*hi<tt: hi*=2
            out[i]=brentq(lambda x:float(viscosity(x,model,p))*x-tt,0,hi,xtol=1e-13,rtol=1e-12)
        return out
    raise ValueError(model)


def solve(model,p,dp=1000,R=0.01,L=1,N=501):
    r=np.linspace(0,R,N)
    tau=dp*r/(2*L)
    g=shear_rate(tau,model,p)
    v=cumulative_trapezoid(g[::-1],r[::-1],initial=0)[::-1]*(-1)
    Q=2*np.pi*trapezoid(v*r,r)
    tau_w=tau[-1]
    rplug=R*min(1,p['yield']/tau_w) if model=='Bingham' and tau_w>0 else 0.
    return dict(r=r,tau=tau,gamma=g,v=v,Q=Q,vmax=v[0],vmean=Q/(np.pi*R**2),tau_w=tau_w,rplug=rplug)


def analytic_Q(model,p,dp,R,L):
    tw=dp*R/(2*L)
    if model=='Newtonian':return np.pi*dp*R**4/(8*p['mu']*L)
    if model=='Power-law':return np.pi*R**3*p['n']/(3*p['n']+1)*(tw/p['K'])**(1/p['n'])
    if model=='Bingham':
        if tw<=p['yield']:return 0.
        b=p['yield']/tw
        return np.pi*tw*R**3/(12*p['mu_p'])*(3-4*b+b**4)
    return None


def convergence(model,p,dp,R,L,grids=(51,101,201,401,801)):
    rows=[]
    qref=analytic_Q(model,p,dp,R,L)
    for N in grids:
        q=solve(model,p,dp,R,L,N)['Q']
        rows.append({'N':N,'Q (m³/s)':q,'Relative error (%)':abs(q-qref)/abs(qref)*100 if qref not in (None,0) else np.nan})
    return rows
