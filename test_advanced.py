import numpy as np
from physics import solve, analytic_Q, viscosity
from advanced import pressure_for_flow, fit_carreau, flow_diagnostics, stress_residual
R=.01;L=1.;dp=1000.
def test_analytic():
    for m,p in [('Newtonian',{'mu':.1}),('Power-law',{'K':.1,'n':.5}),('Bingham',{'mu_p':.1,'yield':2.})]:
        z=solve(m,p,dp,R,L,801)
        assert abs(z['Q']/analytic_Q(m,p,dp,R,L)-1)<2e-4
        assert stress_residual(m,p,z)<1e-7

def test_inverse():
    p={'mu':.1};q=solve('Newtonian',p,dp,R,L,301)['Q']
    assert abs(pressure_for_flow('Newtonian',p,q,R,L)-dp)/dp<2e-4

def test_yield():
    assert solve('Bingham',{'mu_p':.1,'yield':5.},dp,R,L)['Q']==0

def test_carreau_limit():
    p={'eta0':10.,'eta_inf':10.,'lam':1.,'n':.5,'a':2.}
    q=solve('Carreau-Yasuda',p,dp,R,L,801)['Q']
    assert abs(q/analytic_Q('Newtonian',{'mu':10.},dp,R,L)-1)<2e-4

def test_fit():
    p={'eta0':10.,'eta_inf':.1,'lam':1.,'n':.5,'a':2.}
    x=np.logspace(-3,4,80);y=viscosity(x,'Carreau-Yasuda',p)
    fit=fit_carreau(x,y)
    assert fit['log_rmse']<.03

def test_diagnostics():
    d=flow_diagnostics('Newtonian',{'mu':.1},dp,R,L)
    assert d['Re_newton']>0
