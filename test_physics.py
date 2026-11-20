import numpy as np
from physics import solve,analytic_Q,viscosity

def rel(a,b):return abs(a-b)/max(abs(b),1e-30)
R=.01;L=1;dp=1000
for model,p in [('Newtonian',{'mu':.1}),('Power-law',{'K':.1,'n':.5}),('Bingham',{'mu_p':.1,'yield':2})]:
    z=solve(model,p,dp,R,L,801)
    assert rel(z['Q'],analytic_Q(model,p,dp,R,L))<1e-4,(model,z['Q'])
assert solve('Bingham',{'mu_p':.1,'yield':5},dp,R,L)['Q']==0
cy={'eta0':10,'eta_inf':.1,'lam':1,'n':.5,'a':2}
zz=solve('Carreau-Yasuda',cy,dp,R,L,401)
assert zz['Q']>0
cy['eta_inf']=cy['eta0']
z=solve('Carreau-Yasuda',cy,dp,R,L,801)
assert rel(z['Q'],analytic_Q('Newtonian',{'mu':10},dp,R,L))<1e-4
print('All physics tests passed')
