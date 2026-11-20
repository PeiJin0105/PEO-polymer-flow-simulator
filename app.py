import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from physics import viscosity,solve,analytic_Q,convergence

st.set_page_config(page_title='Polymer Pipe Flow Explorer',layout='wide')
st.title('Polymer Solution Pipe Flow Explorer')
st.caption('ChemE 7007 · Interactive learning tool · Steady, fully developed, isothermal, laminar circular-pipe flow')

with st.sidebar:
    st.header('Pipe and operating conditions')
    R=st.number_input('Pipe radius R (mm)',1.,100.,10.,1.)/1000
    L=st.number_input('Pipe length L (m)',0.1,100.,1.,0.1)
    dp=st.number_input('Pressure drop ΔP (Pa)',1.,100000.,1000.,100.)
    N=st.select_slider('Radial grid points',options=[51,101,201,401,801],value=401)
    st.divider()
    st.header('Constitutive model')
    model=st.selectbox('Fluid', ['Carreau-Yasuda','Newtonian','Power-law','Bingham'])
    p={}
    if model=='Newtonian':p['mu']=st.number_input('Dynamic viscosity μ (Pa·s)',0.0001,1000.,0.1,0.01,format='%.4f')
    elif model=='Power-law':
        p['K']=st.number_input('Consistency K (Pa·sⁿ)',0.0001,1000.,0.1,0.01,format='%.4f')
        p['n']=st.slider('Flow index n',0.1,2.,0.5,0.05)
    elif model=='Bingham':
        p['mu_p']=st.number_input('Plastic viscosity μp (Pa·s)',0.0001,1000.,0.1,0.01,format='%.4f')
        p['yield']=st.slider('Yield stress τy (Pa)',0.,100.,2.,0.5)
    else:
        p['eta0']=st.number_input('Zero-shear viscosity η₀ (Pa·s)',0.001,1000.,10.,1.)
        p['eta_inf']=st.number_input('Infinite-shear viscosity η∞ (Pa·s)',0.,1000.,0.1,0.05)
        p['lam']=st.number_input('Characteristic time λ (s)',0.001,1000.,1.,0.1)
        p['n']=st.slider('Flow index n',0.1,1.,0.5,0.05)
        p['a']=st.slider('Transition parameter a',0.5,5.,2.,0.1)
        if p['eta_inf']>p['eta0']:st.error('For a shear-thinning model, require η∞ ≤ η₀.');st.stop()

res=solve(model,p,dp,R,L,N)
qexact=analytic_Q(model,p,dp,R,L)
tabs=st.tabs(['1 · Rheology','2 · Pipe flow','3 · Model comparison','4 · Verification & guide'])
with tabs[0]:
    st.subheader('Apparent viscosity versus shear rate')
    if model=='Bingham':
        st.info('Bingham has an unyielded region; apparent viscosity τ/γ diverges as γ approaches zero. This is not a PEO constitutive model.')
        gamma=np.logspace(-3,4,220);eta=p['mu_p']+p['yield']/gamma
    else:
        gamma=np.logspace(-3,4,220);eta=viscosity(gamma,model,p)
    fig=go.Figure(go.Scatter(x=gamma,y=eta,mode='lines',name=model));fig.update_layout(xaxis_type='log',yaxis_type='log',xaxis_title='Shear rate (s⁻¹)',yaxis_title='Apparent viscosity (Pa·s)',template='plotly_white')
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Carreau–Yasuda defaults are illustrative only: not fitted PEO measurements. Actual PEO response depends on molecular weight, concentration and temperature.')
with tabs[1]:
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Q (mL/s)',f"{res['Q']*1e6:.4g}")
    c2.metric('Maximum velocity (m/s)',f"{res['vmax']:.4g}")
    c3.metric('Wall shear stress (Pa)',f"{res['tau_w']:.4g}")
    c4.metric('Plug radius / R',f"{res['rplug']/R:.3f}" if model=='Bingham' else 'N/A')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=res['v'],y=res['r']/R,mode='lines',name='Velocity'))
    fig.add_trace(go.Scatter(x=res['v'][::-1],y=-res['r'][::-1]/R,mode='lines',name='Mirrored radius',showlegend=False,line=dict(color='#1f77b4')))
    fig.update_layout(xaxis_title='Axial velocity (m/s)',yaxis_title='Signed normalized radius r/R',template='plotly_white')
    st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        f=go.Figure(go.Scatter(x=res['r']/R,y=res['gamma'],mode='lines'));f.update_layout(xaxis_title='r/R',yaxis_title='Shear rate (s⁻¹)',template='plotly_white');st.plotly_chart(f,use_container_width=True)
    with c2:
        f=go.Figure(go.Scatter(x=res['r']/R,y=res['tau'],mode='lines'));f.update_layout(xaxis_title='r/R',yaxis_title='Shear stress magnitude (Pa)',template='plotly_white');st.plotly_chart(f,use_container_width=True)
    st.download_button('Download profile CSV',pd.DataFrame({'r_m':res['r'],'velocity_m_s':res['v'],'shear_rate_s_inv':res['gamma'],'stress_Pa':res['tau']}).to_csv(index=False),'profile.csv','text/csv')
with tabs[2]:
    st.subheader('Comparison under the same pressure drop and pipe geometry')
    st.caption('Comparison uses illustrative independent model parameters; do not interpret differences as measured PEO model fits.')
    samples=[('Newtonian',{'mu':0.1}),('Power-law',{'K':0.1,'n':0.5}),('Carreau-Yasuda',{'eta0':10.,'eta_inf':0.1,'lam':1.,'n':0.5,'a':2.}),('Bingham',{'mu_p':0.1,'yield':2.})]
    f=go.Figure();records=[]
    for name,params in samples:
        z=solve(name,params,dp,R,L,201)
        f.add_trace(go.Scatter(x=z['r']/R,y=z['v'],name=name,mode='lines'))
        records.append({'Model':name,'Flow rate (mL/s)':z['Q']*1e6,'Maximum velocity (m/s)':z['vmax']})
    f.update_layout(xaxis_title='r/R',yaxis_title='Velocity (m/s)',template='plotly_white');st.plotly_chart(f,use_container_width=True)
    st.dataframe(pd.DataFrame(records),hide_index=True)
    dps=np.geomspace(max(dp/10,1),dp*10,18)
    f=go.Figure()
    for name,params in samples:
        vals=[solve(name,params,d,R,L,101)['Q']*1e6 for d in dps]
        f.add_trace(go.Scatter(x=dps,y=vals,mode='lines+markers',name=name))
    f.update_layout(xaxis_title='Pressure drop (Pa)',yaxis_title='Volumetric flow rate (mL/s)',template='plotly_white');st.plotly_chart(f,use_container_width=True)
with tabs[3]:
    st.subheader('Analytical benchmarks and numerical convergence')
    if qexact is not None:
        st.write(f'Analytical Q = **{qexact:.8g} m³/s**; numerical Q = **{res["Q"]:.8g} m³/s**')
        if qexact>0:st.write(f'Relative error = **{abs(res["Q"]-qexact)/qexact*100:.5f}%**')
        else:st.write('Zero-flow limit: numerical and analytical solutions are both zero; relative error is undefined.')
        st.dataframe(pd.DataFrame(convergence(model,p,dp,R,L)),hide_index=True)
    else:
        st.info('Carreau–Yasuda has no general closed-form pipe-flow solution. Validate using the Newtonian limit η₀=η∞, numerical refinement, and stress-reconstruction residuals.')
        pn={'eta0':p['eta0'],'eta_inf':p['eta0'],'lam':p['lam'],'n':p['n'],'a':p['a']}
        zz=solve('Carreau-Yasuda',pn,dp,R,L,401)
        exact=analytic_Q('Newtonian',{'mu':p['eta0']},dp,R,L)
        st.write(f'Newtonian limit: CY Q = **{zz["Q"]:.8g} m³/s** vs analytical **{exact:.8g} m³/s**; relative error = **{abs(zz["Q"]-exact)/exact*100:.5f}%**')
        rows=[]
        qfine=solve(model,p,dp,R,L,1601)['Q']
        for grid in [51,101,201,401,801]:
            z=solve(model,p,dp,R,L,grid)
            rows.append({'Grid N':grid,'Q (m³/s)':z['Q'],'Difference from N=1601 (%)':abs(z['Q']-qfine)/max(abs(qfine),1e-30)*100})
        st.dataframe(pd.DataFrame(rows),hide_index=True)
        residual=np.max(np.abs(viscosity(res['gamma'],model,p)*res['gamma']-res['tau']))
        st.write(f'Maximum constitutive stress residual: **{residual:.3g} Pa**')
    st.divider()
    st.markdown('''**How to use**: (1) Set pipe radius, length and pressure drop in the sidebar. (2) Select a constitutive model and adjust its parameters. (3) Inspect viscosity, velocity, shear rate, and stress. (4) Compare model curves. (5) Check limiting cases and grid convergence. Export the profile CSV for your report.\n\n**Assumptions**: steady, incompressible, isothermal, axisymmetric, fully developed, no-slip, horizontal, rigid circular pipe; negligible entrance and elastic effects. Laminar flow must be checked independently for each fluid/condition. No polymer elasticity, normal-stress effects, thixotropy or degradation are modeled.\n\n**PEO interpretation**: Demonstration parameters are synthetic; a real PEO solution requires rheometry data at specified molecular weight, concentration, and temperature, followed by fitting. This tool does not infer these from PEO identity.''')
