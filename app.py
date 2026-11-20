"""ChemE 7007 upgraded educational simulator. No experimental PEO data bundled."""
import io
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from physics import solve, viscosity, analytic_Q
from advanced import pressure_for_flow, flow_diagnostics, fit_carreau, stress_residual

st.set_page_config(page_title='PEO Polymer Pipe Flow Lab',layout='wide')
st.title('PEO Polymer Solution · Pipe Flow Learning Lab')
st.caption('ChemE 7007 | Steady, fully developed, isothermal, generalized-Newtonian pipe flow | SI units')
DEFAULTS={'Rmm':10.,'L':1.,'dp':1000.,'density':1000.,'grid':401,'model':'Carreau-Yasuda',
          'mu':.1,'K':.1,'power_n':.5,'mu_p':.1,'yield_stress':2.,'eta0':10.,'eta_inf':.1,
          'lam':1.,'cy_n':.5,'a':2.,'target_ml':10.,'reference_rate':10.}

def reset():
    for k,v in DEFAULTS.items():st.session_state[k]=v
    for k in ('fitted_parameters','fit_metadata'):st.session_state.pop(k,None)

with st.sidebar:
    st.button('↺ Reset all parameters',on_click=reset,use_container_width=True)
    st.header('Pipe geometry / operating point')
    R=st.number_input('Radius R (mm)',1.,100.,key='Rmm')/1000
    L=st.number_input('Length L (m)',.1,100.,key='L')
    dp=st.number_input('Pressure drop ΔP (Pa)',1.,1e7,key='dp')
    density=st.number_input('Density ρ (kg/m³)',100.,2500.,key='density')
    N=st.select_slider('Radial grid points',options=[51,101,201,401,801],key='grid')
    st.divider()
    model=st.selectbox('Constitutive model',['Carreau-Yasuda','Newtonian','Power-law','Bingham'],key='model')
    if model=='Newtonian':p={'mu':st.number_input('μ (Pa·s)',.0001,1000.,key='mu')}
    elif model=='Power-law':
        p={'K':st.number_input('K (Pa·sⁿ)',.0001,1000.,key='K'),
           'n':st.slider('n',.1,2.,key='power_n')}
    elif model=='Bingham':
        p={'mu_p':st.number_input('Plastic viscosity (Pa·s)',.0001,1000.,key='mu_p'),
           'yield':st.number_input('Yield stress (Pa)',0.,1000.,key='yield_stress')}
    else:
        p={'eta0':st.number_input('η₀ (Pa·s)',.0001,10000.,key='eta0'),
           'eta_inf':st.number_input('η∞ (Pa·s)',0.,10000.,key='eta_inf'),
           'lam':st.number_input('λ (s)',.00001,10000.,key='lam'),
           'n':st.slider('n',.05,1.,key='cy_n'),
           'a':st.slider('a',.2,10.,key='a')}
        if p['eta_inf']>p['eta0']:st.error('Require η∞ ≤ η₀.');st.stop()
    st.caption('The default PEO-like parameters are synthetic teaching examples, NOT measured PEO properties.')

z=solve(model,p,dp,R,L,N)
diag=flow_diagnostics(model,p,dp,R,L,density,N)

def chart(fig):
    fig.update_layout(template='plotly_white',margin=dict(l=15,r=15,t=35,b=15),legend=dict(orientation='h'))
    st.plotly_chart(fig,use_container_width=True)

def csv_download(df,name):st.download_button('Download '+name,df.to_csv(index=False).encode(),name,'text/csv')

def figure_download(fig,name):
    # Browser Plotly modebar offers 'Download plot as PNG'; no Kaleido dependency.
    st.caption(f'Use the Plotly camera icon (top-right of chart) to save {name} as PNG.')

pages=st.tabs(['1 · Rheology & data fitting','2 · Pipe flow & 2D field','3 · Engineering analysis','4 · Verification','5 · User guide'])
with pages[0]:
    st.subheader('Apparent viscosity vs shear rate')
    rates=np.logspace(-3,4,250)
    eta=p['mu_p']+p['yield']/rates if model=='Bingham' else viscosity(rates,model,p)
    fig=go.Figure(go.Scatter(x=rates,y=eta,name=model,mode='lines'))
    fig.update_layout(xaxis_type='log',yaxis_type='log',xaxis_title='Shear rate (s⁻¹)',yaxis_title='Apparent viscosity (Pa·s)')
    chart(fig);figure_download(fig,'rheology')
    st.divider()
    st.subheader('Optional experimental rheometry CSV fitting')
    st.info('Upload measured shear-rate and viscosity columns (positive values, SI units). No PEO measurements are supplied with this tool. Fitting is exploratory and requires independent validation.')
    uploaded=st.file_uploader('CSV file',type=['csv'])
    if uploaded:
        try:
            df=pd.read_csv(uploaded)
            if len(df.columns)<2:raise ValueError('CSV needs at least two columns')
            c1,c2=st.columns(2)
            with c1: xcol=st.selectbox('Shear-rate column (s⁻¹)',df.columns,index=0)
            with c2: ycol=st.selectbox('Viscosity column (Pa·s)',df.columns,index=1)
            fixed=st.checkbox('Fix η∞ using a known independent value')
            einf_fixed=st.number_input('Fixed η∞ (Pa·s)',min_value=0.,value=.001,format='%.5f') if fixed else None
            if st.button('Fit Carreau–Yasuda to CSV'):
                fit=fit_carreau(df[xcol].to_numpy(),df[ycol].to_numpy(),einf_fixed)
                st.session_state['fitted_parameters']=fit['params']
                st.session_state['fit_metadata']={'log_rmse':fit['log_rmse'],'log_r2':fit['log_r2'],'n_points':len(fit['gamma'])}
                st.success('Fit computed. Inspect residuals and parameter identifiability before interpreting.')
            if 'fitted_parameters' in st.session_state:
                fit=fit_carreau(df[xcol].to_numpy(),df[ycol].to_numpy(),einf_fixed)
                st.json(fit['params'])
                st.write(f"Log-space RMSE: {fit['log_rmse']:.4g}; log-space R²: {fit['log_r2']:.4f}")
                f=go.Figure()
                f.add_trace(go.Scatter(x=fit['gamma'],y=fit['observed'],mode='markers',name='Observed'))
                f.add_trace(go.Scatter(x=fit['gamma'],y=fit['predicted'],mode='lines',name='Fitted'))
                f.update_layout(xaxis_type='log',yaxis_type='log',xaxis_title='Shear rate (s⁻¹)',yaxis_title='Viscosity (Pa·s)')
                chart(f)
                csv_download(pd.DataFrame({'shear_rate_s_inv':fit['gamma'],'viscosity_observed_Pa_s':fit['observed'],'viscosity_fitted_Pa_s':fit['predicted'],'log_residual':np.log(fit['predicted']/fit['observed'])}),'fit_results.csv')
                if st.button('Apply fitted parameters to pipe-flow sidebar'):
                    for name,key in [('eta0','eta0'),('eta_inf','eta_inf'),('lam','lam'),('n','cy_n'),('a','a')]:
                        st.session_state[key]=float(fit['params'][name])
                    st.session_state['model']='Carreau-Yasuda';st.rerun()
                st.warning('A fit is not a validation. Without measured low/high-shear plateaus, η₀, η∞, λ, n and a may be poorly identifiable. Check the measurement range and residuals.')
        except Exception as exc:st.error(f'Unable to fit CSV: {exc}')
with pages[1]:
    st.subheader('Fully developed velocity, shear rate and stress')
    c=st.columns(4)
    c[0].metric('Q (mL/s)',f"{z['Q']*1e6:.5g}")
    c[1].metric('Umean (m/s)',f"{z['vmean']:.5g}")
    c[2].metric('τwall (Pa)',f"{z['tau_w']:.5g}")
    c[3].metric('Plug radius / R',f"{z['rplug']/R:.3f}" if model=='Bingham' else 'N/A')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=z['v'],y=z['r']/R,mode='lines',name='Upper radius'))
    fig.add_trace(go.Scatter(x=z['v'][::-1],y=-z['r'][::-1]/R,mode='lines',name='Lower radius'))
    fig.update_layout(xaxis_title='Axial velocity (m/s)',yaxis_title='Signed radial coordinate / R')
    chart(fig)
    a,b=st.columns(2)
    with a:
        f=go.Figure(go.Scatter(x=z['r']/R,y=z['gamma'],mode='lines'))
        f.update_layout(xaxis_title='r/R',yaxis_title='Shear rate (s⁻¹)');chart(f)
    with b:
        f=go.Figure(go.Scatter(x=z['r']/R,y=z['tau'],mode='lines'))
        f.update_layout(xaxis_title='r/R',yaxis_title='Shear stress magnitude (Pa)');chart(f)
    st.subheader('2D axial velocity field (derived from the 1D solution)')
    x=np.linspace(0,L,90); y=np.linspace(-R,R,110)
    vv=np.interp(np.abs(y),z['r'],z['v'])
    field=np.repeat(vv[:,None],len(x),axis=1)
    f=go.Figure(go.Heatmap(x=x,y=y*1000,z=field,colorscale='Viridis',colorbar=dict(title='m/s')))
    f.update_layout(xaxis_title='Axial position x (m)',yaxis_title='Radial coordinate (mm)');chart(f)
    st.caption('This 2D view repeats the fully developed 1D solution along the pipe. It is NOT an independent CFD solution or entrance-flow simulation.')
    csv_download(pd.DataFrame({'r_m':z['r'],'velocity_m_s':z['v'],'shear_rate_s_inv':z['gamma'],'shear_stress_Pa':z['tau']}),'velocity_profile.csv')
with pages[2]:
    st.subheader('Specified flow rate → required pressure drop')
    target=st.number_input('Target flow rate (mL/s)',.001,10000.,key='target_ml')
    if st.button('Calculate required pressure drop'):
        try:
            required=pressure_for_flow(model,p,target*1e-6,R,L)
            st.metric('Required pressure drop (Pa)',f'{required:,.4f}')
            st.metric('Ideal hydraulic power ΔP × Q (W)',f'{required*target*1e-6:.5g}')
            st.caption('Hydraulic power is ideal; pump inefficiency and fittings are excluded.')
        except Exception as exc:st.error(str(exc))
    st.divider()
    st.subheader('Pressure drop–flow rate curve')
    dps=np.geomspace(max(dp/10,1e-4),dp*10,18)
    qs=[solve(model,p,float(d),R,L,151)['Q']*1e6 for d in dps]
    f=go.Figure(go.Scatter(x=dps,y=qs,mode='lines+markers',name=model))
    f.update_layout(xaxis_title='Pressure drop (Pa)',yaxis_title='Flow rate (mL/s)');chart(f)
    csv_download(pd.DataFrame({'pressure_drop_Pa':dps,'flow_rate_mL_s':qs}),'pressure_flow.csv')
    st.subheader('Compare constitutive models at a matched reference viscosity')
    gref=st.number_input('Reference shear rate (s⁻¹)',.01,1e5,key='reference_rate')
    base_eta=.1
    models=[('Newtonian',{'mu':base_eta}),('Power-law',{'K':base_eta*gref**(1-.5),'n':.5}),
            ('Carreau-Yasuda',{'eta0':10.,'eta_inf':.01,'lam':1.,'n':.5,'a':2.})]
    cy=models[-1][1]
    eta_at_ref=float(viscosity(np.array([gref]),'Carreau-Yasuda',cy)[0])
    scale=base_eta/eta_at_ref
    cy={**cy,'eta0':cy['eta0']*scale,'eta_inf':cy['eta_inf']*scale}
    f=go.Figure();records=[]
    for name,params in models:
        zz=solve(name,params,dp,R,L,301)
        f.add_trace(go.Scatter(x=zz['r']/R,y=zz['v'],name=name))
        records.append({'model':name,'viscosity_at_reference_Pa_s':float(viscosity(np.array([gref]),name,params)[0]),'flow_rate_mL_s':zz['Q']*1e6})
    f.update_layout(xaxis_title='r/R',yaxis_title='Velocity (m/s)');chart(f)
    st.dataframe(pd.DataFrame(records),hide_index=True)
    st.caption('Matched at ONE reference shear rate, not matched over the full shear-rate range. Model parameters are synthetic and not independently fitted to the same PEO sample.')
with pages[3]:
    st.subheader('Numerical and physical checks')
    exact=analytic_Q(model,p,dp,R,L)
    residual=stress_residual(model,p,z)
    st.write(f'Maximum constitutive stress residual: **{residual:.4g} Pa**')
    if exact is not None:
        st.write(f'Analytical Q: **{exact:.9g} m³/s**; numerical Q: **{z["Q"]:.9g} m³/s**')
        st.write('Relative error: **{:.6g}%**'.format(abs(z['Q']-exact)/exact*100) if exact>0 else 'Both solutions give zero flow; relative error undefined.')
    else:
        pn={**p,'eta_inf':p['eta0']}
        zn=solve('Carreau-Yasuda',pn,dp,R,L,801)
        en=analytic_Q('Newtonian',{'mu':p['eta0']},dp,R,L)
        st.write(f'Newtonian limiting case: numerical **{zn["Q"]:.9g}** vs analytical **{en:.9g} m³/s**')
    st.subheader('Grid convergence')
    reference=solve(model,p,dp,R,L,1601)['Q']
    rows=[]
    for grid in (51,101,201,401,801):
        q=solve(model,p,dp,R,L,grid)['Q']
        rows.append({'grid_N':grid,'flow_rate_m3_s':q,'difference_from_N1601_pct':abs(q-reference)/abs(reference)*100 if reference else 0})
    df=pd.DataFrame(rows);st.dataframe(df,hide_index=True)
    f=go.Figure(go.Scatter(x=df['grid_N'],y=df['difference_from_N1601_pct'],mode='lines+markers'))
    f.update_layout(xaxis_title='Number of radial nodes',yaxis_title='Difference from N=1601 (%)');chart(f)
    csv_download(df,'grid_convergence.csv')
    st.subheader('Applicability screening')
    st.write(f'Wall apparent viscosity: **{diag["eta_wall"]:.5g} Pa·s**')
    st.write(f'ρUD/ηwall screening ratio: **{diag["Re_wall_screen"]:.5g}**')
    st.warning('ρUD/ηwall is only a qualitative screening ratio for non-Newtonian fluids, NOT a validated generalized Reynolds number or a universal transition criterion. Confirm laminar flow with an appropriate correlation or experiment.')
    if model=='Newtonian':
        st.write(f'Newtonian Reynolds number: **{diag["Re_newton"]:.5g}**')
        st.write(f'Approximate laminar entrance length 0.05 Re D: **{diag["entrance_newton_estimate_m"]:.4g} m** (empirical estimate).')
        if diag['Re_newton']>=2300:st.error('Re ≥ 2300: the laminar assumption is questionable for this Newtonian example.')
        if diag['entrance_newton_estimate_m']>L:st.warning('Estimated entrance length exceeds pipe length; fully developed assumption is questionable.')
    st.caption('Fully developed behavior, incompressibility, no slip and negligible polymer elasticity are assumptions—not outputs verified by this model.')
with pages[4]:
    st.markdown('''### Learning objectives
1. Distinguish Newtonian, power-law, Bingham and Carreau–Yasuda constitutive behavior.
2. Connect axial momentum balance to radial shear stress and velocity profiles.
3. Predict pressure drop at specified flow rate and interpret engineering implications.
4. Verify solutions using exact limits, constitutive residuals and grid convergence.

### Workflow
1. Choose a fluid and pipe conditions in the sidebar.
2. Explore **Rheology** and **Pipe flow**. Use the Plotly camera icon to save a figure as PNG.
3. Use **Engineering analysis** to invert Q(ΔP) and compare models at matched reference viscosity.
4. Use **Verification** to inspect error, convergence and model applicability.
5. Optionally upload real rheometry CSV and fit Carreau–Yasuda parameters. Record PEO molecular weight, concentration, solvent, temperature and test protocol outside the tool.
6. Download CSV tables for Origin/Excel and cite the parameter set in your report.

### Assumptions and limitations
Steady, isothermal, incompressible, axisymmetric, fully developed flow in a rigid horizontal circular pipe, no slip. Carreau–Yasuda is a generalized-Newtonian *shear-viscosity* model: it does not model elasticity, normal stresses, memory, entrance effects, turbulence, thixotropy or polymer degradation. Bingham is a teaching comparison, not an assumed PEO constitutive law. **Default PEO-like parameters are synthetic.**

### Example CSV format
`shear_rate_s_inv,viscosity_Pa_s`\n`0.1,8.7`\n`1.0,6.5`\n`10.0,2.8`\n... at least eight valid rows.

### Reproducibility
Use `python -m pytest -q` to run regression tests. The app and calculations are open-source Python files, and no experimental PEO dataset is bundled.''')
    st.download_button('Download current parameter record (JSON)',json.dumps({'model':model,'constitutive_parameters':p,'radius_m':R,'length_m':L,'pressure_drop_Pa':dp,'density_kg_m3':density,'grid_nodes':N,'data_status':'illustrative unless explicitly fitted'},indent=2),file_name='simulation_parameters.json',mime='application/json')
