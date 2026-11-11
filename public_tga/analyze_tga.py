"""No-refit first-order PMMA application on public measured TGA histories."""
from pathlib import Path
import csv, hashlib, json, os, sys
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT/'sources/PMMA'
PARAM = SRC/'Material_Properties/2021/MaCFP_PMMA_NIST.json'
P = json.loads(PARAM.read_text())['Kinetics']
A, E, R = P['Pre-exponential'], P['Activation Energy'], 8.31446261815324
assert P['Reaction Order'] == 1 and P['Number of Reactions'] == 1 and P['Solid Yield'] == 0
EPSILON = 0.05

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def integrate(t,T,n=8):
    x,w = np.polynomial.legendre.leggauss(n)
    mid=(T[:-1,None]+T[1:,None])/2
    half=(T[1:,None]-T[:-1,None])/2
    delta=(np.diff(t)/2)*(np.exp(-E/(R*(mid+half*x)))*w).sum(axis=1)*A
    return np.r_[0,np.cumsum(delta)]

def first_cross(x,y,v):
    ii=np.flatnonzero(y>=v)
    if not len(ii):return None
    i=ii[0]
    if i==0:return float(x[0])
    return float(x[i-1]+(x[i]-x[i-1])*(v-y[i-1])/(y[i]-y[i-1]))

def dtg(t,T,y,halfwidth=2.5):
    # Local linear least-squares slope over a +-2.5 K window; no clipping.
    out=np.zeros_like(y)
    for i in range(len(y)):
        ix=np.flatnonzero(abs(T-T[i])<=halfwidth)
        if len(ix)<3:ix=np.arange(max(i-1,0),min(i+2,len(y)))
        q=t[ix]-t[ix].mean()
        out[i]=-np.dot(q,y[ix]-y[ix].mean())/np.dot(q,q)
    return out

def analyze(path):
    a=np.loadtxt(path,delimiter=',',skiprows=2)
    t,T=a[:,0],a[:,1]
    assert np.all(np.diff(t)>0) and np.isfinite(a).all()
    is_umd=path.name.startswith('UMD')
    mass=a[:,2] if is_umd else a[:,2]/a[0,2]
    merr=a[:,3] if is_umd else np.full_like(mass,np.nan)
    H=integrate(t,T); Hcheck=integrate(t,T,16)
    chi=np.exp(-H)
    k=A*np.exp(-E/(R*T))
    rate=k*chi
    rate_exp=dtg(t,T,mass)
    rate_model_processed=dtg(t,T,chi)
    Es=-np.expm1(-H)
    # Infinite floating-point values after practical burnout are explicit, not capped.
    with np.errstate(over='ignore'):
        bias=np.expm1(H)
    scope=(T>=450)&(T<=750)
    core=(mass>=.05)&(mass<=.95)&scope
    exp_peak=np.flatnonzero(scope)[np.argmax(rate_exp[scope])]
    model_peak=np.flatnonzero(scope)[np.argmax(rate[scope])]
    threshold_H=float(np.log1p(EPSILON))
    t5=first_cross(t,H,threshold_H)
    T5=float(np.interp(t5,t,T))
    row=dict(dataset=path.stem,role='cross-laboratory test; no refit' if is_umd else 'calibration-domain consistency; not independent validation',
        source_sha256=sha(path),temperature_definition='reported TGA Temperature [K]; independent particle-vs-furnace temperature is not provided',
        initial_time_s=float(t[0]),initial_temperature_K=float(T[0]),samples=len(t),
        mass_normalization='repository mass fraction, unchanged' if is_umd else 'mass / first recorded mass',
        quadrature_abs_H_difference_max=float(abs(H-Hcheck).max()),
        quadrature_abs_chi_difference_max=float(abs(chi-np.exp(-Hcheck)).max()),
        nominal_heating_rate_K_min=10,
        mean_measured_heating_rate_450_750_K_min=float(60*(T[scope][-1]-T[scope][0])/(t[scope][-1]-t[scope][0])),
        mass_rmse_450_750_K=float(np.sqrt(np.mean((chi[scope]-mass[scope])**2))),
        mass_mae_450_750_K=float(np.mean(abs(chi[scope]-mass[scope]))),
        mass_rmse_experimental_5_95pct_remaining=float(np.sqrt(np.mean((chi[core]-mass[core])**2))),
        mass_max_abs_error_450_750_K=float(abs(chi[scope]-mass[scope]).max()),
        mass_rmse_full_record=float(np.sqrt(np.mean((chi-mass)**2))),
        temperature_at_measured_50pct_loss_K=first_cross(T,1-mass,.5),
        temperature_at_model_50pct_loss_K=first_cross(T,Es,.5),
        dtg_processing='negative least-squares slope of mass fraction vs time in +/-2.5 K window; no clipping',
        measured_dtg_peak_temperature_K=float(T[exp_peak]),model_analytical_dtg_peak_temperature_K=float(T[model_peak]),
        measured_dtg_peak_per_s=float(rate_exp[exp_peak]),model_analytical_dtg_peak_per_s=float(rate[model_peak]),
        dtg_model_processed_peak_per_s=float(rate_model_processed[scope].max()),
        dtg_rmse_450_750_K_per_s=float(np.sqrt(np.mean((rate_model_processed[scope]-rate_exp[scope])**2))),
        bias_5pct_crossing=dict(time_s=t5,temperature_K=T5,log_exposure=threshold_H,
             Es=EPSILON/(1+EPSILON),finite_source_mass_fraction=1/(1+EPSILON),
             experimental_remaining_mass_fraction=float(np.interp(t5,t,mass)),
             note='The bias is against the finite-source model at the same reported T; it is not an experimental error guarantee.'),
        at_finite_source_rate_peak=dict(temperature_K=float(T[model_peak]),Es=float(Es[model_peak]),
             remaining_fraction=float(chi[model_peak]),finite_source_relative_bias=float(bias[model_peak])),
        end_experimental_residue_fraction=float(mass[-1]),end_model_residue_fraction=float(chi[-1]))
    if is_umd:
        row['mass_fraction_inside_supplied_error_band_450_750_K']=float(np.mean(abs(chi[scope]-mass[scope])<=merr[scope]))
        row['supplied_error_definition']='UMD metadata: two standard deviations of the mean of seven tests; no reinterpretation as prediction interval'
    out=np.c_[t,T,mass,merr,H,chi,k,rate,rate_exp,rate_model_processed,Es,bias]
    outdir=ROOT/'processed';outdir.mkdir(exist_ok=True)
    np.savetxt(outdir/(path.stem+'.csv'),out,delimiter=',',header='time_s,reported_temperature_K,experimental_mass_fraction,experimental_mass_fraction_error,log_exposure,model_remaining_fraction,undepleted_specific_rate_per_s,finite_source_specific_rate_per_s,experimental_dtg_local5K_per_s,model_dtg_local5K_per_s,Es,finite_source_relative_bias',comments='')
    return row,dict(t=t,T=T,m=mass,error=merr,H=H,chi=chi,k=k,rate=rate,rate_exp=rate_exp,Es=Es,bias=bias)

def main():
    paths=sorted((SRC/'Calibration_Data/NIST').glob('*TGA*.csv'))+[SRC/'Calibration_Data/UMD/UMD_TGA_N2_10K.csv']
    results=[];arrays={}
    for p in paths:
        row,ar=analyze(p);results.append(row);arrays[p.stem]=ar
    payload=dict(material='MaCFP ACRYLITE cast black 9H01 GT PMMA',
        parameter_record='MaCFP_PMMA_NIST.json (2021)',parameter_sha256=sha(PARAM),
        A_per_s=A,E_J_mol=E,R_J_mol_K=R,reaction_order=1,residue_yield=0,
        parameter_calibration='NIST 10 K/min TGA; reaction peak equations; no fitting or parameter changes in this application',
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),
        epsilon_finite_source_relative=EPSILON,
        ode='dchi/dt=-A*exp(-E/(R*T_reported(t)))*chi; chi(t0)=1',
        numerics='piecewise-linear interpolation of actual reported T(t), 8-point Gauss-Legendre interval quadrature, checked against 16-point; continuous first-order solution, not the flame solver explicit Euler update',
        diagnostics='single uniform channel: Es=1-chi, Eb=Es, covariance=0, Jensen bound equality, finite-source-relative bias=exp(H)-1; infinite bias after floating-point underflow of remaining mass is not a finite physical-rate prediction',
        mass_error_primary_window_K=[450,750],
        exclusion='Downloaded LCPP 10/20 K/min files are not analyzed because their own README flags unresolved duplicate-temperature-rate datasets; files retained with hashes. No result-based filtering of NIST/UMD records.',
        results=results)
    (ROOT/'tga_metrics.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(ROOT/'tga_arrays.npz',**{f'{name}__{k}':v for name,a in arrays.items() for k,v in a.items()})
    print(json.dumps(payload,indent=2))
    # Plot independent external evidence and diagnostic; no synthetic temperature ramp.
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        if exc.name != 'matplotlib':
            raise
        fallback = os.environ.get('FSJ_MATPLOTLIB_PATH')
        if not fallback:
            raise ModuleNotFoundError(
                'Install dependencies with: python -m pip install numpy matplotlib; '
                'or set FSJ_MATPLOTLIB_PATH to an existing optional Matplotlib library directory.'
            ) from exc
        sys.path.append(fallback)
        import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(8.0,6.1),constrained_layout=True)
    n=arrays['NIST_TGA_N2_10K_1'];u=arrays['UMD_TGA_N2_10K']
    for j,(group,a,title) in enumerate([('NIST',n,'(a) NIST: calibration-domain consistency'),('UMD',u,'(b) UMD: cross-laboratory test')]):
        ax=axs[0,j]
        if group=='NIST':
            for key,z in arrays.items():
                if key.startswith('NIST'):ax.plot(z['T'],z['m'],color='#646464',alpha=.5,lw=.85,label='Measured: 5 repeats' if key.endswith('_1') else None)
        else:
            ax.fill_between(a['T'],a['m']-a['error'],a['m']+a['error'],color='#999999',alpha=.35,label='Reported error band')
            ax.plot(a['T'],a['m'],color='#444444',lw=1.5,label='Measured: 7-test mean')
        ax.plot(a['T'],a['chi'],color='#2166ac',lw=1.8,label='Fixed NIST one-step model')
        ax.set(xlim=(450,750),ylim=(-.025,1.05),xlabel='Reported TGA temperature (K)',ylabel='Remaining mass / initial mass',title=title)
        ax.legend(fontsize=7,loc='lower left')
    ax=axs[1,0]
    ax.plot(u['T'],u['rate_exp']*60,color='#444444',lw=1.4,label='UMD mass-derived rate (5 K slope)')
    ax.plot(u['T'],u['rate']*60,color='#2166ac',lw=1.8,label='Finite-source model: k chi')
    ax.plot(u['T'],u['k']*60,color='#b2182b',ls='--',lw=1.5,label='Undepleted formal source: k')
    ax.set(xlim=(530,700),ylim=(-.005,.45),xlabel='Reported TGA temperature (K)',ylabel=r'Normalized mass-loss rate (min$^{-1}$)',title='(c) Same-temperature source comparison')
    ax.legend(fontsize=7,loc='upper left')
    ax=axs[1,1]
    ax.plot(u['T'],100*u['bias'],color='#b2182b',lw=1.8,label='Finite-source-relative bias')
    ax.axhline(5,color='#444444',ls=':',lw=1)
    T5=results[-1]['bias_5pct_crossing']['temperature_K']
    ax.plot(T5,5,'o',color='#b2182b',ms=4)
    ax.annotate(f'5% at {T5:.1f} K',xy=(T5,5),xytext=(T5+7,13),fontsize=8,arrowprops={'arrowstyle':'-','lw':.6})
    ax.set(xlim=(500,640),ylim=(0,40),xlabel='Reported TGA temperature (K)',ylabel='(undepleted / finite-source - 1) (%)',title='(d) One-channel certificate is exact')
    ax.legend(fontsize=7,loc='upper left')
    fig.savefig(ROOT/'independent_tga_application.png',dpi=230)
    fig.savefig(ROOT/'independent_tga_application.pdf')
    fig.savefig(ROOT/'independent_tga_application.svg')
    plt.close(fig)

if __name__=='__main__':main()
