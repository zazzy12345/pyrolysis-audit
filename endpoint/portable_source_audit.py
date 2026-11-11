"""Portable endpoint source-memory audit; requires NumPy only."""
from __future__ import annotations
import argparse, csv, hashlib
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def alpha_q(source,capacity,q=.90):
    """Minimum capacity share, ranked by source/capacity, with fractional last cell."""
    s=np.asarray(source,float).ravel(); m=np.asarray(capacity,float).ravel()
    if s.shape!=m.shape or np.any(s<0) or np.any(m<=0): raise ValueError('invalid source/capacity')
    if not s.sum()>0: return float('nan')
    o=np.argsort(s/m)[::-1]; s,m=s[o],m[o]; target=q*float(s.sum()); c=np.cumsum(s); i=int(np.searchsorted(c,target,'left')); before=0. if i==0 else float(c[i-1])
    return float((m[:i].sum()+((target-before)/s[i])*m[i])/m.sum())
def self_test():
    # Unequal cells: density ranks source [100,90] / capacity [100,1] as cell 2 first.
    if not np.isclose(alpha_q([100.,90.],[100.,1.]),82./101.,rtol=0,atol=1e-15): raise AssertionError('nonuniform-capacity C90 test failed')
    # A zero-activity cell at positive source weight makes D_eff and B infinite.
    all_lost=audit(np.array([1.,0.]),np.array([0.,0.]),np.array([1.,5.]))
    if not (np.isinf(all_lost['B_relative_finite_source']) and np.isinf(all_lost['D_eff']) and np.isnan(all_lost['C90_active_capacity_fraction'])): raise AssertionError('zero-active-source handcheck failed')
    # k=0, chi=0 is a zero-weight cell: it must not make D_eff NaN or infinite.
    zero_weight=audit(np.array([1.,0.]),np.array([.5,0.]),np.array([1.,5.]))
    if not np.isclose(zero_weight['D_eff'],np.log(2.),rtol=0,atol=1e-15): raise AssertionError('zero-weight chi handcheck failed')
def audit(k,chi,M):
    if k.shape!=chi.shape or k.shape!=M.shape or not all(np.isfinite(x).all() for x in (k,chi,M)): raise ValueError('matching finite fields required')
    if np.any(k<0) or np.any(M<=0) or np.any(chi<0) or np.any(chi>1): raise ValueError('need k>=0, M>0, 0<=chi<=1')
    potential=M*k; active=potential*chi
    if not potential.sum()>0: raise ValueError('zero potential source')
    J0=float(potential.sum()); Jchi=float(active.sum()); p=M/M.sum(); w=potential/J0; loss=1-chi; eb=float((p*loss).sum()); es=float((w*loss).sum())
    positive_weight=potential>0
    D=float('inf') if np.any(positive_weight & (chi==0)) else float(np.sum(w[positive_weight]*(-np.log(chi[positive_weight]))))
    B=float('inf') if Jchi==0 else J0/Jchi-1.0
    upper=float('inf') if np.isinf(D) else float(np.exp(D)-1)
    return {'E_bulk':eb,'E_source':es,'B_relative_finite_source':B,'D_eff':D,'Jensen_relative_upper':upper,'C90_potential_capacity_fraction':alpha_q(potential,M),'C90_active_capacity_fraction':alpha_q(active,M),'C90_active_defined':bool(Jchi>0),'potential_source_sum_kg_m_s':J0,'active_source_sum_kg_m_s':Jchi}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input',nargs='?',type=Path,help='NPZ file containing time, rate, activity and capacity arrays')
    ap.add_argument('--output',type=Path,default=HERE/'portable_source_audit.csv')
    ap.add_argument('--self-test',action='store_true',help='run small algebraic hand checks without research data')
    a=ap.parse_args(); self_test()
    if a.self_test and a.input is None:
        print('Endpoint algebraic self-tests passed.')
        return
    if a.input is None:ap.error('provide an input NPZ file, or use --self-test')
    with np.load(a.input,allow_pickle=False) as d:
        need=('time_s','k_s_inv','chi','M_kg_per_m'); missing=[x for x in need if x not in d.files]
        if missing: raise ValueError('missing NPZ fields: '+', '.join(missing))
        t=np.asarray(d['time_s'],float); k=np.asarray(d['k_s_inv'],float); chi=np.asarray(d['chi'],float); M=np.asarray(d['M_kg_per_m'],float)
    if k.ndim<2 or chi.shape!=k.shape or M.shape!=k.shape[1:] or t.shape!=(k.shape[0],): raise ValueError('expected time=(nt,), k/chi=(nt,...), M=(...)')
    rows=[{'time_s':float(tt),**audit(k[i],chi[i],M)} for i,tt in enumerate(t)]; a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8') as f:
        f.write('# input_sha256='+sha(a.input)+'\n# computation_sha256='+sha(Path(__file__))+'\n# C90_order=source_per_capacity; fractional_final_cell=true\n'); w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'wrote {a.output} from {a.input} ({len(rows)} endpoints)')
if __name__=='__main__': main()
