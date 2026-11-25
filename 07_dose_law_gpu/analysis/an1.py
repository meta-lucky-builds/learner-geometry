import json,sys,math
import numpy as np

R=json.load(open(sys.argv[1]))['conditions']
def med(x): return float(np.median(x))
rows=[]
for k,c in R.items():
    t=c['tasks']; s=c['summary']
    rows.append(dict(k=k,grid=c['grid'],model=c['model'],seq=c['seq'],eps=c['eps'],L=c['learner'],N=c['N'],seed=c['seed'],
        A=s['A_final'],D=s['D_cum'],Ahi=s['A_hi_final'],steps=s['steps_total'],unfit=s['unfit_total'],
        dead2=(t[-1]['dead_frac']+[0,0])[1],dead1=(t[-1]['dead_frac']+[0,0])[0],dead2_0=(t[0]['dead_frac']+[0,0])[1],
        psi=t[-1]['psi'],psi0=t[0]['psi'],lam0=t[0]['lam_max'],lamf=t[-1]['lam_max'],tr0=t[0]['trK'],trf=t[-1]['trK'],
        p10_0=t[0]['lam_p10'],p10f=t[-1]['lam_p10'],p50_0=t[0]['lam_p50'],p50f=t[-1]['lam_p50'],p90f=t[-1]['lam_p90'],p90_0=t[0]['lam_p90'],
        er0=t[0]['eff_rank'],erf=t[-1]['eff_rank'],W=t[-1]['W_move_cum'],Draw=t[-1]['D_raw_cum'],
        bot=sum(x['D_band']['BOT'] for x in t[1:]),top=sum(x['D_band']['TOP'] for x in t[1:]),mid=sum(x['D_band']['MID'] for x in t[1:]),
        sat2=(t[-1]['sat_frac']+[0,0])[1],inact2=(t[-1]['inactive_frac']+[0,0])[1],last_steps=t[-1]['steps'],first_steps=t[1]['steps'],
        A_series=[x['A_raw']-t[0]['A_raw'] for x in t]))
print('n',len(rows))
# 1. table by (model,seq,eps,learner,N): median over seeds
from collections import defaultdict
g=defaultdict(list)
for r in rows: g[(r['model'],r['seq'],r['eps'],r['L'],r['N'])].append(r)
print('\n== per condition medians: A_final D_cum Dbot dead2 dead1 psi_f/psi_0 lam_max f/0 trK f/0 p10 f/0 p50 f/0 p90 f/0 effrank f/0 Wmove laststeps/firststeps')
for key in sorted(g,key=lambda x:(x[0],x[1],str(x[2]),x[3],x[4])):
    v=g[key]; m=lambda f: med([f(r) for r in v])
    print(f"{key[0]:6s} {key[1]} eps={str(key[2]):4s} {key[3]:10s} N={key[4]:2d} | A={m(lambda r:r['A']):6.3f} D={m(lambda r:r['D']):8.2f} Dbot={m(lambda r:r['bot']):8.2f} dead2={m(lambda r:r['dead2']):.3f} dead1={m(lambda r:r['dead1']):.3f} psi={m(lambda r:r['psi']/r['psi0']):7.3f} lmax={m(lambda r:r['lamf']/r['lam0']):6.2f} tr={m(lambda r:r['trf']/r['tr0']):6.2f} p10={m(lambda r:r['p10f']/r['p10_0']):6.3f} p50={m(lambda r:r['p50f']/r['p50_0']):6.3f} p90={m(lambda r:r['p90f']/r['p90_0']):6.2f} er={m(lambda r:r['erf']/r['er0']):5.2f} W={m(lambda r:r['W']):7.1f} st={m(lambda r:r['last_steps']/max(r['first_steps'],1)):6.2f} unfit={m(lambda r:r['unfit']):.0f}")
json.dump(rows,open(sys.argv[2],'w'))
