"""Reproducible COMP3821 poster figures from Kevin's raw experiment CSVs.
Run from repository root: python make_figures_fixed.py
Reads extracted results/cc and results/mission, or results/cc_mission_results.zip.
Requires pandas, numpy, scipy, matplotlib. Saves 300-dpi PNGs in figures/.
"""
from pathlib import Path
import io, zipfile
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT=Path('results'); OUT=Path('figures'); OUT.mkdir(exist_ok=True)
SCALE=1.0
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':12*SCALE,'axes.titlesize':14*SCALE,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
C={'aco_full_reset':'#3B4B6B','aco_retention':'#E8590C','aco_fast_evaporation':'#F59F00','aco_selective_reset':'#0C8F8F','vns':'#7048E8','greedy':'#868E96'}
LABEL={'aco_full_reset':'Full reset','aco_retention':'Retention','aco_fast_evaporation':'Fast evap.','aco_selective_reset':'Selective','vns':'VNS','greedy':'Greedy'}
zip_path=ROOT/'cc_mission_results.zip'
archive=zipfile.ZipFile(zip_path) if zip_path.exists() else None

def read_csv(relative):
    local=ROOT/relative
    if local.exists(): return pd.read_csv(local)
    if archive and relative in archive.namelist(): return pd.read_csv(io.BytesIO(archive.read(relative)))
    raise FileNotFoundError(f'Missing {local} (or {relative} in {zip_path})')

def save(name):
    plt.savefig(OUT/name,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close()
    print('Saved',OUT/name)

# Pair within each scenario/seed/change cell, then average within scenario/seed
# before constructing uncertainty intervals. 60 independent scenario/seed units.
KEY=['scenario','trial','target_fraction','locality']
UNIT=['scenario','trial']
def paired(df, metric, baseline='aco_full_reset'):
    p=df.pivot_table(index=KEY,columns='planner',values=metric,aggfunc='mean')
    if baseline not in p: raise ValueError(f'Baseline {baseline} missing')
    diffs=p.sub(p[baseline],axis=0)
    return diffs.groupby(level=UNIT).mean()

def stats(x):
    x=pd.Series(x).dropna().astype(float)
    n=len(x)
    if n<2: raise ValueError('Not enough paired observations')
    m=x.mean(); se=x.std(ddof=1)/np.sqrt(n); ci=student_t.ppf(.975,n-1)*se
    return m,m-ci,m+ci

camp=read_csv('cc/camp.csv')
# 1: Paired differences against full reset
fig,ax=plt.subplots(1,2,figsize=(10.5,2.8),sharey=True)
order=['greedy','vns','aco_retention','aco_fast_evaporation','aco_selective_reset']
for a,(metric,title) in zip(ax,[('margin','Margin'),('total_overdue','Total overdue')]):
    paired_units=paired(camp,metric)
    for i,pl in enumerate(order):
        m,lo,hi=stats(paired_units[pl])
        a.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color=C[pl],ms=8,capsize=4,lw=2)
    a.axvline(0,color=C['aco_full_reset'],ls='--',lw=1.6)
    a.set_yticks(range(len(order)),[LABEL[p] for p in order]);a.invert_yaxis()
    a.set_title(title,loc='left');a.set_xlabel('Difference vs full reset (95% CI)  ← better')
    a.grid(axis='x',alpha=.2)
fig.subplots_adjust(left=0.14,right=0.99,top=0.84,bottom=0.27,wspace=0.20);save('forest.png')

# 2: Recovery from actual traces; take last available best score at/before each budget.
# Each curve paired with full reset *within the same parameter configuration*.
TRACEKEY=KEY+['planner']
def checkpoint(trace,metric,evaluations):
    s=trace[trace.evaluations<=evaluations].sort_values('evaluations')
    if s.empty: raise ValueError(f'No trace rows at {evaluations} evaluations')
    return s.groupby(TRACEKEY,as_index=False).tail(1)[TRACEKEY+[metric]]

normal=read_csv('cc/camp_trace.csv')
low=read_csv('cc/camp_low_rho_trace.csv')
series=[('Retention (tuned)',normal,'aco_retention','-'),('Selective reset (tuned)',normal,'aco_selective_reset','-'),('Retention (low evaporation)',low,'aco_retention','--')]
evs=[20,60,120,300,600,1200]
fig,a=plt.subplots(figsize=(8.5,4.8))
for label,tr,pl,ls in series:
    vals=[stats(paired(checkpoint(tr,'best_total_overdue',e),'best_total_overdue')[pl]) for e in evs]
    arr=np.asarray(vals)
    a.plot(evs,arr[:,0],ls,marker='o',lw=2.3,ms=5,color=C[pl],label=label)
    a.fill_between(evs,arr[:,1],arr[:,2],alpha=.12,color=C[pl])
a.axhline(0,color=C['aco_full_reset'],ls='--',lw=1.6,label='Full reset reference')
a.set_xscale('log');a.set_xticks(evs,[str(e) for e in evs]);a.minorticks_off()
a.set_xlabel('Planning evaluations after change');a.set_ylabel('Δ total overdue vs full reset (lower is better)')
a.set_title('Recovery after a fire change',loc='left');a.grid(alpha=.2);a.legend(fontsize=9)
fig.tight_layout();save('recovery.png')

# 3: Budget: use actual ACO/VNS runs at each budget.
# Note: separate budget files can have different trial counts. Label accordingly.
files={1200:{'vns':'cc/camp.csv','aco_full_reset':'cc/camp.csv','aco_retention':'cc/camp.csv'},
       4800:{'vns':'cc/camp_vns4800.csv','aco_full_reset':'cc/camp_aco4800.csv','aco_retention':'cc/camp_aco4800.csv'},
       12000:{'vns':'cc/camp_vns12000.csv','aco_full_reset':'cc/camp_aco12000.csv','aco_retention':'cc/camp_aco12000.csv'}}
fig,axes=plt.subplots(1,2,figsize=(10,4))
for a,(metric,title) in zip(axes,[('margin','Margin (worst point)'),('total_overdue','Total overdue')]):
    for pl in ['vns','aco_full_reset','aco_retention']:
        ys=[]
        for budget,groups in files.items():
            d=read_csv(groups[pl]); d=d[d.planner==pl]
            # Average over change settings per scenario/seed first.
            ys.append(d.groupby(UNIT)[metric].mean().mean())
        a.plot([str(b) for b in files],ys,'-o',lw=2.5,ms=7,color=C[pl],label=LABEL[pl])
    a.set_title(title,loc='left');a.set_xlabel('Evaluations per replan');a.grid(alpha=.2)
axes[0].set_ylabel('Mean (lower is better)');axes[0].legend(fontsize=9)
fig.tight_layout();save('budget.png')

# 4: Stage 40: compare matching cycle-direction runs, untuned vs tuned.
# Do NOT mix the tuned 'up' experiment with untuned 'cycle'.
untuned=read_csv('cc/camp_untuned_cycle.csv')
tuned=read_csv('cc/camp_cycle.csv')
fig,a=plt.subplots(figsize=(7.5,4))
stages=[6,40,80]; xx=np.arange(3)
for offset,(label,df,color) in zip([-.19,.19],[('Untuned',untuned,'#C92A2A'),('Tuned',tuned,C['aco_selective_reset'])]):
    vals=[]
    for st in stages:
        subset=df[df.scenario.str.endswith('stage'+str(st))]
        vals.append(paired(subset,'margin')['aco_retention'].mean())
    a.bar(xx+offset,vals,.36,color=color,label=label)
a.set_xticks(xx,[f'Stage {s}' for s in stages]);a.axhline(0,color='#333',lw=1)
a.set_ylabel('Retention margin − full reset');a.set_title('Stage-specific retention difference',loc='left')
a.legend();a.grid(axis='y',alpha=.2);fig.tight_layout();save('stage40.png')

# 5: 24-hour Camp Fire missions — Claude-style two-panel plot
# Data and seed dots are calculated directly from the mission CSVs.
mission_pl=['greedy','vns','aco_full_reset','aco_retention','aco_fast_evaporation','aco_selective_reset']
fig,axes=plt.subplots(1,2,figsize=(10.4,3.55))
for a,(n,f) in zip(axes,[(6,'mission/d6.csv'),(10,'mission/d10.csv')]):
    df=read_csv(f)
    piv=df.pivot_table(index='seed',columns='planner',values='true_overdue_mean')
    for i,pl in enumerate(mission_pl):
        a.bar(i,piv[pl].mean(),color=C[pl],width=.72,zorder=2)
        jitter=np.random.default_rng(i).uniform(-.16,.16,len(piv))
        a.scatter(i+jitter,piv[pl],s=19,color='#222222',alpha=.5,zorder=3)
    a.set_xlim(-.65,5.65)
    a.set_xticks(range(len(mission_pl)))
    a.set_xticklabels([LABEL[p] for p in mission_pl],rotation=27,ha='right',fontsize=10.5)
    a.set_title(f'{n} drones',loc='left',fontsize=15,pad=8)
    a.grid(axis='y',alpha=.17,zorder=0)
    a.tick_params(axis='y',labelsize=11)
    a.tick_params(axis='x',length=0)
    # Match Claude's deliberately zoomed y-axis, without hiding seed points.
    a.set_ylim((47,60) if n==6 else (36,50))
axes[0].set_ylabel('Overdue points',fontsize=12)
fig.subplots_adjust(left=.085,right=.985,top=.86,bottom=.32,wspace=.20)
save('mission.png')

# 6: Harness flowchart — descriptive diagram, no numeric results hardcoded
fig,a=plt.subplots(figsize=(14,2.2));a.set_xlim(0,14);a.set_ylim(0,2);a.axis('off')
steps=['Warm up ACO,\nsave pheromone\n+ RNG state','Apply change\n5–60% of points,\nlocal or spread','6 planners replan:\nsame state, seed,\nbudget','Re-score,\npaired stats,\nHolm correction']
for i,label in enumerate(steps):
    x=i*3.4+.15;hot=i==1
    a.add_patch(FancyBboxPatch((x,.35),2.9,1.3,boxstyle='round,pad=0.02,rounding_size=0.08',fc='#fff3e8' if hot else '#eef1f6',ec='#E8590C' if hot else '#cfd6e4',lw=2))
    a.text(x+1.45,1,label,ha='center',va='center',fontsize=12*SCALE,fontweight='bold',color='#1d2433')
    if i<3:a.add_patch(FancyArrowPatch((x+2.95,1),(x+3.35,1),arrowstyle='-|>',mutation_scale=20,color='#E8590C',lw=2))
save('harness_flow.png')

# 7: Illustrative pheromone strategies (not measured results)
nodes=[(.4,.3),(1.2,.2),(1.85,.7),(1.3,1.2),(.5,1.1),(1.05,.68)]
edges=[(0,1),(1,2),(2,3),(3,4),(4,0),(0,5),(1,5),(2,5),(3,5),(4,5)]
cards=[('Retention','aco_retention','Keep pheromones'),('Fast evaporation','aco_fast_evaporation','Accelerate decay'),('Selective reset','aco_selective_reset','Reset near changes'),('Full reset','aco_full_reset','Clear pheromones')]
fig,axes=plt.subplots(1,4,figsize=(12,3))
for a,(name,pl,desc) in zip(axes,cards):
    a.set_xlim(0,2.25);a.set_ylim(-.4,1.5);a.set_aspect('equal');a.axis('off')
    for i,j in edges:
        thin=pl=='aco_full_reset' or (pl=='aco_selective_reset' and 5 in (i,j))
        a.plot(*zip(nodes[i],nodes[j]),color='#b8bfca' if thin else C[pl],lw=1.5 if thin else 5,solid_capstyle='round')
    for i,(x,y) in enumerate(nodes):a.add_patch(Circle((x,y),.1 if i==5 else .07,color='#C92A2A' if i==5 else '#2b3445',zorder=3))
    a.text(1.12,-.12,name,ha='center',fontweight='bold',color=C[pl],fontsize=12*SCALE)
    a.text(1.12,-.32,desc,ha='center',fontsize=9*SCALE)
fig.tight_layout();save('strategies.png')
print('Finished: all plots generated from CSVs, except explanatory diagrams.')
