"""Reproduce synthetic mechanism tests. Python>=3.8; numpy, scipy, matplotlib.
All losses are exact occupancy expectations. No external datasets are used.
Run: python experiments/run_experiments.py
"""
import csv,json,math,os,platform,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'.vendor'
if VENDOR.exists() and sys.version_info[:2]==(3,8):
    sys.path.insert(0,str(VENDOR))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache'/'matplotlib'))
import numpy as np
from scipy.optimize import minimize,linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'experiments'/'results';OUT.mkdir(parents=True,exist_ok=True)
METHODS=['Reference','Departure','Rolling','Fixed','Hedge','EAFR-OMD']
STAGE_METHODS=METHODS+['SA-EAFR-OMD']
RECOVERY_METHODS=METHODS+['Fresh-copy EAFR-OMD','SA-EAFR-OMD']
RESOURCE_METHODS=['Reference','Rolling','EAFR-OMD']
COLOR_BY_METHOD={'Reference':'#777777','Departure':'#CC79A7','Rolling':'#D55E00',
                 'Fixed':'#56B4E9','Hedge':'#E69F00','EAFR-OMD':'#0072B2',
                 'Fresh-copy EAFR-OMD':'#009E73','SA-EAFR-OMD':'#000000'}
DISPLAY_NAME={'EAFR-OMD':'EAFR\u2013OMD','SA-EAFR-OMD':'SA\u2013EAFR\u2013OMD',
              'Fresh-copy EAFR-OMD':'Fresh-copy EAFR\u2013OMD'}
STATS={'qp_calls':0,'lp_calls':0,'max_feasibility_residual':0.,'max_objective_increase':0.,
       'shared_master_mass_checks':0,'max_shared_master_mass_residual':0.}
EQ=np.array([[1.,1.,0.,0.,0.],[0.,0.,1.,1.,1.]])
G=np.array([1.,0.,1.,1.,0.])


class SharedControllerMaster:
    """Compressed every-start Prod master over three persistent controllers."""
    def __init__(self,N,L,d=3):
        assert d==3
        self.n=0;self.N=N;self.L=L;self.d=d
        self.J=math.ceil(math.log2(math.sqrt(N)))
        self.rates=2.**(-np.arange(self.J+1)-1)
        self.pi=1./(d*N*(self.J+1))
        self.W=np.zeros((d,self.J+1))
    @property
    def exact_suffix_bound(self):
        return self.L*math.log(3*self.N*(self.J+1))/math.log(3/2)
    def combine(self,losses):
        losses=np.asarray(losses,dtype=float)
        assert losses.shape==(self.d,) and np.all(np.isfinite(losses))
        assert losses.min()>=-1e-9 and losses.max()<=self.L+1e-9
        self.n+=1;self.W+=self.pi
        controller_weight=(self.W*self.rates).sum(1)
        alpha=controller_weight/controller_weight.sum()
        normalized=losses/self.L;mix=float(alpha@normalized)
        self.W*=1.+self.rates[None,:]*(mix-normalized[:,None])
        residual=abs(self.W.sum()-self.n/self.N)
        STATS['shared_master_mass_checks']+=1
        STATS['max_shared_master_mass_residual']=max(
            STATS['max_shared_master_mass_residual'],float(residual))
        assert residual<2e-12,(self.n,self.W.sum(),self.n/self.N)
        return mix*self.L

def project(v):
    cons=[{'type':'eq','fun':lambda q:EQ@q-1,'jac':lambda q:EQ},
          {'type':'ineq','fun':lambda q:1-G@q,'jac':lambda q:-G}]
    res=minimize(lambda q:.5*np.sum((q-v)**2),np.array([0.,1.,.5,.5,0.]),
        jac=lambda q:q-v,bounds=[(0.,1.)]*5,constraints=cons,method='SLSQP',
        options={'ftol':1e-11,'maxiter':150})
    assert res.success,res.message
    return res.x

def replan(q0,M,mu,stage):
    eq=EQ.copy();rhs=np.ones(2)
    if stage:eq=np.vstack([eq,np.eye(5)[:1]]);rhs=np.r_[rhs,q0[:1]]
    if mu==0:
        STATS['lp_calls']+=1
        res=linprog(M,A_ub=G[None,:],b_ub=[1.],A_eq=eq,b_eq=rhs,
                    bounds=[(0.,1.)]*5,method='highs')
        assert res.success,res.message
        q=res.x
    else:
        STATS['qp_calls']+=1
        Ae=np.c_[eq,np.zeros((len(rhs),5))]
        Ai=np.vstack([np.c_[-np.eye(5),np.eye(5)],np.c_[np.eye(5),np.eye(5)],np.r_[-G,np.zeros(5)][None,:]])
        offset=np.r_[q0,-q0,1.]
        cons=[{'type':'eq','fun':lambda x:Ae@x-rhs,'jac':lambda x:Ae},
              {'type':'ineq','fun':lambda x:Ai@x+offset,'jac':lambda x:Ai}]
        res=minimize(lambda x:M@x[:5]+mu/2*x[5:].sum()**2,np.r_[q0,np.zeros(5)],
            jac=lambda x:np.r_[M,np.full(5,mu*x[5:].sum())],constraints=cons,
            bounds=[(0.,1.)]*5+[(0.,None)]*5,method='SLSQP',options={'ftol':1e-11,'maxiter':250})
        assert res.success,res.message
        q=res.x[:5]
    err=max(abs(eq@q-rhs).max(),max(0.,G@q-1),max(0.,-q.min()))
    obj=M@(q-q0)+mu/2*np.abs(q-q0).sum()**2
    STATS['max_feasibility_residual']=max(STATS['max_feasibility_residual'],float(err))
    STATS['max_objective_increase']=max(STATS['max_objective_increase'],float(obj))
    assert err<2e-7 and obj<2e-7,(err,obj)
    return q

def separation(seed,N=256,record_details=False):
    rng=np.random.default_rng(seed);y=np.array([0.,1.,.5,.5,0.]);yd=y.copy()
    E=Ed=0.;S=np.zeros(2);loss=np.zeros((N,7));damage=loss.copy();sm=SharedControllerMaster(N,2)
    opportunity=np.zeros((N,2));captured=opportunity.copy();hw=np.ones(2)/2
    heta=math.sqrt(8*math.log(2)/N)/2;costs=[]
    # Optional traces support the appendix horizon sweep without changing this
    # generator, its default outputs, or its original main-figure experiment.
    if record_details:
        occupancies=np.zeros((N,6,5));early_eafr=np.zeros((N,5))
        score_history=np.zeros((N,2));bits=np.zeros(N,dtype=np.int8)
    for n in range(N):
        z=rng.integers(2);c=np.array([0.,0.,float(z),float(1-z),1.]);M=np.array([0.,1.,0.,0.,0.])
        eta=2/math.sqrt(5+E);etad=2/math.sqrt(5+Ed);b=y.copy()
        dep=project(yd-etad*M);rolling=b.copy();fixed=b.copy();q=b.copy();oldS=S.copy()
        for l in range(2):
            f=M if l==0 else c
            rolling=replan(rolling,f,0.,l);fixed=replan(fixed,f,1.,l)
            before=q.copy();oracle=replan(before,c,0.,l)
            opportunity[n,l]=max(0.,c@(before-oracle))
            q=replan(before,f,math.sqrt(oldS[l])/(2*(2-l)),l)
            if record_details and l==0:early_eafr[n]=q
            d=q-before;delta=c@d;captured[n,l]=-delta
            # Only oldS is used this episode; no current-feedback leakage.
            if delta>1e-8:S[l]+=(max(0.,(c-f)@d)/np.abs(d).sum())**2
        hedge=hw[0]*b+hw[1]*rolling
        plans=np.array([b,dep,rolling,fixed,hedge,q]);loss[n,:6]=plans@c
        if record_details:
            occupancies[n]=plans;score_history[n]=S;bits[n]=z
        loss[n,6]=sm.combine([c@b,c@q,c@rolling])
        damage[n]=np.maximum(loss[n]-c@b,0.)
        hw*=np.exp(-heta*np.array([c@b,c@rolling]));hw/=hw.sum()
        y=project(y-eta*c);yd=project(yd-etad*c);E+=c@c;Ed+=np.sum((c-M)**2);costs.append(c)
    opt=linprog(np.sum(costs,axis=0),A_ub=G[None,:],b_ub=[1.],A_eq=EQ,b_eq=np.ones(2),bounds=[(0,1)]*5,method='highs').fun
    assert damage[:,5].sum()<=4*(2*np.sqrt(S[0])+np.sqrt(S[1]))+1e-5
    result={'loss':loss,'damage':damage,'regret':loss.sum(0)-opt,'opportunity':opportunity,
            'captured':captured,'shared_master_bound':sm.exact_suffix_bound}
    if record_details:
        result.update({'occupancies':occupancies,'early_eafr':early_eafr,
                       'stage_score_history':score_history,'bits':bits,
                       'static_comparator_loss':float(opt)})
    return result

def scalar_plan(b,gap,S,cap=1.,fixed=None):
    mu=np.sqrt(np.asarray(S))/2 if fixed is None else np.broadcast_to(fixed,np.asarray(S).shape)
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.where(mu>0,np.clip(b-gap/(4*mu),0.,cap),cap if gap<0 else 0.)

def recovery(seed,N=2048):
    rng=np.random.default_rng(seed);gaps=rng.choice([-1.,1.],size=N)
    J=math.ceil(math.log2(math.sqrt(N)));rates=2.**(-np.arange(J+1)-1)
    w=np.ones((N+1,J+1))/((N+1)*(J+1));S=np.zeros(N+1)
    loss=np.zeros((N,8));damage=loss.copy();sm=SharedControllerMaster(N,1);y=yd=.5;E=Ed=0.;hw=np.ones(2)/2
    heta=math.sqrt(8*math.log(2)/N)
    for n in range(1,N+1):
        gap=gaps[n-1];c0=float(gap<0);mg=-gap if n<=N//2 else gap;b=y
        eta=math.sqrt(2)/math.sqrt(2+E);etad=math.sqrt(2)/math.sqrt(2+Ed)
        dep=np.clip(yd-etad*mg/2,0.,1.);rolling=float(mg<0);fixed=scalar_plan(b,mg,0.,fixed=1.)
        fresh=scalar_plan(b,mg,S[1:n+1]);active=np.r_[b,fresh]
        alpha=(w[:n+1]*rates).sum(1);alpha/=alpha.sum();master=alpha@active
        hedge=hw[0]*b+hw[1]*rolling
        plans=np.array([b,dep,rolling,fixed,hedge,fresh[0],master])
        loss[n-1,:7]=c0+gap*plans
        loss[n-1,7]=sm.combine([c0+gap*b,c0+gap*fresh[0],c0+gap*rolling])
        damage[n-1]=np.maximum(loss[n-1]-(c0+gap*b),0.)
        eloss=c0+gap*active;mixloss=c0+gap*master
        w[:n+1]*=1.+(mixloss-eloss[:,None])*rates
        assert abs(w.sum()-1.)<2e-12
        S[1:n+1]+=(gap*(fresh-b)>1e-13)*(abs(gap-mg)/2)**2
        hw*=np.exp(-heta*np.array([c0+gap*b,c0+gap*rolling]));hw/=hw.sum()
        y=float(np.clip(y-eta*gap/2,0.,1.));yd=float(np.clip(yd-etad*gap/2,0.,1.))
        E+=1.;Ed+=2. if mg!=gap else 0.
    h=N//2;fresh_copy_bound=min(math.log((N+1)*(J+1))/rates+rates*h)
    shared_exact_suffix_bound=sm.exact_suffix_bound
    assert loss[h:,6].sum()<=fresh_copy_bound+1e-7
    assert loss[h:,7].sum()<=shared_exact_suffix_bound+1e-7
    assert damage[:,5].sum()<=4*np.sqrt(S[1])+1e-7
    opt=min(np.sum(gaps<0),np.sum(gaps>0))
    return {'loss':loss,'damage':damage,'regret':loss.sum(0)-opt,
            'suffix_loss':loss[h:].sum(0),'fresh_copy_bound':fresh_copy_bound,
            'shared_exact_suffix_bound':shared_exact_suffix_bound}

def resource(seed,N=2048):
    rng=np.random.default_rng(seed);rs=rng.uniform(.6,1.,N);gaps=rng.choice([-1.,1.],N);flip=rng.random(N)<.35
    y=.5;E=S=gsum=0.;loss=np.zeros((N,3));violation=loss.copy();damage=loss.copy()
    for n in range(1,N+1):
        beta=1. if n==1 else math.sqrt(math.log(4*n*n/.05)/(2*(n-1)))
        low=0. if n==1 else max(0.,gsum/(n-1)-beta)
        cap=min(1.,.4/low) if low else 1.;y=min(y,cap)
        gap=gaps[n-1];c0=float(gap<0);mg=-gap if flip[n-1] else gap;eta=math.sqrt(2)/math.sqrt(2+E)
        p=float(scalar_plan(y,mg,S,cap));rolling=cap if mg<0 else 0.;plans=np.array([y,rolling,p])
        loss[n-1]=c0+gap*plans;violation[n-1]=np.maximum(.8*plans-.4,0.);damage[n-1]=np.maximum(gap*(plans-y),0.)
        if gap*(p-y)>1e-13:S+=(abs(gap-mg)/2)**2
        y=float(np.clip(y-eta*gap/2,0.,cap));E+=1.;gsum+=rs[n-1]
    opt=np.sum(gaps<0)+min(0.,gaps.sum()*.5)
    assert damage[:,2].sum()<=4*np.sqrt(S)+1e-7
    return {'loss':loss,'violation':violation,'damage':damage,'regret':loss.sum(0)-opt}

def checks():
    rng=np.random.default_rng(917);S=10**rng.uniform(-12,8,50000);z=rng.random(50000)
    assert np.all(np.minimum(z,z*z/(2*np.sqrt(S)))<=2*z*z/(np.sqrt(S+z*z)+np.sqrt(S))+1e-12)
    b=rng.random(50000);gap=rng.uniform(-1,1,50000);mu=10**rng.uniform(-5,5,50000)
    q=np.clip(b-gap/(4*mu),0.,1.);opt=np.where(gap<0,1.,0.);A=gap*(b-opt)
    assert np.all(gap*(b-q)+1e-12>=np.minimum(A/2,A*A/(4*mu)))
    # Exact compression check against explicitly stored every-start specialists.
    N,L,d=64,2.,3;J=math.ceil(math.log2(math.sqrt(N)))
    rates=2.**(-np.arange(J+1)-1);pi=1./(d*N*(J+1))
    explicit=np.zeros((N,d,J+1));compressed=np.zeros((d,J+1))
    max_mix=max_weight=0.
    for n in range(N):
        losses=rng.uniform(0.,L,d);explicit[n]=pi;compressed+=pi
        cw_exp=(explicit[:n+1]*rates[None,None,:]).sum((0,2))
        cw_cmp=(compressed*rates[None,:]).sum(1)
        alpha_exp=cw_exp/cw_exp.sum();alpha_cmp=cw_cmp/cw_cmp.sum()
        mix_exp=float(alpha_exp@losses/L);mix_cmp=float(alpha_cmp@losses/L)
        max_mix=max(max_mix,abs(mix_exp-mix_cmp))
        factor=1.+rates[None,:]*(mix_exp-losses[:,None]/L)
        explicit[:n+1]*=factor[None,:,:];compressed*=factor
        max_weight=max(max_weight,float(np.abs(compressed-explicit[:n+1].sum(0)).max()))
    assert max_mix<1e-12 and max_weight<1e-12
    return {'scalar_potential_checks':50000,'opportunity_checks':50000,
            'compression_episodes':N,'max_compressed_mix_discrepancy':max_mix,
            'max_aggregate_weight_discrepancy':max_weight}

def summary(runs,names,experiment):
    rows=[]
    for i,name in enumerate(names):
        row={'experiment':experiment,'method':name,'runs':len(runs),'episodes':len(runs[0]['loss'])}
        for k in ['loss','damage','violation','regret','suffix_loss']:
            if k not in runs[0]:continue
            vals=np.array([r[k].sum(0)[i] if r[k].ndim==2 else r[k][i] for r in runs])
            row[k+'_mean']=float(vals.mean());row[k+'_se']=float(vals.std(ddof=1)/np.sqrt(len(vals)))
        rows.append(row)
    return rows

def main():
    start=time.perf_counter();check=checks();sep=[]
    for seed in range(10):
        sep.append(separation(seed));print('separation',seed,'done',flush=True)
    rec=[recovery(seed) for seed in range(20)];print('recovery done',flush=True)
    res=[resource(seed) for seed in range(20)]
    rows=(summary(sep,STAGE_METHODS,'stage_separation')+
          summary(rec,RECOVERY_METHODS,'bad_to_exact')+
          summary(res,RESOURCE_METHODS,'stochastic_resource'))
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (OUT/'summary.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    np.savez_compressed(
        OUT/'raw_runs.npz',
        separation_method_names=np.asarray(STAGE_METHODS),
        separation_loss=np.array([r['loss'] for r in sep]),
        separation_damage=np.array([r['damage'] for r in sep]),
        separation_opportunity=np.array([r['opportunity'] for r in sep]),
        separation_captured=np.array([r['captured'] for r in sep]),
        separation_shared_master_bound=np.array([r['shared_master_bound'] for r in sep]),
        recovery_method_names=np.asarray(RECOVERY_METHODS),
        recovery_loss=np.array([r['loss'] for r in rec]),
        recovery_damage=np.array([r['damage'] for r in rec]),
        recovery_fresh_copy_bound=np.array([r['fresh_copy_bound'] for r in rec]),
        recovery_shared_exact_suffix_bound=np.array([r['shared_exact_suffix_bound'] for r in rec]),
        resource_method_names=np.asarray(RESOURCE_METHODS),
        resource_loss=np.array([r['loss'] for r in res]),
        resource_violation=np.array([r['violation'] for r in res]))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(1,3,figsize=(9.8,3.25))
    for i,name in enumerate(STAGE_METHODS):
        a=np.array([r['loss'][:,i].cumsum() for r in sep]);ax[0].plot(np.arange(1,257),a.mean(0),color=COLOR_BY_METHOD[name],label=DISPLAY_NAME.get(name,name),ls='--' if i in [1,3] else '-')
    ax[0].set(xlabel='Episode',ylabel='Cumulative loss')
    for i in [0,2,4,5,6,7]:
        a=np.array([r['loss'][1024:,i].cumsum() for r in rec]);avg=a.mean(0);se=a.std(0,ddof=1)/np.sqrt(len(a))
        name=RECOVERY_METHODS[i];ax[1].plot(np.arange(1,1025),avg,color=COLOR_BY_METHOD[name],label=DISPLAY_NAME.get(name,name));ax[1].fill_between(np.arange(1,1025),avg-se,avg+se,color=COLOR_BY_METHOD[name],alpha=.15)
    ax[1].set(xlabel='Episodes after the change',ylabel='Suffix cumulative loss')
    for i,name in enumerate(RESOURCE_METHODS):
        a=np.array([r['violation'][:,i].cumsum() for r in res]);ax[2].plot(np.arange(1,2049),a.mean(0),color=COLOR_BY_METHOD[name],label=DISPLAY_NAME.get(name,name))
    ax[2].set(xlabel='Episode',ylabel='Positive expected-resource violation')
    handles=[];labels=[]
    for axis in ax:
        for handle,label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:handles.append(handle);labels.append(label)
    fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.12));fig.tight_layout()
    fig.savefig(ROOT/'figures'/'mechanisms.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures'/'mechanisms.png',dpi=180,bbox_inches='tight')
    from scipy.stats import ttest_rel,t
    comparisons=[]
    contrasts=[
        ('Stage: EAFR-OMD minus Hedge',sep,'loss',5,4),
        ('Stage: SA-EAFR-OMD minus EAFR-OMD',sep,'loss',6,5),
        ('Recovery: SA-EAFR-OMD minus EAFR-OMD',rec,'suffix_loss',7,5),
        ('Recovery: SA-EAFR-OMD minus Rolling',rec,'suffix_loss',7,2),
        ('Recovery: SA-EAFR-OMD minus fresh-copy EAFR-OMD',rec,'suffix_loss',7,6)]
    for title,runset,key,i,j in contrasts:
        a=np.array([r[key].sum(0)[i] if r[key].ndim==2 else r[key][i] for r in runset]);b=np.array([r[key].sum(0)[j] if r[key].ndim==2 else r[key][j] for r in runset]);d=a-b;se=d.std(ddof=1)/np.sqrt(len(d));radius=t.ppf(.975,len(d)-1)*se
        comparisons.append({'contrast':title,'mean_difference':float(d.mean()),'ci95':[float(d.mean()-radius),float(d.mean()+radius)],'paired_t_pvalue':float(ttest_rel(a,b).pvalue)})
    (OUT/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2),encoding='utf8')
    recovery_N=len(rec[0]['loss']);recovery_L=1
    recovery_J=math.ceil(math.log2(math.sqrt(recovery_N)))
    theorem_bound=recovery_L*math.log(3*recovery_N*(recovery_J+1))/math.log(3/2)
    shared_suffix=np.array([r['suffix_loss'][7] for r in rec])
    assert all(abs(r['shared_exact_suffix_bound']-theorem_bound)<1e-12 for r in rec)
    assert np.all(shared_suffix<=theorem_bound+1e-7)
    assert STATS['shared_master_mass_checks']==10*256+20*2048
    report={
        'checks':check,'solver':STATS,'seconds':time.perf_counter()-start,
        'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),
        'numpy':np.__version__,'scipy':__import__('scipy').__version__,
        'matplotlib':matplotlib.__version__,
        'seed_ranges':{'separation':[0,9],'recovery':[0,19],'resource':[0,19]},
        'shared_master':{
            'controllers':['Reference','EAFR-OMD','Rolling'],'d':3,
            'N':recovery_N,'J':recovery_J,
            'rates':list(2.**(-np.arange(recovery_J+1)-1)),
            'pi':1./(3*recovery_N*(recovery_J+1)),
            'weight_mass_identity':'W.sum() == n/N'},
        'exact_suffix_theorem':{
            'formula':'L*log(3*N*(J+1))/log(3/2)','L':recovery_L,
            'bound':theorem_bound,'runs_checked':len(rec),
            'observed_suffix_loss_mean':float(shared_suffix.mean()),
            'observed_suffix_loss_se':float(shared_suffix.std(ddof=1)/np.sqrt(len(shared_suffix))),
            'max_observed_suffix_loss':float(shared_suffix.max()),
            'min_slack':float((theorem_bound-shared_suffix).min()),'all_passed':True},
        'fresh_copy_master':{'exact_suffix_bound':rec[0]['fresh_copy_bound']}}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
