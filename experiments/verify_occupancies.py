"""Independent checks of occupancy causality with stochastic transitions.
This is a numerical proof audit, not a comparative performance benchmark.
"""
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'.vendor'
if sys.version_info[:2]==(3,8) and VENDOR.exists():sys.path.insert(0,str(VENDOR))
import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize,linprog

stats={'episodes':0,'replans':0,'exact_opportunity_checks':0,'max_occupancy_error':0.,'max_feasibility_error':0.}

def run(seed):
    rng=np.random.default_rng(seed);L=3;m=10;blocks=[slice(0,2),slice(2,6),slice(6,10)]
    transitions=[rng.dirichlet([1,1],2),rng.dirichlet([1,1],4)]
    A=np.zeros((5,m));A[0,:2]=1;rhs=np.array([1.,0.,0.,0.,0.])
    for l in [1,2]:
        for state in range(2):
            row=1+(l-1)*2+state;start=blocks[l].start+2*state
            A[row,start:start+2]=1;A[row,blocks[l-1]]=-transitions[l-1][:,state]
    gmean=np.array([.05,.75]*5);B=1.2
    def optimize(target,c,mu,prefix,low,projection=False):
        eq=A.copy();erhs=rhs.copy()
        if prefix:
            eq=np.vstack([eq,np.eye(m)[:prefix]]);erhs=np.r_[erhs,target[:prefix]]
        _,R,piv=qr(eq.T,pivoting=True,mode='economic');rank=np.linalg.matrix_rank(R)
        select=piv[:rank];eq=eq[select];erhs=erhs[select]
        if not projection and mu==0:
            res=linprog(c,A_eq=eq,b_eq=erhs,A_ub=low[None,:],b_ub=[B],bounds=[(0,1)]*m,method='highs')
            assert res.success,res.message
            return res.x
        if projection:
            cons=[{'type':'eq','fun':lambda x:eq@x-erhs,'jac':lambda x:eq},
                  {'type':'ineq','fun':lambda x:B-low@x,'jac':lambda x:-low}]
            res=minimize(lambda x:.5*np.sum((x-target)**2),feasible(),jac=lambda x:x-target,
                constraints=cons,bounds=[(0,1)]*m,method='SLSQP',options={'ftol':1e-11,'maxiter':300})
            assert res.success,res.message
            return res.x
        ae=np.c_[eq,np.zeros((len(erhs),m))]
        ai=np.vstack([np.c_[-np.eye(m),np.eye(m)],np.c_[np.eye(m),np.eye(m)],np.r_[-low,np.zeros(m)][None,:]])
        off=np.r_[target,-target,B]
        cons=[{'type':'eq','fun':lambda x:ae@x-erhs,'jac':lambda x:ae},
              {'type':'ineq','fun':lambda x:ai@x+off,'jac':lambda x:ai}]
        res=minimize(lambda x:c@x[:m]+mu/2*x[m:].sum()**2,np.r_[target,np.zeros(m)],
            jac=lambda x:np.r_[c,np.full(m,mu*x[m:].sum())],constraints=cons,bounds=[(0,1)]*m+[(0,None)]*m,
            method='SLSQP',options={'ftol':1e-11,'maxiter':300})
        assert res.success,res.message
        return res.x[:m]
    def feasible():
        # Always choose action zero: feasible even for the true resource budget.
        q=np.zeros(m);q[0]=1
        for l in [1,2]:q[blocks[l]][::2]=q[blocks[l-1]]@transitions[l-1]
        return q
    y=feasible();S=np.zeros(L);E=0.;gsum=np.zeros(m);damage=0.;violation=0.;refreg=0.;totalloss=0.;costsum=np.zeros(m)
    for n in range(1,97):
        beta=1. if n==1 else math.sqrt(math.log(2*m*n*n/.05)/(2*(n-1)))
        low=np.maximum(0.,gsum/max(1,n-1)-beta)
        assert np.all(low<=gmean+1e-10)
        y=optimize(y,None,None,0,low,True);b=y.copy();eta=math.sqrt(2*L)/math.sqrt(m+E)
        c=rng.uniform(0,1,m);q=b.copy();oldS=S.copy();deltas=[];updates=[];executed=np.zeros(m);stateprob=np.ones(1)
        for l in range(L):
            M=1-c if l==0 and n<=48 else c.copy()
            before=q.copy();mu=math.sqrt(oldS[l])/(2*(L-l))
            oracle=optimize(before,c,0.,blocks[l].start,low)
            opp=c@(before-oracle)
            q=optimize(before,M,mu,blocks[l].start,low)
            d=q-before;delta=c@d;deltas.append(delta)
            resid=max(np.max(np.abs(A@q-rhs)),max(0.,low@q-B),np.max(np.abs(d[:blocks[l].start])) if l else 0.)
            stats['max_feasibility_error']=max(stats['max_feasibility_error'],float(resid));assert resid<2e-7
            mat=q[blocks[l]].reshape(-1,2);mass=mat.sum(1);policy=np.divide(mat,mass[:,None],out=np.full_like(mat,.5),where=mass[:,None]>1e-12)
            executed[blocks[l]]=(stateprob[:,None]*policy).ravel()
            if l<L-1:stateprob=executed[blocks[l]]@transitions[l]
            if np.all(M==c):
                bound=opp if mu==0 else min(opp/2,opp*opp/(mu*(2*(L-l))**2))
                assert -delta>=bound-2e-6,(-delta,bound)
                stats['exact_opportunity_checks']+=1
            updates.append((max(0.,(c-M)@d)/np.abs(d).sum())**2 if delta>1e-8 else 0.)
            stats['replans']+=1
        err=np.max(np.abs(executed-q));stats['max_occupancy_error']=max(stats['max_occupancy_error'],float(err));assert err<2e-7
        assert abs(c@(q-b)-sum(deltas))<2e-7
        S+=updates;damage+=max(0.,c@(q-b));violation+=max(0.,gmean@q-B)
        assert damage<=4*sum((L-l)*math.sqrt(S[l]) for l in range(L))+1e-5
        assert max(0.,gmean@q-B)<=2*L*beta+1e-7
        y=optimize(y-eta*c,None,None,0,low,True);E+=c@c
        gsum+=rng.uniform(gmean-.05,gmean+.05);costsum+=c;totalloss+=c@q;stats['episodes']+=1
    comparator=linprog(costsum,A_eq=A,b_eq=rhs,A_ub=gmean[None,:],b_ub=[B],bounds=[(0,1)]*m,method='highs').fun
    bound=2*math.sqrt(2*L)*math.sqrt(m+E)+4*sum((L-l)*math.sqrt(S[l]) for l in range(L))
    assert totalloss-comparator<=bound+1e-5

if __name__=='__main__':
    for seed in range(5):run(seed);print('stochastic-flow seed',seed,'passed',flush=True)
    (ROOT/'experiments'/'results'/'occupancy_audit.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    print(json.dumps(stats,indent=2))
