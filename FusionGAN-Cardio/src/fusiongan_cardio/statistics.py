"""Five/ten-seed statistics, exact paired Wilcoxon, Holm, paired bootstrap."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, t as t_dist, rankdata
from sklearn.metrics import f1_score,balanced_accuracy_score
from .utils import write_json

def mean_sd_ci(values,alpha=.05):
    x=np.asarray(values,dtype=float)
    if len(x)<2:raise ValueError('CI requires >=2 independent runs')
    mean=x.mean();sd=x.std(ddof=1)
    half=t_dist.ppf(1-alpha/2,len(x)-1)*sd/(len(x)**.5)
    return {'n':len(x),'mean':float(mean),'sd':float(sd),'ci_95':[float(mean-half),float(mean+half)]}

def paired_significance(a,b):
    """Paired B-A difference; two-sided exact Wilcoxon and paired effect sizes."""
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if len(a)!=len(b) or len(a)<2:raise ValueError('Require matched pairs, N>=2')
    d=b-a;nonzero=d[d!=0]
    if not len(nonzero):p=1.;r=0.
    else:
        if len(nonzero)<=50:
            p=float(wilcoxon(nonzero,zero_method='wilcox',method='exact',alternative='two-sided').pvalue)
        else:p=float(wilcoxon(nonzero,zero_method='wilcox',method='auto',alternative='two-sided').pvalue)
        ranks=rankdata(np.abs(nonzero))
        positive=ranks[nonzero>0].sum();negative=ranks[nonzero<0].sum()
        r=float((positive-negative)/(positive+negative))
    sd=float(d.std(ddof=1));dz=float(d.mean()/sd) if sd>0 else None
    return {'n':len(d),'mean_difference_b_minus_a':float(d.mean()),'wilcoxon_two_sided_p':p,
            'rank_biserial':r,'cohens_dz':dz}

def holm_adjust(pvalues):
    p=np.asarray(pvalues,dtype=float)
    order=np.argsort(p);out=np.zeros(len(p));current=0
    for j,i in enumerate(order):
        current=max(current,min(1.,(len(p)-j)*p[i]));out[i]=current
    return out.tolist()

def compare_seeds(csv_file,output=None):
    """Input CSV columns: seed, method, metric (numeric). All methods paired on seeds."""
    data=pd.read_csv(csv_file)
    if not {'seed','method','metric'}.issubset(data.columns):raise ValueError('Need columns seed,method,metric')
    means={name:mean_sd_ci(group.metric) for name,group in data.groupby('method')}
    pivot=data.pivot(index='seed',columns='method',values='metric').dropna()
    if len(pivot)<2:raise ValueError('Fewer than 2 paired runs')
    if 'fusiongan' not in pivot.columns:raise ValueError('Comparison requires method=fusiongan')
    tests={name:paired_significance(pivot[name],pivot['fusiongan']) for name in pivot.columns if name!='fusiongan'}
    adjusted=holm_adjust([t['wilcoxon_two_sided_p'] for t in tests.values()])
    for item,adj in zip(tests.values(),adjusted):item['holm_adjusted_p']=adj
    report={'per_method':means,'comparisons_vs_fusiongan':tests,'paired_runs':len(pivot)}
    if output:write_json(output,report)
    return report

def _metric(y,p,name):
    pred=p.argmax(axis=1)
    if name=='balanced_accuracy':return balanced_accuracy_score(y,pred)
    if name=='macro_f1':return f1_score(y,pred,average='macro',zero_division=0)
    raise ValueError('Bootstrap supports balanced_accuracy or macro_f1')

def paired_study_bootstrap(first,second,metric='macro_f1',reps=10000,seed=42):
    """Prediction CSVs with y,study_id,p_0..p_C and identical patients.
    Aggregate once per study; bootstrap studies rather than correlated frames.
    """
    a=pd.read_csv(first);b=pd.read_csv(second)
    a_cols=sorted([k for k in a.columns if k.startswith('p_')]);b_cols=sorted([k for k in b.columns if k.startswith('p_')])
    if a_cols!=b_cols or not a_cols:raise ValueError('Probability columns disagree')
    a=a.groupby('study_id')[["y"]+a_cols].mean();b=b.groupby('study_id')[["y"]+b_cols].mean()
    common=sorted(set(a.index)&set(b.index))
    if len(common)<5:raise ValueError('Bootstrap needs at least 5 common test studies')
    aa=a.loc[common];bb=b.loc[common]
    if not np.array_equal(aa.y.to_numpy(),bb.y.to_numpy()):raise ValueError('Ground-truth mismatches between predictions')
    y=aa.y.to_numpy().astype(int);pa=aa[a_cols].to_numpy();pb=bb[b_cols].to_numpy()
    observed=float(_metric(y,pb,metric)-_metric(y,pa,metric))
    rng=np.random.default_rng(seed);d=[];n=len(y)
    for _ in range(reps):
        s=rng.integers(0,n,n)
        d.append(_metric(y[s],pb[s],metric)-_metric(y[s],pa[s],metric))
    lo,hi=np.quantile(d,[.025,.975]);p_two=min(1.,2*min(np.mean(np.asarray(d)<=0),np.mean(np.asarray(d)>=0)))
    return {'study_count':n,'metric':metric,'difference_b_minus_a':observed,
            'bootstrap_95_ci':[float(lo),float(hi)],'bootstrap_two_sided_p':float(p_two),
            'resamples':int(reps)}
