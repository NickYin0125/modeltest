
# -*- coding: utf-8 -*-
"""
rf_fusion_minimal.py

What it does:
- Pick 4 clinical features (prefers: Internal echo, Boundary, Blood flow signal, Maximum diameter; else top-4 by univariate p)
- Train two RF models:
  * Clinical-only (4 features)
  * Fusion (Rad_score + 4 clinical)
- Use StratifiedKFold OOF for train metrics; fit on full train for holdout test metrics
- Export: predictions (xlsx), combined ROC (train/test), DCA (train/test), DeLong(Fusion vs Clinical), Fusion RF importances, metrics JSON

Usage:
python rf_fusion_minimal.py --clinical_excel 临床.xlsx --sheng_pred_xlsx prediction_results_sheng.xlsx --output_dir outputs_min
"""
import argparse, json, re, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact, norm
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

def find_col(df,cands,default=None):
    cols=list(df.columns); low={str(c).lower():c for c in cols}
    for c in cands:
        if c in cols: return c
    for c in cands:
        if str(c).lower() in low: return low[str(c).lower()]
    for c in cands:
        for col in cols:
            if str(c).lower() in str(col).lower(): return col
    return default

def to_bin(s):
    s=s.copy(); u=sorted(pd.unique(s.dropna()))
    if len(u)==2: return s.map({u[0]:0,u[1]:1}).astype(int)
    m={'benign':0,'icr':0,'0':0,0:0,'malignant':1,'ecr':1,'1':1,1:1}
    return s.map(lambda x:m.get(str(x).lower(),np.nan)).astype(float).astype('Int64')

def dtypes_split(X):
    num=[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat=[c for c in X.columns if c not in num]; return num,cat

def build_rf(num,cat):
    pre=ColumnTransformer([("num",SimpleImputer(strategy="median"),num),
                           ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                                            ("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat)])
    rf=RandomForestClassifier(n_estimators=800,random_state=42,n_jobs=-1)
    return Pipeline([("pre",pre),("rf",rf)])

def oof_and_test(pipe,Xtr,ytr,Xte,cv=5,seed=42):
    skf=StratifiedKFold(n_splits=cv,shuffle=True,random_state=seed)
    oof=np.zeros(len(Xtr))
    for tr,va in skf.split(Xtr,ytr):
        p=Pipeline(steps=pipe.steps); p.fit(Xtr.iloc[tr],ytr[tr])
        oof[va]=p.predict_proba(Xtr.iloc[va])[:,1]
    pipe.fit(Xtr,ytr); te=pipe.predict_proba(Xte)[:,1]
    return oof,te,pipe

def metrics(y,prob,thr=0.5):
    auc=roc_auc_score(y,prob); pred=(prob>=thr).astype(int)
    acc=accuracy_score(y,pred)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sens=tp/(tp+fn) if tp+fn else np.nan; spec=tn/(tn+fp) if tn+fp else np.nan
    return dict(Accuracy=acc,Sensitivity=sens,Specificity=spec,AUC=auc,
                Precision=precision_score(y,pred,zero_division=0),
                Recall=recall_score(y,pred,zero_division=0),
                F1=f1_score(y,pred,zero_division=0))

def boot_ci(y,prob,n_boot=2000,seed=42,alpha=0.05):
    rng=np.random.default_rng(seed); n=len(y); idx=np.arange(n); aucs=[]
    for _ in range(n_boot):
        bs=rng.choice(idx,size=n,replace=True); yb=y[bs]; pb=prob[bs]
        if len(np.unique(yb))<2: continue
        try: aucs.append(roc_auc_score(yb,pb))
        except: pass
    if not aucs: return (np.nan,np.nan)
    return (float(np.percentile(aucs,100*alpha/2)), float(np.percentile(aucs,100*(1-alpha/2))))

def plot_roc(y1,p1,l1,y2,p2,l2,title,outpng):
    f1,t1,_=roc_curve(y1,p1); a1=roc_auc_score(y1,p1)
    f2,t2,_=roc_curve(y2,p2); a2=roc_auc_score(y2,p2)
    plt.figure(figsize=(5,4)); plt.plot(f1,t1,label=f"{l1} (AUC={a1:.3f})")
    plt.plot(f2,t2,label=f"{l2} (AUC={a2:.3f})"); plt.plot([0,1],[0,1],'--')
    plt.xlabel("1 - Specificity (FPR)"); plt.ylabel("Sensitivity (TPR)")
    plt.title(title); plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(outpng,dpi=200); plt.close()

def dca_curve(y,prob,thr):
    y=np.asarray(y).astype(int); p=np.asarray(prob).astype(float); N=len(y)
    prev=y.mean(); model=[]; allnb=[]; nonenb=[]
    for t in thr:
        pred=(p>=t).astype(int)
        TP=np.sum((pred==1)&(y==1)); FP=np.sum((pred==1)&(y==0))
        nb=(TP/N) - (FP/N)*(t/(1-t))
        model.append(nb); allnb.append(prev - (1-prev)*(t/(1-t))); nonenb.append(0.0)
    return np.array(model), np.array(allnb), np.array(nonenb)

def plot_dca(thr,curves,title,outpng):
    plt.figure(figsize=(6,4))
    for name,y in curves.items(): plt.plot(thr,y,label=name)
    plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
    plt.title(title); plt.legend(loc="best"); plt.tight_layout(); plt.savefig(outpng,dpi=200); plt.close()


def compute_midrank(x):
    x = np.asarray(x)
    J = len(x)
    order = np.argsort(x)
    x_sorted = x[order]
    t = np.zeros(J)
    i = 0
    while i < J:
        j = i
        while j < J and x_sorted[j] == x_sorted[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(J)
    out[order] = t
    return out

def fastDeLong(preds_sorted_transposed, m):
    n_classifiers, n_examples = preds_sorted_transposed.shape
    n = n_examples - m
    tx = np.zeros((n_classifiers, m))
    ty = np.zeros((n_classifiers, n))
    tz = np.zeros((n_classifiers, m + n))
    for r in range(n_classifiers):
        preds = preds_sorted_transposed[r]
        tx[r] = compute_midrank(preds[:m])
        ty[r] = compute_midrank(preds[m:])
        tz[r] = compute_midrank(preds)
    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    return aucs, s

def delong_roc_test(y_true, p1, p2):
    y = np.asarray(y_true).astype(int)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    pos_idx = y == 1
    neg_idx = y == 0
    m = int(pos_idx.sum())
    if m == 0 or m == len(y):
        return dict(auc1=np.nan, auc2=np.nan, delta=np.nan, se=np.nan, z=np.nan, p=np.nan)
    preds = np.vstack([
        np.concatenate([p1[pos_idx], p1[neg_idx]]),
        np.concatenate([p2[pos_idx], p2[neg_idx]])
    ])
    aucs, cov = fastDeLong(preds, m)
    diff = aucs[0] - aucs[1]
    var = cov[0,0] + cov[1,1] - 2*cov[0,1]
    se = np.sqrt(var) if np.isfinite(var) and var>0 else np.nan
    z  = diff / se if (se and np.isfinite(se) and se!=0) else np.inf
    p  = 2*(1-norm.cdf(abs(z))) if np.isfinite(z) else 0.0
    return dict(auc1=float(aucs[0]), auc2=float(aucs[1]), delta=float(diff),
                se=float(se), z=float(z), p=float(p))

def bootstrap_delta_auc_ci(y_true, prob1, prob2, n_boot=2000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(prob1, dtype=float)
    p2 = np.asarray(prob2, dtype=float)
    n = len(y_true)
    idx = np.arange(n)
    diffs = []
    for _ in range(n_boot):
        bs = rng.choice(idx, size=n, replace=True)
        yb = y_true[bs]
        if len(np.unique(yb)) < 2:
            continue
        try:
            diffs.append(roc_auc_score(yb, p1[bs]) - roc_auc_score(yb, p2[bs]))
        except Exception:
            pass
    if not diffs:
        return (np.nan, np.nan)
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    return (lo, hi)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clinical_excel",required=True)
    ap.add_argument("--sheng_pred_xlsx",required=True)
    ap.add_argument("--output_dir",required=True)
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    clin=pd.read_excel(args.clinical_excel)
    idc=find_col(clin,["number","id","case_id","patient_id"])
    yl =find_col(clin,["label","结果","Result","group"])
    trc=find_col(clin,["train","split","is_train"])
    clin["id"]=clin[idc]; clin["y"]=to_bin(clin[yl]).astype(int); clin["train_flag"]=clin[trc].astype(int)

    xl=pd.ExcelFile(args.sheng_pred_xlsx); smap={s.lower():s for s in xl.sheet_names}
    df_tr=pd.read_excel(args.sheng_pred_xlsx, sheet_name=smap[[s for s in smap if "train" in s][0]])
    df_te=pd.read_excel(args.sheng_pred_xlsx, sheet_name=smap[[s for s in smap if "test" in s][0]])
    def unify(df):
        idc=find_col(df,["number","id","case_id","patient_id"])
        prob=[c for c in df.columns if re.search(r"prob|score|rad",str(c),re.I)]
        if not prob:
            num=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            prob=num[-1:] if num else []
        out=df[[idc,prob[0]]].copy(); out.columns=["id","Rad_score"]; return out
    rad=pd.concat([unify(df_tr),unify(df_te)],ignore_index=True)
    data=clin.merge(rad,on="id",how="left")

    # Select 4 clinical
    preferred=["Internal echo","Boundary","Blood flow signal","Maximum diameter"]
    clin_cols=[c for c in clin.columns if c not in [idc,yl,trc,"id","y","train_flag"]]
    use=[]
    for c in preferred:
        if c in data.columns: use.append(c)
    if len(use)<4:
        def uni(series,y):
            if pd.api.types.is_numeric_dtype(series):
                a=series[data["y"]==0].dropna(); b=series[data["y"]==1].dropna()
                if len(a)>2 and len(b)>2:
                    try: return mannwhitneyu(a,b,alternative="two-sided")[1]
                    except: return 1.0
                else: return 1.0
            tab=pd.crosstab(series.fillna("NA"), y)
            try:
                if tab.min().min()<5 and tab.shape==(2,2): return fisher_exact(tab)[1]
                else: return chi2_contingency(tab, correction=False)[1]
            except: return 1.0
        cand=[]
        for c in clin_cols:
            try: cand.append((c, uni(data[c], data["y"])))
            except: pass
        cand=sorted(cand,key=lambda x:x[1])
        for c,_ in cand:
            if c not in use: use.append(c)
            if len(use)==4: break

    feat_clin=use; feat_fuse=use+["Rad_score"]
    train=data[data["train_flag"]==1].copy(); test=data[data["train_flag"]!=1].copy()

    def run(feats,name):
        Xtr=train[feats]; Xte=test[feats]
        ytr=train["y"].values; yte=test["y"].values
        num,cat=dtypes_split(Xtr); pipe=build_rf(num,cat)
        oof,te,model=oof_and_test(pipe,Xtr,ytr,Xte)
        mtr=metrics(ytr,oof); mte=metrics(yte,te)
        ci_tr=boot_ci(ytr,oof); ci_te=boot_ci(yte,te)
        xlsx=out/f"prediction_results_{name}.xlsx"
        with pd.ExcelWriter(xlsx, engine="xlsxwriter") as w:
            pd.DataFrame({"id":train["id"],"label":ytr,"probability":oof}).to_excel(w,"Train",index=False)
            pd.DataFrame({"id":test["id"],"label":yte,"probability":te}).to_excel(w,"Test",index=False)
        return dict(name=name, feats=feats, oof=oof, te=te, ytr=ytr, yte=yte, m_tr=mtr, m_te=mte, ci_tr=ci_tr, ci_te=ci_te, xlsx=str(xlsx), model=model)

    res_clin=run(feat_clin,"clinical4_rf")
    res_fuse=run(feat_fuse,"fused_radscore_4clin_rf")

    # ROC
    def plot_roc(y1,p1,l1,y2,p2,l2,title,outpng):
        f1,t1,_=roc_curve(y1,p1); a1=roc_auc_score(y1,p1)
        f2,t2,_=roc_curve(y2,p2); a2=roc_auc_score(y2,p2)
        plt.figure(figsize=(5,4)); plt.plot(f1,t1,label=f"{l1} (AUC={a1:.3f})")
        plt.plot(f2,t2,label=f"{l2} (AUC={a2:.3f})"); plt.plot([0,1],[0,1],'--')
        plt.xlabel("1 - Specificity (FPR)"); plt.ylabel("Sensitivity (TPR)")
        plt.title(title); plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(outpng,dpi=200); plt.close()

    plot_roc(res_clin["ytr"],res_clin["oof"],"Clinical-4 (OOF)",res_fuse["ytr"],res_fuse["oof"],"Fusion (OOF)","Train ROC - Clinical vs Fusion",out/"train_auc_all.png")
    plot_roc(res_clin["yte"],res_clin["te"], "Clinical-4 (Test)", res_fuse["yte"],res_fuse["te"], "Fusion (Test)", "Test ROC - Clinical vs Fusion", out/"test_auc_all.png")

    # DCA
    def dca_curve(y,prob,thr):
        y=np.asarray(y).astype(int); p=np.asarray(prob).astype(float); N=len(y); prev=y.mean()
        model=[]; allnb=[]; nonenb=[]
        for t in thr:
            pred=(p>=t).astype(int)
            TP=np.sum((pred==1)&(y==1)); FP=np.sum((pred==1)&(y==0))
            nb=(TP/N) - (FP/N)*(t/(1-t))
            model.append(nb); allnb.append(prev - (1-prev)*(t/(1-t))); nonenb.append(0.0)
        return np.array(model), np.array(allnb), np.array(nonenb)
    def plot_dca(thr,curves,title,outpng):
        plt.figure(figsize=(6,4))
        for name,y in curves.items(): plt.plot(thr,y,label=name)
        plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
        plt.title(title); plt.legend(loc="best"); plt.tight_layout(); plt.savefig(outpng,dpi=200); plt.close()

    thr=np.arange(0.01,0.51,0.01)
    nb_c_tr, all_tr, none_tr = dca_curve(res_clin["ytr"], res_clin["oof"], thr)
    nb_f_tr, _, _            = dca_curve(res_fuse["ytr"], res_fuse["oof"], thr)
    plot_dca(thr, {"Clinical-4":nb_c_tr, "Fusion":nb_f_tr, "Treat All":all_tr, "Treat None":none_tr},
             "DCA (Train) - Clinical vs Fusion", out/"dca_train.png")
    nb_c_te, all_te, none_te = dca_curve(res_fuse["yte"], res_clin["te"], thr)  # use same y
    nb_f_te, _, _            = dca_curve(res_fuse["yte"], res_fuse["te"], thr)
    plot_dca(thr, {"Clinical-4":nb_c_te, "Fusion":nb_f_te, "Treat All":all_te, "Treat None":none_te},
             "DCA (Test) - Clinical vs Fusion", out/"dca_test.png")

    # Fusion RF feature importances
    rf=res_fuse["model"].named_steps["rf"]
    num=res_fuse["model"].named_steps["pre"].transformers_[0][2]
    cat=res_fuse["model"].named_steps["pre"].transformers_[1][2] if len(res_fuse["model"].named_steps["pre"].transformers_)>1 else []
    names=list(num) if num else []
    if cat:
        ohe=res_fuse["model"].named_steps["pre"].transformers_[1][1].named_steps["ohe"]
        names += list(ohe.get_feature_names_out(cat))
    imp=rf.feature_importances_
    imp_df=pd.DataFrame({"feature":names,"gini_importance":imp}).sort_values("gini_importance",ascending=False)
    imp_df.to_excel(out/"fusion_rf_importance.xlsx",index=False)
    top=min(20,len(imp_df))
    plt.figure(figsize=(8,max(3,0.35*top))); plt.barh(range(top), imp_df["gini_importance"].values[:top][::-1])
    plt.yticks(range(top), imp_df["feature"].values[:top][::-1]); plt.xlabel("Gini importance")
    plt.title("Fusion RF Feature Importances"); plt.tight_layout(); plt.savefig(out/"fusion_rf_importance.png",dpi=200); plt.close()

    # DeLong fusion vs clinical
    def delong(y, s1, s2):
        def _mid(x):
            J=len(x); order=np.argsort(x); xs=x[order]; uniq,cnts=np.unique(xs,return_counts=True)
            mid=np.zeros(J); start=0
            for c in cnts: end=start+c; mid[start:end]=0.5*(start+end-1)+1; start=end
            out=np.empty(J); out[order]=mid; return out
        def _fast(y,score):
            pos=score[y==1]; neg=score[y==0]; m=len(pos); n=len(neg)
            if m==0 or n==0: return np.nan,np.nan
            r=_mid(np.concatenate([pos,neg])); rpos=r[:m]; rneg=r[m:]
            auc=(rpos.sum()/m - (m+1)/2.0)/n
            v01=(rpos-(m+1)/2.0)/n; v10=1.0 - (rneg-(n+1)/2.0)/m
            sx=np.var(v01,ddof=1)/m; sy=np.var(v10,ddof=1)/n; return auc, sx+sy
        a1,v1=_fast(y,s1); a2,v2=_fast(y,s2); delta=a1-a2
        se=np.sqrt(v1+v2) if np.isfinite(v1) and np.isfinite(v2) else np.nan
        z=delta/se if (se and np.isfinite(se) and se!=0) else np.inf
        p=2*(1-norm.cdf(abs(z))) if np.isfinite(z) else 0.0
        return dict(auc1=float(a1),auc2=float(a2),delta=float(delta),se=float(se),z=float(z),p=float(p))
    dl_tr=delong_roc_test(res_fuse["ytr"], res_fuse["oof"], res_clin["oof"])
    dl_te=delong_roc_test(res_fuse["yte"], res_fuse["te"], res_clin["te"])
    df_dl = pd.DataFrame([dict(split="train",**dl_tr), dict(split="test",**dl_te)])
    try:
        ci_lo_tr, ci_hi_tr = bootstrap_delta_auc_ci(res_fuse["ytr"], res_fuse["oof"], res_clin["oof"])
    except Exception:
        ci_lo_tr, ci_hi_tr = (np.nan, np.nan)
    try:
        ci_lo_te, ci_hi_te = bootstrap_delta_auc_ci(res_fuse["yte"], res_fuse["te"], res_clin["te"])
    except Exception:
        ci_lo_te, ci_hi_te = (np.nan, np.nan)
    df_dl.loc[df_dl["split"]=="train", ["delta_auc_ci_low","delta_auc_ci_high"]] = [ci_lo_tr, ci_hi_tr]
    df_dl.loc[df_dl["split"]=="test",  ["delta_auc_ci_low","delta_auc_ci_high"]] = [ci_lo_te, ci_hi_te]
    df_dl.to_excel(out/"delong_fusion_vs_clinical.xlsx", index=False)

    # Save metrics JSON
    Path(out/"metrics_fusion.json").write_text(json.dumps({
        "model_C_fusion": {
            "train": res_fuse["m_tr"], "test": res_fuse["m_te"],
            "train_ci": list(boot_ci(res_fuse['ytr'], res_fuse['oof'])),
            "test_ci": list(boot_ci(res_fuse['yte'], res_fuse['te']))
        },
        "clinical_model": {"train": res_clin["m_tr"], "test": res_clin["m_te"]},
        "features": {"clinical_4": feat_clin, "fusion": feat_fuse}
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done. Outputs @", out)

if __name__=="__main__":
    main()
