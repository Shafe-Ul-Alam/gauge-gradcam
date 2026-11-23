import torch, numpy as np, torch.nn.functional as F, json
torch.set_num_threads(1)
d=torch.load('results_imaging/feature_cache/resnet18_gap_seed0.pt'); ftr,ytr=d['ftr'],d['ytr']
D=ftr.shape[1]; eta=1e-3; S=5; C=4; R=30
print("active feature fraction:", float((ftr>0).float().mean()))
out={}
for K in [2,3,4]:
  res=[]
  for seed in [0,1,2]:
    torch.manual_seed(seed); y=np.asarray(ytr); y=np.where(y<=K-2,y,K-1) if K<4 else y
    rng=np.random.default_rng(seed); by={c:list(rng.permutation(np.where(y==c)[0])) for c in range(K)}
    holders={c:[i for i in range(C) if i%K==c] for c in range(K)}; parts=[[] for _ in range(C)]
    for c,ix in by.items():
        for hh,ch in zip(holders[c],np.array_split(np.array(ix),len(holders[c]))): parts[hh]+=ch.tolist()
    parts=[torch.tensor(sorted(p)) for p in parts if p]; yt=torch.tensor(y)
    g=torch.zeros(K,D); torch.nn.init.normal_(g,std=0.01)
    norms=[]; own=[]; oth=[]; sumrow=[]
    for t in range(R):
        st=[]
        for ci,idx in enumerate(parts):
            w=g.clone().requires_grad_(True); opt=torch.optim.Adam([w],lr=eta); kdom=int(np.bincount(y[idx.numpy()]).argmax())
            for s in range(S):
                w0=w.detach().clone(); loss=F.cross_entropy(ftr[idx]@w.T,yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
                dW=(w.detach()-w0)
                own.append(float(dW[kdom].mean()/eta)); oth.append(float(torch.cat([dW[j:j+1] for j in range(K) if j!=kdom]).mean()/eta))
                sumrow.append(float(dW.sum(0).mean()/eta))
            st.append(w.detach().clone())
        g=torch.stack(st).mean(0); norms.append(float(g.mean(0).norm()))
    h=R//2; slope=(norms[-1]-norms[h])/(R-h)
    pred=S*(K-2)*eta*np.sqrt(D)/K
    res.append(dict(slope=slope, pred_eq5=pred, own=np.mean(own), other=np.mean(oth), rowsum=np.mean(sumrow)))
  m={k:float(np.mean([r[k] for r in res])) for k in res[0]}
  m['ratio_measured_over_pred']= m['slope']/m['pred_eq5'] if m['pred_eq5'] else None
  m['realized_pred']= abs(m['rowsum'])*eta*S*np.sqrt(D)/K
  out[f"K={K}"]=m; print(K, {k:(round(v,5) if v is not None else None) for k,v in m.items()})
json.dump(out,open('drift_check.json','w'),indent=1)
