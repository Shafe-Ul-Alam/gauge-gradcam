import sys, json, numpy as np, torch, torch.nn as nn, torchvision.models as tvm
sys.path.insert(0,'src_imaging'); import camera_ready_experiments as E
torch.set_num_threads(1)
U='/mnt/user-data/uploads/fedhealth_q1/results_imaging/model_cache_scaleup/'
CK={'collapsed (non-IID Adam)':'resnet18_noniid_adam_plain_pre1_s6.pt',
    'gauge-normalised':'resnet18_noniid_adam_gauge_pre1_s6.pt',
    'healthy (IID)':'resnet18_iid_adam_plain_pre1_s0.pt'}
X,y,cl=E.load_dedup(None); Xtr,Xte,ytr,yte=E.split(X,y,123); Xp=Xte[:24]
STEPS=32
def load(f):
    m=tvm.resnet18(weights=None); m.fc=nn.Linear(512,4); m.load_state_dict(torch.load(U+f,map_location='cpu')); return m.eval()
def ig(fn, x):
    al=torch.linspace(1/STEPS,1,STEPS).view(-1,1,1,1); xs=(al*x).requires_grad_(True)
    out=fn(xs).sum(); g=torch.autograd.grad(out,xs)[0]
    return (x*g.mean(0,keepdim=True))[0].sum(0)      # sum over channels -> HxW (baseline 0)
res={}
for name,f in CK.items():
    m=load(f); W=m.fc.weight.detach(); wb=W.mean(0); R=W-wb
    feat=nn.Sequential(*list(m.children())[:-1], nn.Flatten())
    off_share=[]; cls_corr=[]; corr_off_full=[]
    for i in range(len(Xp)):
        x=Xp[i:i+1]
        with torch.no_grad(): t=int(m(x).argmax())
        A_off=ig(lambda z: feat(z)@wb, x)
        A_res={k: ig(lambda z,k=k: feat(z)@R[k], x) for k in range(4)}
        full=A_off+A_res[t]
        off_share.append(float(A_off.abs().sum()/(A_off.abs().sum()+A_res[t].abs().sum())))
        # class sensitivity of LOGIT-IG: correlation between maps for different target classes
        maps=[(A_off+A_res[k]).flatten() for k in range(4)]
        cs=[float(np.corrcoef(maps[a],maps[b])[0,1]) for a in range(4) for b in range(a+1,4)]
        cls_corr.append(np.mean(cs))
        corr_off_full.append(float(np.corrcoef(A_off.flatten(),full.flatten())[0,1]))
    rho,_=E.rho_of(m) if False else (float(wb.norm()/(R.norm()/2)),0)
    res[name]=dict(rho=rho, offset_share_of_IG_mass=float(np.mean(off_share)),
                   mean_corr_between_class_maps=float(np.mean(cls_corr)),
                   corr_offset_vs_full_map=float(np.mean(corr_off_full)))
    print(name, {k:round(v,3) for k,v in res[name].items()}, flush=True)
json.dump(res,open('ig_check.json','w'),indent=1)
