import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from glr import r2
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.linewidth":1,"xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,"mathtext.default":"regular"})
T=pd.read_csv("train_oof.csv"); loc=(T.p_local>=.3).values
def eg(eVh):
    v1=-T.Eg1/2; c1=T.Eg1/2; v2=T.dE-eVh-T.Eg2/2; return (np.minimum(c1,v2+T.Eg2)-np.maximum(v1,v2)).values
M=[("Anderson","oof_and",None),("midgap",None,T.dE.values),("sLR, global S","oof_slr",None),("gLR, S(contact)","oof_glrc",None)]
TC={"I":"#7B2FBE","II":"#1A9E77","III":"#4FA8E0"}; CC={"XX":"#2F5DA8","QX":"#2E9E62","QQ":"#C0392B"}
fig,A=plt.subplots(2,4,figsize=(7.3,3.9),gridspec_kw=dict(wspace=.38,hspace=.42))
for k,(nm,col,val) in enumerate(M):
    p=T[col].values if col else val; ax=A[0,k]
    for c in ("XX","QX","QQ"): m=(T.contact==c).values; ax.scatter(T.eVh_B[m],p[m],s=4,c=CC[c],alpha=.6,lw=0,label={"QX":"XQ"}.get(c,c))
    lim=(-1,1.6) if nm=="midgap" else (-.9,.9); ax.plot(lim,lim,c="k",lw=.6); ax.set(xlim=lim,ylim=lim,title=nm)
    ax.text(.05,.95,f"$r^2$ = {r2(T.eVh_B,p)+0:.2f}".replace("-0.00","0.00"),transform=ax.transAxes,va="top",fontsize=7)
    ax.set_xlabel(r"$eV_h$ DFT (eV)",fontsize=7.5)
    e=eg(p); ax=A[1,k]
    for t in ("I","II","III"):
        m=(T.type_dft==t).values; ax.scatter(T.Eg_bi[m&loc],e[m&loc],s=6,c=TC[t],lw=0,alpha=.85,label=f"type {t}"); ax.scatter(T.Eg_bi[m&~loc],e[m&~loc],s=5,facecolors="none",edgecolors=TC[t],lw=.4,alpha=.5)
    l2=(-1.2,2.0); ax.plot(l2,l2,c="k",lw=.6); ax.set(xlim=l2,ylim=l2); ax.set_xlabel(r"$E_g^\mathrm{het}$ DFT (eV)",fontsize=7.5)
    ax.text(.05,.95,f"$r^2$ = {r2(T.Eg_bi,e):.2f}\nlocalized: {r2(T.Eg_bi[loc],e[loc]):.2f}",transform=ax.transAxes,va="top",fontsize=6.5)
A[0,0].set_ylabel(r"$eV_h$ model (eV)"); A[1,0].set_ylabel(r"$E_g^\mathrm{het}$ model (eV)")
A[0,3].legend(loc="lower right",fontsize=6,markerscale=2,handletextpad=.1); A[1,3].legend(loc="lower right",fontsize=6,markerscale=1.5,handletextpad=.1)
for i,a in enumerate(A.ravel()): a.text(-.32,1.08,f"({'abcdefgh'[i]})",transform=a.transAxes,fontweight="bold",fontsize=8.5)
fig.savefig("figS_models.pdf",bbox_inches="tight"); fig.savefig("figS_models.png",bbox_inches="tight",dpi=160)
