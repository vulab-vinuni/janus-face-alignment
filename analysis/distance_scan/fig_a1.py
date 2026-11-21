import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":1.0,"xtick.direction":"in","ytick.direction":"in",
  "xtick.top":True,"ytick.right":True,"xtick.major.size":4,"ytick.major.size":4,"legend.frameon":False})
F=pd.read_csv("data/distance_scan/distance_scan_results.csv"); D=pd.read_csv("data/heterostructures/het599_training_set.csv").set_index("id")
sty={"H20992":("Te-Te (XX)","#1f5fbf","o"),"H21356":("Te-Te' (XX)","#1f5fbf","s"),"H03396":("S-S (XX)","#6fa8dc","D"),
     "H14018":("P-Se (XQ)","#e08e2b","^"),"H21905":("P-Te (XQ)","#b35806","v"),"H03523":("P-Sb (QQ)","#2a8c6e","o"),"H15111":("As-Sb (QQ)","#7fc8a9","s")}
fig,ax=plt.subplots(1,2,figsize=(7.2,3.1),gridspec_kw=dict(wspace=.32))
for i,(nm,c,m) in sty.items():
    g=F[F.id==i].sort_values("d"); h=D.loc[i]; sep=abs(h.dE)>(h.Eg1+h.Eg2)/2
    ok=g.Eg_bi>0.05
    for a,x in ((ax[0],g.d),(ax[1],g.w_min)):
        a.plot(x,g.S,"-",color=c,lw=1.2,alpha=.8,dashes=(4,2) if sep else ())
        a.scatter(x[ok],g.S[ok],marker=m,s=26,facecolor=c,edgecolor="k",lw=.5,zorder=3,label=nm if a is ax[0] else None)
        a.scatter(x[~ok],g.S[~ok],marker=m,s=26,facecolor="w",edgecolor=c,lw=1,zorder=3)
        e=g[g.delta==0]; a.scatter(e.d if a is ax[0] else e.w_min,e.S,marker=m,s=90,facecolor="none",edgecolor="k",lw=1.2,zorder=4)
ax[0].axvspan(2.2,2.6,color="#2a8c6e",alpha=.08,lw=0); ax[0].text(2.4,1.0,"QQ equil.",ha="center",va="top",fontsize=7,color="#2a8c6e")
ax[0].set(xlabel="interlayer plane separation $d$ (Å)",ylabel="screening factor $S=1-eV_h^{(B)}/\\Delta E_m$",xlim=(1.5,4.8),ylim=(.25,1.02))
ax[1].set(xlabel="band-edge layer weight $w_\\mathrm{min}$",xlim=(0,1),ylim=(.25,1.02))
ax[0].legend(loc="lower right",fontsize=6.8,ncol=2,handletextpad=.3,columnspacing=.8)
for a,l in zip(ax,"ab"): a.text(-.16,1.02,l,transform=a.transAxes,fontweight="bold",fontsize=11)
fig.savefig("figures/figS_distance_scan.pdf",bbox_inches="tight"); pass
