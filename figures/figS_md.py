import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ase.io import read; from ase.units import kB
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.linewidth":1,"xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,"xtick.major.width":1,"ytick.major.width":1})
D=pd.read_csv("md300K/summary.csv")
sel=[("SeInPbP","#2B5BA8"),("SGaSnSb","#D9661F"),("SInSnSb","#14946A")]
fig,ax=plt.subplots(1,3,figsize=(7.2,2.3),gridspec_kw=dict(wspace=.42,left=.07,right=.985,bottom=.2,top=.86))
for n,c in sel:
    tr=read(f"md300K/traj/{n}.extxyz",":"); t=np.arange(len(tr))*0.05
    E=np.array([a.get_potential_energy() for a in tr])/len(tr[0]); E=(E-E[len(E)//5:].mean())*1000
    T=np.array([a.get_kinetic_energy()/(1.5*len(a)*kB) for a in tr])
    ax[0].plot(t,E,c=c,lw=.9,label=n); ax[1].plot(t,T,c=c,lw=.9)
ax[0].set(xlabel="Time (ps)",ylabel="$E_\\mathrm{pot}-\\langle E_\\mathrm{pot}\\rangle$ (meV/atom)",xlim=(0,5)); ax[1].set(xlabel="Time (ps)",ylabel="Temperature (K)",xlim=(0,5)); ax[1].axhline(300,c="k",ls=":",lw=.8)
ax[0].legend(frameon=False,fontsize=7,loc="lower right",handlelength=1.2)
ax[2].hist(D.dz_max_A,bins=np.arange(.2,.62,.025),color="#7E3BA8",edgecolor="k",lw=.5)
ax[2].set(xlabel="Max. out-of-plane displacement (Å)",ylabel="Number of compounds",ylim=(0,23))
ax[2].text(.04,.93,f"108/108: no broken bonds\n$\\langle T\\rangle$ = {D.T_mean.min():.0f}–{D.T_mean.max():.0f} K",transform=ax[2].transAxes,va="top",fontsize=6.8)
for i,a in enumerate(ax): a.text(-.22,1.06,"(%s)"%"abc"[i],transform=a.transAxes,fontweight="bold",fontsize=9.5)
fig.savefig("figS_md.pdf"); fig.savefig("figS_md.png",dpi=160)
