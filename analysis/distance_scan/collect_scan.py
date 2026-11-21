import json, os, numpy as np, pandas as pd
D=pd.read_csv("data/heterostructures/het599_training_set.csv").set_index("id")
lab={"H20992":"Te-Te (XX)","H21356":"Te-Te' (XX)","H03396":"S-S (XX)","H03523":"P-Sb (QQ)","H15111":"As-Sb (QQ)","H14018":"P-Se (QX)","H21905":"P-Te (QX)"}
rows=[]
for csv,suf in [("data/distance_scan/a1_points.csv",""),("data/distance_scan/a1b_points.csv","c60")]:
    P=pd.read_csv(csv)
    for _,p in P.iterrows():
        f=f"a1/qe/{p.dir}/post/bilayer.json"
        if not os.path.exists(f): continue
        j=json.load(open(f)); h=D.loc[p.id]
        eB=h.eVh_B+h.dPhi_bi-j["dPhi_bi"]
        rows.append(dict(id=p.id,grp=lab[p.id],stack=p.stack,cell=suf or "c45",delta=p.delta,d=p.d_plane,dmin=p.dmin_atom,dE=h.dE,eVhB=eB,S=1-eB/h.dE,
             SA=1-(h.dE-(j["Em2p"]-j["Em1p"]))/h.dE,w_min=j["w_min"],Eg_bi=j["Eg_bi"],flat=max(j["flat_bot"],j["flat_top"]),dPhi=j["dPhi_bi"],dPhi_prod=h.dPhi_bi,Etot=j["Etot"],a=j["a"]))
R=pd.DataFrame(rows)
# consistency: c45 vs c60 at delta 0
c=R[R.delta==0].pivot_table(index="grp",columns="cell",values=["dPhi","flat"]).round(4); print(c)
# final set: c60 for delta>=0 and all H14018 (AAr); c45 for delta<0 (except H14018 BAr)
F=R[((R.cell=="c60")) | ((R.cell=="c45")&(R.delta<0)&(R.id!="H14018"))].copy()
print("max flat in final set:",F.flat.max().round(4)); print(F.pivot_table(index="grp",columns="delta",values="flat").round(3).to_string())
print(F.pivot_table(index="grp",columns="delta",values="S").round(3).to_string())
print(F.pivot_table(index="grp",columns="delta",values="d").round(2).to_string())
print(F.pivot_table(index="grp",columns="delta",values="w_min").round(2).to_string())
print(F.pivot_table(index="grp",columns="delta",values="Eg_bi").round(2).to_string())
F.to_csv("a1_final.csv",index=False)
