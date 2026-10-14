import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":9,
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8,"pdf.fonttype":42})
DS=["PaySim\n(Fraud)","SAML-D\n(AML)","Botswana\n(Churn)"]
COL=["#1F4E79","#C0504D","#4F8F5B","#E0A030","#7B5EA7"]

# ---------- Fig 1: pipeline ----------
W,H=1500,600
fig=plt.figure(figsize=(7.2,2.9)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis("off")
C={"blue":("#DBEAFE","#2563EB"),"orange":("#FEF3C7","#D97706"),"green":("#D1FAE5","#059669"),"purple":("#EDE9FE","#7C3AED")}
def box(x,y,w,h,c,t,s=None,fill=None,edge=None,lw=1.4,fs=7.2):
    f,e=C[c] if c else (fill,edge)
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=14",fc=f,ec=e,lw=lw,zorder=2))
    cx,cy=x+w/2,y+h/2
    if s:
        ax.text(cx,cy-13,t,ha="center",va="center",fontsize=fs,fontweight="bold",zorder=3)
        ax.text(cx,cy+15,s,ha="center",va="center",fontsize=fs-0.8,color="#475569",zorder=3)
    else: ax.text(cx,cy,t,ha="center",va="center",fontsize=fs,fontweight="bold",zorder=3)
def arr(p,q): ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=9,lw=1.1,color="#475569",shrinkA=0,shrinkB=0,zorder=1))
for x,(k,l) in zip([32,262,492,722],[("blue","Data Preparation"),("orange","Validation Setup"),("green","Modeling"),("purple","Evaluation")]):
    f,e=C[k]; ax.add_patch(FancyBboxPatch((x,15),25,25,boxstyle="round,pad=0,rounding_size=6",fc=f,ec=e,lw=1.2))
    ax.text(x+35,28,l,va="center",fontsize=7.2,color="#475569")
w,h=190,85; xs=[32,275,517,759,1002,1244]
for x,(c,t,s) in zip(xs,[("blue","Raw Datasets",None),("blue","Data","Understanding"),("blue","Data Cleaning",None),("blue","Feature","Engineering"),("orange","70/30 Train-Test","Split"),("orange","Preprocessing",None)]): box(x,120,w,h,c,t,s)
for i in range(5): arr((xs[i]+w+3,162),(xs[i+1]-3,162))
ax.plot([1339,1339],[205,288],color="#475569",lw=1.1,zorder=1); ax.plot([1339,127],[288,288],color="#475569",lw=1.1,zorder=1); arr((127,288),(127,410))
box(32,412,w,h,"orange","5-Fold","Cross Validation"); box(275,412,w,h,"orange","SMOTE")
ax.add_patch(FancyBboxPatch((517,350),190,209,boxstyle="round,pad=0,rounding_size=20",fc="#D1FAE5",ec="#059669",lw=1.2,ls=(0,(4,3)),zorder=1))
for y,t in zip([362,426,491],["Decision Tree","Random Forest","XGBoost"]): box(533,y,158,55,None,t,fill="white",edge="#059669",fs=6.6)
box(759,412,w,h,"green","Model Selection","PR-AUC → F1"); box(1002,412,w,h,"purple","Final Untouched","Test"); box(1244,412,w,h,"purple","Cross-Domain","Comparison")
for a,b in [(222,275),(465,517),(707,759),(949,1002),(1192,1244)]: arr((a+3,454),(b-3,454))
fig.savefig("figures/fig_pipeline.pdf"); plt.close(fig)

def clean(ax,ylab=None):
    ax.grid(axis="y",lw=0.4,alpha=0.5); ax.set_axisbelow(True)
    if ylab: ax.set_ylabel(ylab)

# ---------- Fig: confusion matrices ----------
cms=[np.array([[1904215,1],[1,4569]]),np.array([[2847341,1006],[227,2882]]),np.array([[30334,130],[65,4163]])]
titles=["PaySim (Random Forest)","SAML-D (XGBoost)","Botswana Churn (XGBoost)"]
labs=[["Legit","Fraud"],["Normal","Laundering"],["Retained","Churned"]]
fig,axs=plt.subplots(1,3,figsize=(7.2,2.35))
for a,cm,t,l in zip(axs,cms,titles,labs):
    norm=cm/cm.sum(axis=1,keepdims=True)
    a.imshow(norm,cmap="Blues",vmin=0,vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j,i,f"{cm[i,j]:,}\n({norm[i,j]*100:.2f}%)",ha="center",va="center",fontsize=7.3,color="white" if norm[i,j]>0.5 else "black")
    a.set_xticks([0,1]); a.set_yticks([0,1]); a.set_xticklabels(["Pred. "+l[0],"Pred. "+l[1]],fontsize=7); a.set_yticklabels(["True "+l[0],"True "+l[1]],fontsize=7,rotation=90,va="center")
    a.set_title(t,fontsize=8.2); 
    for s in a.spines.values(): s.set_visible(False)
    a.tick_params(length=0)
fig.tight_layout(); fig.savefig("figures/fig_confusion.pdf"); plt.close(fig)

# ---------- Fig: test precision/recall/F1 ----------
P=[99.9781,74.1255,96.9718]; R=[99.9781,92.6986,98.4626]; F=[99.9781,82.3782,97.7115]
fig,ax=plt.subplots(figsize=(3.45,2.5)); x=np.arange(3); bw=0.26
for k,(v,l,c) in enumerate(zip([P,R,F],["Precision","Recall","F1-score"],COL)):
    b=ax.bar(x+(k-1)*bw,v,bw,label=l,color=c)
    for xi,vi in zip(x+(k-1)*bw,v): ax.text(xi,vi+1,f"{vi:.1f}",ha="center",fontsize=5.6,rotation=90,va="bottom")
ax.set_xticks(x); ax.set_xticklabels(DS,fontsize=7.5); ax.set_ylim(0,125); ax.set_yticks([0,25,50,75,100]); clean(ax,"Score (%)")
ax.legend(ncol=3,fontsize=6.8,frameon=False,loc="upper center",bbox_to_anchor=(0.5,1.02),handlelength=1,columnspacing=1)
fig.tight_layout(); fig.savefig("figures/fig_test_prf.pdf"); plt.close(fig)

# ---------- Fig: CV vs Test (PR-AUC and F1) ----------
cvpr=[99.4802,82.6673,99.8187]; tepr=[99.9999,93.5894,99.8634]
cvf=[98.6224,84.0384,97.5142]; tef=[99.9781,82.3782,97.7115]
fig,axs=plt.subplots(1,2,figsize=(7.2,2.45))
for a,(cv,te,t) in zip(axs,[(cvpr,tepr,"PR-AUC"),(cvf,tef,"F1-score")]):
    x=np.arange(3); bw=0.36
    a.bar(x-bw/2,cv,bw,label="5-fold CV",color=COL[0]); a.bar(x+bw/2,te,bw,label="Held-out test",color=COL[1])
    for xi,vi in zip(x-bw/2,cv): a.text(xi,vi+1,f"{vi:.1f}",ha="center",fontsize=6.8)
    for xi,vi in zip(x+bw/2,te): a.text(xi,vi+1,f"{vi:.1f}",ha="center",fontsize=6.8)
    a.set_xticks(x); a.set_xticklabels(DS,fontsize=7.5); a.set_ylim(0,120); a.set_yticks([0,25,50,75,100]); clean(a,t+" (%)")
axs[0].legend(fontsize=7,frameon=False,loc="upper center",ncol=2,bbox_to_anchor=(0.5,1.02))
fig.tight_layout(); fig.savefig("figures/fig_cv_vs_test.pdf"); plt.close(fig)

# ---------- Fig: CV PR-AUC by model ----------
cv={"Decision Tree":[98.9651,72.0481,94.2779],"Random Forest":[99.4802,82.1207,92.8263],"XGBoost":[99.4271,82.6673,99.8187]}
fig,ax=plt.subplots(figsize=(3.45,2.5)); x=np.arange(3); bw=0.26
for k,(m,v) in enumerate(cv.items()):
    ax.bar(x+(k-1)*bw,v,bw,label=m,color=[COL[3],COL[2],COL[4]][k])
    for xi,vi in zip(x+(k-1)*bw,v): ax.text(xi,vi+1,f"{vi:.1f}",ha="center",fontsize=5.6,rotation=90,va="bottom")
ax.set_xticks(x); ax.set_xticklabels(DS,fontsize=7.5); ax.set_ylim(0,125); ax.set_yticks([0,25,50,75,100]); clean(ax,"CV PR-AUC (%)")
ax.legend(ncol=3,fontsize=6.6,frameon=False,loc="upper center",bbox_to_anchor=(0.5,1.02),handlelength=1,columnspacing=0.8)
fig.tight_layout(); fig.savefig("figures/fig_cv_models.pdf"); plt.close(fig)

# ---------- Fig: training time ----------
tt=[1812.92,592.57,42.97]; te_=[19.63,43.07,0.58]
fig,ax=plt.subplots(figsize=(3.45,2.4)); x=np.arange(3); bw=0.36
ax.bar(x-bw/2,tt,bw,label="Training",color=COL[0]); ax.bar(x+bw/2,te_,bw,label="Testing",color=COL[3])
for xi,vi in zip(x-bw/2,tt): ax.text(xi,vi*1.15,f"{vi:,.0f}s",ha="center",fontsize=6.8)
for xi,vi in zip(x+bw/2,te_): ax.text(xi,vi*1.15,f"{vi:.1f}s",ha="center",fontsize=6.8)
ax.set_yscale("log"); ax.set_ylim(0.2,9000); ax.set_xticks(x); ax.set_xticklabels(DS,fontsize=7.5); clean(ax,"Time (s, log scale)")
ax.legend(fontsize=7,frameon=False,loc="upper right")
fig.tight_layout(); fig.savefig("figures/fig_time.pdf"); plt.close(fig)

# ---------- Fig: class imbalance ----------
fig,ax=plt.subplots(figsize=(3.45,2.3)); rates=[0.129,0.104,12.19]
b=ax.bar(x,rates,0.55,color=[COL[1],COL[1],COL[1]]); ax.set_yscale("log"); ax.set_ylim(0.05,40)
for xi,vi in zip(x,rates): ax.text(xi,vi*1.2,f"{vi:g}%",ha="center",fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(DS,fontsize=7.5); clean(ax,"Positive class (%, log scale)")
fig.tight_layout(); fig.savefig("figures/fig_imbalance.pdf"); plt.close(fig)
print("done")
