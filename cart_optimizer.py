"""
CAR-T Cell Therapy Optimizer
==============================
Based on: "Modeling Interstellar Infection & Cellular Cures"
          A Chemotactic Fluid Approach to Reaction-Diffusion-Advection PDEs
          GROUP-23, IIT Kharagpur

Pipeline:
  1. PDE Simulator  — numerically solves the Astrophage/Taumoeba RDA system
                      using Finite Difference (FDM) + Runge-Kutta 4 (RK4)
  2. Dataset Builder — sweeps treatment parameters and records outcomes
  3. ML Optimizer   — trains Random Forest + Gradient Boosting to predict
                      tumor eradication score from treatment parameters
  4. Optimizer      — finds the best treatment strategy via model predictions

Variables (from PPT):
  A(x,y,t) : Tumor/Astrophage density         [prey]
  T(x,y,t) : CAR-T/Taumoeba cell density      [predator]
  D_A, D_T  : diffusion coefficients
  r         : tumor logistic growth rate
  K         : tumor carrying capacity
  c         : predation rate
  e         : T-cell conversion efficiency
  chi (χ)   : chemotactic sensitivity
  lambda_0  : T-cell death rate
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings, time
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. PDE SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class PDESimulator:
    """
    Solves the Reaction-Diffusion-Advection PDE system:

    Tumor (Astrophage):
      ∂A/∂t = D_A·∇²A + r·A·(1 - A/K) - c·A·T

    T-Cell (Taumoeba):
      ∂T/∂t = D_T·∇²T - χ·∇·(T·∇A) + e·c·A·T - λ₀·T

    Numerics: FDM (2nd order) + RK4 + Neumann BCs + CFL timestep
    """

    def __init__(self, N=30, dx=1.0,
                 D_A=0.05, D_T=0.6, r=0.25, K=1.0,
                 c=2.5, e=0.8, chi=1.5, lambda_0=0.05):
        self.N=N; self.dx=dx
        self.D_A=D_A; self.D_T=D_T
        self.r=r; self.K=K
        self.c=c; self.e=e
        self.chi=chi; self.lam=lambda_0
        self.dt = 0.25 * dx**2 / (4 * max(D_A, D_T) + 1e-9)

    def _laplacian(self, F):
        lap = np.zeros_like(F)
        lap[1:-1,1:-1] = (F[2:,1:-1]+F[:-2,1:-1]+F[1:-1,2:]+F[1:-1,:-2]-4*F[1:-1,1:-1])/self.dx**2
        return lap

    def _chemotaxis(self, T, A):
        dAx=np.zeros_like(A); dAy=np.zeros_like(A)
        dAx[1:-1,1:-1]=(A[1:-1,2:]-A[1:-1,:-2])/(2*self.dx)
        dAy[1:-1,1:-1]=(A[2:,1:-1]-A[:-2,1:-1])/(2*self.dx)
        Jx=T*dAx; Jy=T*dAy
        div=np.zeros_like(T)
        div[1:-1,1:-1]=((Jx[1:-1,2:]-Jx[1:-1,:-2])+(Jy[2:,1:-1]-Jy[:-2,1:-1]))/(2*self.dx)
        return div

    def _rhs(self, A, T):
        A=np.clip(A,0,None); T=np.clip(T,0,None)
        dA=self.D_A*self._laplacian(A)+self.r*A*(1-A/self.K)-self.c*A*T
        dT=self.D_T*self._laplacian(T)-self.chi*self._chemotaxis(T,A)+self.e*self.c*A*T-self.lam*T
        return dA, dT

    def _rk4(self, A, T):
        dt=self.dt
        k1A,k1T=self._rhs(A,T)
        k2A,k2T=self._rhs(A+.5*dt*k1A,T+.5*dt*k1T)
        k3A,k3T=self._rhs(A+.5*dt*k2A,T+.5*dt*k2T)
        k4A,k4T=self._rhs(A+dt*k3A,T+dt*k3T)
        return np.clip(A+(dt/6)*(k1A+2*k2A+2*k3A+k4A),0,None), \
               np.clip(T+(dt/6)*(k1T+2*k2T+2*k3T+k4T),0,None)

    def initial_tumor(self):
        N=self.N; cx=cy=N//2
        x=np.arange(N); XX,YY=np.meshgrid(x,x)
        return self.K*np.exp(-((XX-cx)**2+(YY-cy)**2)/(2*(N/6)**2))

    def run(self, t_max=40.0, inject_dose=1.0,
            inject_x=0.5, inject_y=0.5, inject_time=1.0):
        N=self.N
        x=np.arange(N); XX,YY=np.meshgrid(x,x)
        A=self.initial_tumor()
        T=np.zeros((N,N))
        injected=False
        ix=int(inject_x*(N-1)); iy=int(inject_y*(N-1))
        sig=max(2,int(N*0.1))

        steps=int(t_max/self.dt)
        tumor_hist=[]
        t=0.0

        for step in range(steps):
            if not injected and t>=inject_time:
                T+=inject_dose*np.exp(-((XX-ix)**2+(YY-iy)**2)/(2*sig**2))
                injected=True
            A,T=self._rk4(A,T)
            t+=self.dt
            if step%(max(1,steps//100))==0:
                tumor_hist.append(float(np.sum(A)))

        return A, T, np.array(tumor_hist)

    def eradication_score(self, A_final, A0_total):
        return float(np.clip(1-np.sum(A_final)/(A0_total+1e-9),0,1))


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(n_samples=500, seed=42):
    rng=np.random.default_rng(seed)
    print(f"  Building dataset with {n_samples} simulations...")

    base=PDESimulator(N=30)
    A0_total=float(np.sum(base.initial_tumor()))

    records=[]; t0=time.time()
    for i in range(n_samples):
        dose     = rng.uniform(0.5,  4.0)
        inj_x    = rng.uniform(0.0,  1.0)
        inj_y    = rng.uniform(0.0,  1.0)
        inj_t    = rng.uniform(0.0,  8.0)
        chi      = rng.uniform(0.5,  3.0)
        D_T      = rng.uniform(0.2,  1.0)
        lam      = rng.uniform(0.01, 0.15)
        e        = rng.uniform(0.5,  0.95)

        sim=PDESimulator(N=30, chi=chi, D_T=D_T, lambda_0=lam, e=e)
        A_f,T_f,_=sim.run(t_max=40, inject_dose=dose,
                           inject_x=inj_x, inject_y=inj_y, inject_time=inj_t)
        score=sim.eradication_score(A_f, A0_total)

        records.append({'dose':dose,'inject_x':inj_x,'inject_y':inj_y,
                        'inject_time':inj_t,'chi':chi,'D_T':D_T,
                        'death_rate':lam,'conversion_eff':e,
                        'eradication_score':score})

        if (i+1)%100==0:
            avg=np.mean([r['eradication_score'] for r in records])
            print(f"    {i+1}/{n_samples} | {time.time()-t0:.1f}s | avg score: {avg:.3f}")

    import pandas as pd
    df=pd.DataFrame(records)
    df.to_csv('cart_dataset.csv',index=False)
    print(f"  Dataset saved → cart_dataset.csv  ({len(df)} rows)\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. ML MODELS
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS=['dose','inject_x','inject_y','inject_time','chi','D_T','death_rate','conversion_eff']
FEATURE_NAMES={
    'dose':'CAR-T Dose','inject_x':'Injection X Position',
    'inject_y':'Injection Y Position','inject_time':'Injection Timing (days)',
    'chi':'Chemotactic Sensitivity (χ)','D_T':'T-cell Diffusivity (D_T)',
    'death_rate':'T-cell Death Rate (λ₀)','conversion_eff':'Conversion Efficiency (e)',
}

def train_models(df):
    print("  Training ML models...")
    X=df[FEATURE_COLS].values; y=df['eradication_score'].values
    scaler=StandardScaler()
    Xs=scaler.fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.2,random_state=42)

    models={
        'Random Forest':RandomForestRegressor(n_estimators=300,max_depth=12,random_state=42,n_jobs=-1),
        'Gradient Boosting':GradientBoostingRegressor(n_estimators=300,max_depth=6,learning_rate=0.04,random_state=42),
    }
    results={}
    for name,m in models.items():
        m.fit(Xtr,ytr); p=m.predict(Xte)
        r2=r2_score(yte,p); rmse=np.sqrt(mean_squared_error(yte,p))
        cv=cross_val_score(m,Xs,y,cv=5,scoring='r2')
        results[name]={'model':m,'preds':p,'r2':r2,'rmse':rmse,
                       'cv_mean':cv.mean(),'cv_std':cv.std(),'y_test':yte}
        print(f"    {name}: R²={r2:.4f} | RMSE={rmse:.4f} | CV R²={cv.mean():.4f}±{cv.std():.4f}")

    best_name=max(results,key=lambda k:results[k]['r2'])
    print(f"  Best model: {best_name}\n")
    return results, results[best_name]['model'], scaler, best_name

def find_optimal(model, scaler, n=80000, seed=0):
    rng=np.random.default_rng(seed)
    C=np.column_stack([rng.uniform(0.5,4.0,n),rng.uniform(0,1,n),
                       rng.uniform(0,1,n),rng.uniform(0,8,n),
                       rng.uniform(0.5,3.0,n),rng.uniform(0.2,1.0,n),
                       rng.uniform(0.01,0.15,n),rng.uniform(0.5,0.95,n)])
    scores=model.predict(scaler.transform(C))
    idx=np.argmax(scores)
    return C[idx], scores[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(df, results, best_model, scaler, best_name, opt_params, opt_score):
    sns.set_style('darkgrid')
    DARK='#0d0f14'; SURF='#161923'; ACC='#7c6af7'; ACC2='#4fc3f7'
    GRN='#4ade80'; TXT='#e2e8f0'; MUT='#94a3b8'

    fig=plt.figure(figsize=(20,13)); fig.patch.set_facecolor(DARK)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.35,
                         left=0.06,right=0.97,top=0.91,bottom=0.25)

    def style(ax,title):
        ax.set_facecolor(SURF)
        ax.tick_params(colors=MUT)
        ax.xaxis.label.set_color(MUT); ax.yaxis.label.set_color(MUT)
        ax.set_title(title,fontsize=11,fontweight='bold',color=TXT,pad=8)
        for s in ax.spines.values(): s.set_color('#252b38')

    # Panel 1: Score distribution
    ax1=fig.add_subplot(gs[0,0])
    ax1.hist(df['eradication_score'],bins=30,color=ACC,alpha=0.85,edgecolor='none')
    ax1.axvline(df['eradication_score'].mean(),color=GRN,lw=2,linestyle='--',
                label=f"Mean={df['eradication_score'].mean():.2f}")
    ax1.legend(facecolor=SURF,edgecolor='none',labelcolor=TXT,fontsize=9)
    ax1.set_xlabel('Eradication Score'); ax1.set_ylabel('Count')
    style(ax1,'Distribution of Treatment Outcomes')

    # Panel 2: Feature importance
    ax2=fig.add_subplot(gs[0,1])
    imp=best_model.feature_importances_
    labels=[FEATURE_NAMES[f] for f in FEATURE_COLS]
    order=np.argsort(imp)
    colors=[GRN if i==order[-1] else ACC for i in order]
    ax2.barh(np.array(labels)[order],imp[order],color=colors,alpha=0.85)
    ax2.set_xlabel('Importance')
    style(ax2,f'Feature Importance ({best_name})')

    # Panel 3: Predicted vs Actual
    ax3=fig.add_subplot(gs[0,2])
    res=results[best_name]
    ax3.scatter(res['y_test'],res['preds'],alpha=0.5,s=15,color=ACC2,edgecolors='none')
    ax3.plot([0,1],[0,1],'r--',lw=1.5,label='Perfect fit')
    ax3.set_xlabel('Actual Score'); ax3.set_ylabel('Predicted Score')
    ax3.legend(facecolor=SURF,edgecolor='none',labelcolor=TXT,fontsize=9)
    ax3.text(0.05,0.90,f"R² = {res['r2']:.4f}",transform=ax3.transAxes,
             color=GRN,fontsize=11,fontweight='bold')
    style(ax3,'Predicted vs Actual Eradication Score')

    # Panel 4: Dose vs Score
    ax4=fig.add_subplot(gs[1,0])
    sc=ax4.scatter(df['dose'],df['eradication_score'],c=df['chi'],
                   cmap='plasma',s=12,alpha=0.7)
    cb=plt.colorbar(sc,ax=ax4); cb.set_label('χ (Chemotaxis)',color=MUT)
    cb.ax.yaxis.set_tick_params(color=MUT)
    plt.setp(cb.ax.yaxis.get_ticklabels(),color=MUT)
    ax4.set_xlabel('CAR-T Dose'); ax4.set_ylabel('Eradication Score')
    style(ax4,'Dose vs Outcome (coloured by χ)')

    # Panel 5: Injection timing vs Score
    ax5=fig.add_subplot(gs[1,1])
    sc2=ax5.scatter(df['inject_time'],df['eradication_score'],
                    c=df['dose'],cmap='viridis',s=12,alpha=0.7)
    cb2=plt.colorbar(sc2,ax=ax5); cb2.set_label('Dose',color=MUT)
    cb2.ax.yaxis.set_tick_params(color=MUT)
    plt.setp(cb2.ax.yaxis.get_ticklabels(),color=MUT)
    ax5.set_xlabel('Injection Timing (days)'); ax5.set_ylabel('Eradication Score')
    style(ax5,'Injection Timing vs Outcome')

    # Panel 6: Optimal treatment summary (text box)
    ax6=fig.add_subplot(gs[1,2])
    ax6.set_facecolor(SURF); ax6.axis('off')
    for s in ax6.spines.values(): s.set_color('#252b38')
    ax6.set_title('Optimal Treatment Strategy',fontsize=11,fontweight='bold',color=TXT,pad=8)
    ax6.text(0.5,0.90,f'Predicted Eradication: {opt_score:.1%}',
             transform=ax6.transAxes,ha='center',fontsize=13,fontweight='bold',color=GRN)
    for idx,(feat,val) in enumerate(zip(FEATURE_COLS,opt_params)):
        y=0.77-idx*0.095
        ax6.text(0.05,y,FEATURE_NAMES[feat]+':',transform=ax6.transAxes,fontsize=8.5,color=MUT)
        ax6.text(0.95,y,f'{val:.3f}',transform=ax6.transAxes,fontsize=8.5,
                 color=ACC2,ha='right',fontweight='bold')

    # PDE spatial snapshots (bottom strip)
    sim_viz=PDESimulator(N=40,chi=1.8,D_T=0.7,e=0.85,lambda_0=0.04,c=2.5,r=0.25)
    A_none,_,_=sim_viz.run(t_max=40,inject_dose=0,inject_time=999)
    A_opt,T_opt,_=sim_viz.run(t_max=40,inject_dose=3.0,inject_x=0.5,inject_y=0.5,inject_time=1.0)

    insets=[
        (sim_viz.initial_tumor(),'Tumor: t=0 (Initial)','Reds',[0.06,0.02,0.24,0.20]),
        (A_none,'Tumor: No Treatment','Reds',              [0.36,0.02,0.24,0.20]),
        (A_opt, 'Tumor: Optimized CAR-T','Reds',           [0.60,0.02,0.24,0.20]),
        (T_opt, 'CAR-T Cells (Final)','Blues',             [0.76,0.02,0.18,0.20]),
    ]
    for data,title,cmap,pos in insets:
        axin=fig.add_axes(pos)
        axin.imshow(data,cmap=cmap,vmin=0,origin='lower')
        axin.set_title(title,fontsize=7,color=TXT,pad=2)
        axin.axis('off'); axin.set_facecolor(DARK)

    fig.text(0.5,0.21,'← PDE Simulation Snapshots (100×100 tissue domain) →',
             ha='center',fontsize=9,color=MUT,style='italic')

    fig.suptitle('CAR-T Cell Therapy Optimizer  |  RDA-PDE Simulation + Machine Learning\n'
                 'Based on MIICC Project — GROUP-23, IIT Kharagpur',
                 fontsize=14,fontweight='bold',color=TXT,y=0.98)

    plt.savefig('cart_results.png',dpi=150,bbox_inches='tight',facecolor=DARK)
    print("  Plot saved → cart_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__=='__main__':
    print("\n"+"═"*65)
    print("  CAR-T Cell Therapy Optimizer")
    print("  Based on: MIICC — GROUP-23, IIT Kharagpur")
    print("═"*65+"\n")

    print("STEP 1: Running PDE simulations to build training dataset")
    print("─"*65)
    df=build_dataset(n_samples=500)

    print("STEP 2: Training ML models on simulation outcomes")
    print("─"*65)
    results,best_model,scaler,best_name=train_models(df)

    print("STEP 3: Searching for optimal treatment strategy")
    print("─"*65)
    opt_params,opt_score=find_optimal(best_model,scaler)

    print("\n  ★ OPTIMAL TREATMENT STRATEGY ★")
    print("─"*65)
    for feat,val in zip(FEATURE_COLS,opt_params):
        print(f"  {FEATURE_NAMES[feat]:<38} {val:.4f}")
    print(f"\n  Predicted Eradication Score: {opt_score:.1%}")

    print("\nSTEP 4: Validating with PDE simulation")
    print("─"*65)
    val_sim=PDESimulator(N=40,chi=opt_params[4],D_T=opt_params[5],
                         lambda_0=opt_params[6],e=opt_params[7])
    A0_total=float(np.sum(val_sim.initial_tumor()))
    A_f,T_f,_=val_sim.run(t_max=40,inject_dose=opt_params[0],
                           inject_x=opt_params[1],inject_y=opt_params[2],
                           inject_time=opt_params[3])
    actual=val_sim.eradication_score(A_f,A0_total)
    print(f"  ML predicted  : {opt_score:.1%}")
    print(f"  PDE validated : {actual:.1%}")

    print("\nSTEP 5: Generating visualisations")
    print("─"*65)
    make_plots(df,results,best_model,scaler,best_name,opt_params,opt_score)

    print("\n"+"═"*65)
    print("  COMPLETE. Output files:")
    print("    cart_dataset.csv  — 500-row simulation dataset")
    print("    cart_results.png  — full results figure")
    print("═"*65+"\n")
