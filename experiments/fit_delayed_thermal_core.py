"""Diagnose a compact thermal core using public training trajectories.

This identifies physical response conditional on reconstructed equipment inputs.
It is not a deployed policy, a weather forecast, or a leaderboard estimate.
"""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import lsq_linear
from smart_buildings.evaluate import load_hybrid_environment,create_building_meta_data
from smart_buildings.smart_control.simulator.building_utils import process_and_run_connected_components

root=Path(sys.argv[1]); output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=True)
summary=json.loads((root/'summary.json').read_text());data=np.load(root/'trajectory.npz')
scenario=Path(summary['scenario']);env=load_hybrid_environment(scenario/'config.gin');meta=create_building_meta_data(env,scenario)
names=summary['observation_fields'];obs=data['observations'].astype(float);actions=data['continuous_actions'].astype(float);binary=data['discrete_actions'].astype(int)
zones=summary['zone_ids'];vidx={z:i for i,z in enumerate(zones)};K=len(meta['temp_indices']);V=len(zones)
norm=meta['observation_normalization_metadata']['zone_air_temperature_sensor']
temps=np.array([obs[:,names.index(z+'_zone_air_temperature_sensor')]*norm['sample_std']+norm['sample_mean'] for z in zones]).T
components=process_and_run_connected_components(meta['zone_map']);counts=np.bincount(components[components>0]);areas=np.array([counts[int(z.split('_')[-1])] for z in zones],float)
water_idx=next(i for i,n in enumerate(names) if n.endswith('supply_water_temperature_sensor'))
water=obs[:,water_idx]
def command(key,index):
    n=meta['action_normalization_metadata'][key]
    return n['min_native_value']+(actions[:,index]-n['min_normalized_value'])/(n['max_normalized_value']-n['min_normalized_value'])*(n['max_native_value']-n['min_native_value'])
def equipment(start):
    N=len(start);flow=np.zeros((N,V));inlet=np.zeros((N,V));outside=np.zeros((N,V));valve=np.zeros((N,V))
    ahus=sorted(meta['ahu_zone_assignments'],key=lambda x:int(x.split('_')[-1]))
    for a,ahu in enumerate(ahus):
        zs=[r.replace('room_','VAV_') for r in meta['ahu_zone_assignments'][ahu]];indices=[vidx[z] for z in zs]
        out=obs[:N,names.index('ahs_'+ahu+'_outside_air_temperature_sensor')]
        pressure=command(ahu+'_supply_air_static_pressure_setpoint',meta['pressure_indices'][a])[:N]
        pressure=np.where(binary[:N,a],np.maximum(pressure,200),0)
        recirc=np.minimum(command(ahu+'_recirculation_percentage_command',meta['recirculation_indices'][a])[:N],.7)
        target=command(ahu+'_supply_air_temperature_setpoint',meta['temp_indices'][a])[:N]
        ret=np.average(start[:,indices],axis=1,weights=areas[indices]);mixed=recirc*ret+(1-recirc)*out
        supplied=np.where(binary[:N,a],np.minimum(mixed,target),mixed)
        for z,i in zip(zs,indices):
            idx=meta['zone_action_indices'][z]
            damper=np.maximum(command('supply_air_damper_percentage_command',idx['supply_air_damper_percentage_command'])[:N],.1)
            design=obs[:N,names.index(z+'_supply_air_flowrate_setpoint')]
            flow[:,i]=np.maximum(damper*design*np.sqrt(pressure/2000),1e-5)
            valve[:,i]=np.clip(command('reheat_valve_percentage_command',idx['reheat_valve_percentage_command'])[:N],0,1)
            inlet[:,i]=supplied;outside[:,i]=out
        scale=np.minimum(1,20/np.maximum(flow[:,indices].sum(axis=1),1e-12))
        flow[:,indices]*=scale[:,None]
    wp=command('differential_pressure_setpoint',meta['pressure_indices'][-1])[:N]
    wp=np.where(binary[:N,-1],wp,0)
    qwater=valve*.108/3600*np.sqrt(np.maximum(wp,0))[:,None]
    air=1.2*1006*flow;gap=water[:N,None]-inlet
    eff=np.minimum(1,1000*4180*qwater/np.maximum(air,1e-12))
    eff=np.where(gap>0,np.minimum(eff,np.maximum(0,308.15-inlet)/np.maximum(gap,1e-12)),eff)
    heat=air*eff*gap
    return air,inlet,heat,outside
results={}
for lag in [0,1]:
    N=len(temps)-2
    start=temps[lag:lag+N];end=temps[lag+1:lag+N+1]
    air,inlet,reheat,out=equipment(start)
    mass=np.empty_like(start);mass[0]=start[0]
    keep=np.exp(-300/(4*3600))
    for t in range(1,N):mass[t]=keep*mass[t-1]+(1-keep)*start[t-1]
    split=int(N*.7);pred=np.zeros_like(end);parameters=[];ranks=[]
    for i in range(V):
        net=air[:,i]*(inlet[:,i]-end[:,i])+reheat[:,i]
        X=np.stack([out[:,i]-end[:,i],mass[:,i]-end[:,i],net/10000,np.ones(N)],axis=1)
        y=end[:,i]-start[:,i]
        reg=np.diag([.1,.1,.01,.1]);prior=np.array([.01,.05,.2,.02])
        fit=lsq_linear(np.vstack([X[:split],reg]),np.r_[y[:split],reg@prior],bounds=([0,0,.001,-2],[5,5,100,2]),tol=1e-8)
        a,b,k,c=fit.x;k=k/10000
        pred[:,i]=(start[:,i]+a*out[:,i]+b*mass[:,i]+k*(air[:,i]*inlet[:,i]+reheat[:,i])+c)/(1+a+b+k*air[:,i])
        parameters.append([a,b,k,c]);ranks.append(float(np.linalg.cond(X[:split])))
    error=pred[split:]-end[split:]
    results['delay_aligned' if lag else 'naive_alignment']={'held_out_mae_k':float(np.mean(abs(error))),'held_out_p99_abs_error_k':float(np.quantile(abs(error),.99)),'heat_gain_quantiles_k_per_w':np.quantile(np.array(parameters)[:,2],[.1,.5,.9]).tolist(),'parameters':parameters,'median_design_condition':float(np.median(ranks))}
report={'scenario':summary['scenario'],'training_steps':split,'validation_steps':N-split,'scope':'Conditional one-step physical-core identification; equipment priors approximate; not an end-to-end causal forecast or control result','model':'Positive two-node zone thermal core, four-hour filtered mass proxy; per-zone fit','results':results}
(output/'fit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='results'}));print(json.dumps({k:{q:v for q,v in d.items() if q!='parameters'} for k,d in results.items()},indent=2))
