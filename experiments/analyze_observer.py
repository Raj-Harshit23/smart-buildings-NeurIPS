"""Measure current-state estimates using delayed labels after a completed run.

Future sensor values are evaluation labels only. Every replayed action must
exactly equal the recorded causal action before an estimate is measured.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from smart_buildings.evaluate import load_hybrid_environment,create_building_meta_data
from smart_buildings.baselines.lag_aware_allocator import LagAwareAllocatorPolicy

root=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=True)
summary=json.loads((root/'summary.json').read_text());source=Path('smart_buildings/baselines/lag_aware_allocator.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()==summary['source_sha256'][str(source)]
scenario=Path(summary['scenario']);env=load_hybrid_environment(scenario/'config.gin');metadata=create_building_meta_data(env,scenario)
policy=LagAwareAllocatorPolicy(metadata,env.action_spec());data=np.load(root/'trajectory.npz');raw=[];estimated=[];truth=[];weights=[]
observations=data['observations'];continuous=data['continuous_actions'];discrete=data['discrete_actions']
for t,obs in enumerate(observations):
 action=policy.get_action(obs)
 if not np.array_equal(action['continuous_action'],continuous[t]) or not np.array_equal(action['discrete_action'],discrete[t]):
  raise ValueError(f'Causal replay differs from recorded action at step {t}')
 if t+1==len(observations):break
 raw.append(np.asarray([policy._denormalize_zone_temp(obs[i]) for i in policy._indices]))
 estimated.append(policy._previous_temps.copy())
 truth.append(np.asarray([policy._denormalize_zone_temp(observations[t+1,i]) for i in policy._indices]))
 weights.append(policy.last_diagnostics['observer_weight'])
 if (t+1)%1024==0:print(json.dumps({'replayed_steps':t+1}),flush=True)
raw=np.asarray(raw);estimated=np.asarray(estimated);truth=np.asarray(truth)
metrics={}
for label,start in [('whole_episode',0),('after_one_day',288)]:
 er=abs(raw[start:]-truth[start:]);ee=abs(estimated[start:]-truth[start:])
 metrics[label]={'stale_sensor_mae_k':float(er.mean()),'observer_mae_k':float(ee.mean()),
                 'stale_sensor_p99_k':float(np.quantile(er,.99)),'observer_p99_k':float(np.quantile(ee,.99))}
valve_indices=[metadata['zone_action_indices'][z]['reheat_valve_percentage_command'] for z in policy._ids]
valves=(continuous[:,valve_indices].astype(float)+1)/2
pidx=metadata['pressure_indices'][-1];pressure=10*(continuous[:,pidx].astype(float)+1)
pressure=np.where(discrete[:,-1]>0,pressure,0)
maximum=valves.max(axis=1);q=valves.sum(axis=1)*.108/3600*np.sqrt(pressure)
pump=1e5*pressure*q/.98
potential=pump*(1-maximum**2)
report={'scenario':summary['scenario'],'actions_replayed_exactly':len(raw)+1,'scope':'Causal replay; future sensor values are offline current-temperature labels only',
        'estimation':metrics,'mean_observer_weight':float(np.mean(weights)),
        'conditional_hydraulic_opportunity':{'assumptions':'Zero header resistance, unchanged water flow, current pump-efficiency prior; not measured score improvement','mean_pump_power_w':float(pump.mean()),'mean_potential_saving_w':float(potential.mean())}}
(output/'analysis.json').write_text(json.dumps(report,indent=2));np.savez_compressed(output/'estimates.npz',raw=raw,estimated=estimated,truth=truth,weight=weights)
print(json.dumps(report,indent=2),flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
zone=int(np.argmax(np.mean(abs(raw-truth)-abs(estimated-truth),axis=0)));begin=576;end=begin+288;hours=np.arange(begin,end)/12
fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
axes[0].plot(hours,truth[begin:end,zone],label='Physical temperature (offline delayed label)',lw=1.5,color='#18273c')
axes[0].plot(hours,raw[begin:end,zone],label='Observed stale temperature',lw=1,color='#d27724',alpha=.8)
axes[0].plot(hours,estimated[begin:end,zone],label='Causal current-state estimate',lw=1,color='#168c7c',alpha=.85)
axes[0].set_ylabel('Temperature (K)');axes[0].legend(fontsize=8);axes[0].set_title('Phoenix / building 8: '+policy._ids[zone]+', evaluation day 3')
axes[1].plot(hours,abs(raw[begin:end,zone]-truth[begin:end,zone]),label='Stale sensor error',color='#d27724')
axes[1].plot(hours,abs(estimated[begin:end,zone]-truth[begin:end,zone]),label='Observer error',color='#168c7c')
axes[1].set_xlabel('Elapsed simulation hours');axes[1].set_ylabel('Absolute error (K)');axes[1].legend(fontsize=8)
fig.savefig(output/'observer.png',dpi=160);plt.close(fig)
