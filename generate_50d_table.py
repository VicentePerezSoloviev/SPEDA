import pandas as pd


def load_data(algorithm, dim, sep= ','):
    dt = pd.read_csv('experiment_results/' + str(dim) + 'd_' + str(algorithm) + '.csv', sep=sep)
    for i in range(1, 20 + 1):
        dt.loc[dt['cost_function'] == 'cec' + str(i), 'result'] = round(dt[dt['cost_function'] == 'cec' + str(i)]['result'] - i*100, 2)
    # dt.loc[dt['cost_function'] == 'cec8', 'result'] = round(dt[dt['cost_function'] == 'cec8']['result'] - 100, 2)
    # dt = dt.groupby(variables).mean().round(2)
    return dt


dt_egna_30d = load_data('egna', 50).round(2)
dt_egna_30d = dt_egna_30d[(dt_egna_30d['size_gen'] == 300) & (dt_egna_30d['alpha'] == 0.6) & (dt_egna_30d['l'] == 10)].groupby(['cost_function']).mean()[['result']].round(2)
dt_egna_30d.columns=['EGNA']

dt_egna_30d_std = load_data('egna', 50).round(2)
dt_egna_30d_std = dt_egna_30d_std[(dt_egna_30d_std['size_gen'] == 300) & (dt_egna_30d_std['alpha'] == 0.6) & (dt_egna_30d_std['l'] == 10)].groupby(['cost_function']).std()[['result']].round(2)
dt_egna_30d_std.columns=['EGNA']

dt_egna_30d = dt_egna_30d.join(dt_egna_30d_std, lsuffix='_a')
dt_egna_30d["EGNA"] = dt_egna_30d["EGNA_a"].round(2).astype(str) + " $\pm$ " + dt_egna_30d["EGNA"].round(2).astype(str)

dt_jade_30d = load_data('jade_', 50, sep=' ').round(2)
dt_jade_30d = dt_jade_30d.groupby(['cost_function']).mean()[['result']]
dt_jade_30d.columns=['JADE']

dt_jade_30d_std = load_data('jade_', 50, sep=' ').round(2)
dt_jade_30d_std = dt_jade_30d_std.groupby(['cost_function']).std()[['result']]
dt_jade_30d_std.columns=['JADE']

dt_jade_30d = dt_jade_30d.join(dt_jade_30d_std, lsuffix='_a')
dt_jade_30d["JADE"] = dt_jade_30d["JADE_a"].round(2).astype(str) + " $\pm$ " + dt_jade_30d["JADE"].round(2).astype(str)

dt_cmaes_30d = load_data('cmaes_prueba', 50).round(2)
dt_cmaes_30d = dt_cmaes_30d.groupby(['cost_function']).mean()[['result']]
dt_cmaes_30d.columns=['CMAES']

dt_cmaes_30d_std = load_data('cmaes_prueba', 50).round(2)
dt_cmaes_30d_std = dt_cmaes_30d_std.groupby(['cost_function']).std()[['result']]
dt_cmaes_30d_std.columns=['CMAES']

dt_cmaes_30d = dt_cmaes_30d.join(dt_cmaes_30d_std, lsuffix='_a')
dt_cmaes_30d["CMAES"] = dt_cmaes_30d["CMAES_a"].round(2).astype(str) + " $\pm$ " + dt_cmaes_30d["CMAES"].round(2).astype(str)

dt_speda_30d = load_data('speda', 50).round(2)
dt_speda_30d = dt_speda_30d[(dt_speda_30d['size_gen'] == 300) & (dt_speda_30d['alpha'] == 0.4) & (dt_speda_30d['l'] == 15)].groupby(['cost_function']).mean()[['result']].round(2)
# dt_speda_30d = dt_speda_30d.groupby(['cost_function']).std()
dt_speda_30d.columns=['SPEDA']

dt_speda_30d_std = load_data('speda', 50).round(2)
dt_speda_30d_std = dt_speda_30d_std[(dt_speda_30d_std['size_gen'] == 300) & (dt_speda_30d_std['alpha'] == 0.4) & (dt_speda_30d_std['l'] == 15)].groupby(['cost_function']).std()[['result']].round(2)
# dt_speda_30d = dt_speda_30d.groupby(['cost_function']).std()
dt_speda_30d_std.columns=['SPEDA']

dt_speda_30d = dt_speda_30d.join(dt_speda_30d_std, lsuffix='_a')
dt_speda_30d["SPEDA"] = dt_speda_30d["SPEDA_a"].round(2).astype(str) + " $\pm$ " + dt_speda_30d["SPEDA"].round(2).astype(str)

dt_emna_30d = load_data('emna', 50).round(2)
dt_emna_30d = dt_emna_30d[(dt_emna_30d['size_gen'] == 300) & (dt_emna_30d['alpha'] == 0.4)].groupby(['cost_function']).mean()[['result']].round(2)
# dt_emna_30d = dt_emna_30d.groupby(['cost_function']).std()
dt_emna_30d.columns=['EMNA']

dt_emna_30d_std = load_data('emna', 50).round(2)
dt_emna_30d_std = dt_emna_30d_std[(dt_emna_30d_std['size_gen'] == 300) & (dt_emna_30d_std['alpha'] == 0.4)].groupby(['cost_function']).std()[['result']].round(2)
# dt_emna_30d = dt_emna_30d.groupby(['cost_function']).std()
dt_emna_30d_std.columns=['EMNA']

dt_emna_30d = dt_emna_30d.join(dt_emna_30d_std, lsuffix='_a')
dt_emna_30d["EMNA"] = dt_emna_30d["EMNA_a"].round(2).astype(str) + " $\pm$ " + dt_emna_30d["EMNA"].round(2).astype(str)

gb = dt_emna_30d.join(dt_egna_30d).join(dt_speda_30d).join(dt_cmaes_30d).join(dt_jade_30d).round(2)
gb = gb[['EMNA', 'EGNA', 'SPEDA', 'CMAES', 'JADE']]

gb.to_csv('2.csv')
