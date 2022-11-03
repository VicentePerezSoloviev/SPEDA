import pandas as pd


def load_data(algorithm, dim, sep= ','):
    dt = pd.read_csv('experiment_results/' + str(dim) + 'd_' + str(algorithm) + '.csv', sep=sep)
    for i in range(1, 20 + 1):
        dt.loc[dt['cost_function'] == 'cec' + str(i), 'result'] = round(dt[dt['cost_function'] == 'cec' + str(i)]['result'] - i*100, 2)
    # dt.loc[dt['cost_function'] == 'cec8', 'result'] = round(dt[dt['cost_function'] == 'cec8']['result'] - 100, 2)
    # dt = dt.groupby(variables).mean().round(2)
    return dt


dt_egna_30d = load_data('egna', 30).round(2)
dt_egna_30d = dt_egna_30d[(dt_egna_30d['size_gen'] == 300) & (dt_egna_30d['alpha'] == 0.6) & (dt_egna_30d['l'] == 10)]
dt_egna_30d = dt_egna_30d[['it', 'result', 'cost_function']]
dt_egna_30d['algorithm'] = 'EGNA'

dt_jade_30d = load_data('jade_', 30, sep=' ').round(2)
dt_jade_30d = dt_jade_30d[['it', 'result', 'cost_function']]
dt_jade_30d['algorithm'] = 'JADE'

dt_cmaes_30d = load_data('cmaes_prueba', 30).round(2)
dt_cmaes_30d = dt_cmaes_30d[['it', 'result', 'cost_function']]
dt_cmaes_30d['algorithm'] = 'CMAES'

dt_speda_30d = load_data('speda', 30).round(2)
dt_speda_30d = dt_speda_30d[(dt_speda_30d['size_gen'] == 300) & (dt_speda_30d['alpha'] == 0.4) & (dt_speda_30d['l'] == 15)]
dt_speda_30d.columns = ['it', 'alpha', 'result', 'size_gen', 'cost_function', 'l']
dt_speda_30d = dt_speda_30d[['it', 'result', 'cost_function']]
dt_speda_30d['algorithm'] = 'SPEDA'

dt_emna_30d = load_data('emna', 30).round(2)
dt_emna_30d = dt_emna_30d[(dt_emna_30d['size_gen'] == 300) & (dt_emna_30d['alpha'] == 0.4)]
dt_emna_30d = dt_emna_30d[['it', 'result', 'cost_function']]
dt_emna_30d['algorithm'] = 'EMNA'

dt_tot = dt_egna_30d.append(dt_jade_30d).append(dt_cmaes_30d).append(dt_speda_30d).append(dt_emna_30d)

dt_tot
