import time
import numpy as np
import pandas as pd
from pyomo.environ import *
from sklearn.cluster import AgglomerativeClustering, KMeans


def time_model_solve(inputs, renewable, load, weight=None):
    # Benchmark method
    horizon = len(load)  # Time horizon

    m = ConcreteModel()  # Model preparation and initial parameters

    m.eta = inputs["effi"]

    m.rps = inputs["rps"]

    m.horizon = horizon

    m.H = RangeSet(0, m.horizon - 1)

    m.renewable = renewable

    renewable_type = len(renewable)
    m.R = RangeSet(0, renewable_type - 1)

    m.load = load

    m.c_bat = inputs["c_bat"]
    m.c_bat_power = inputs["c_bat_power"]
    m.c_renewable = [inputs['c_wind'], inputs['c_pv']]
    m.c_gen_inv = inputs["c_gen_inv"]
    m.mdc = inputs["mdc"]
    m.c_gen_a = inputs["c_gen_a"]
    m.c_gen_b = inputs["c_gen_b"]

    if weight is None:
        weight = np.ones(horizon)
    m.weight = weight
    sum_load = np.sum(m.load[i]*m.weight[i] for i in m.H)
    m.renewable_cap = Var(m.R, domain=NonNegativeReals)

    m.max_energy = Var(domain=NonNegativeReals)  # Battery energy capacity
    m.max_power = Var(domain=NonNegativeReals)  # Battery power capacity

    m.pd = Var(m.H, domain=Reals)  # Battery discharging power at time i
    m.pc = Var(m.H, domain=Reals)  # Battery charging power at time i
    m.e = Var(m.H, domain=Reals)  # Battery Energy Stored (SOC) at time i

    m.curtail = Var(m.H, domain=Reals)  # Curtailed renewable energy at time i
    m.gen = Var(m.H, domain=Reals)  # Thermal generator power output at time i

    m.total_cost = Var(domain=Reals)
    m.cost_sto_inv = Var(domain=Reals)
    m.cost_pv_inv = Var(domain=Reals)
    m.cost_gen_inv = Var(domain=Reals)
    m.cost_var = Var(domain=Reals)

    # Number of online thermal generator units at time i
    m.n = Var(m.H, domain=NonNegativeIntegers)
    # Number of starting-up thermal generator units at time i
    m.n_start = Var(m.H, domain=NonNegativeIntegers)
    # Number of shutting-down thermal generator units at time i
    m.n_shut = Var(m.H, domain=NonNegativeIntegers)
    # Number of thermal generator unit number
    m.N = Var(domain=NonNegativeIntegers)
    m.gen_cap = inputs["gen_cap"]
    m.up_time = inputs["up_time"]  # Minimum online time
    m.down_time = inputs["down_time"]  # Minimum offline time

    # constraints set
    function_i = 0
    function_list = []

    def fun(m, i):
        return m.gen[i] + sum(m.renewable_cap[r] * renewable[r][i] for r in m.R) + m.pd[i] - m.pc[i] - m.curtail[i] - \
               m.load[i] == 0

    function_list.append(fun)
    m.balance_cons = Constraint(m.H, rule=function_list[function_i])  # Load balance
    function_i += 1

    def fun(m, i):
        if i == 0:
            return m.e[i] - m.e[m.horizon - 1] + m.pd[m.horizon - 1] / m.eta - m.pc[m.horizon - 1] * m.eta == 0
        else:
            return m.e[i] - (m.e[i - 1] - m.pd[i - 1] / m.eta + m.pc[i - 1] * m.eta) == 0

    function_list.append(fun)
    m.soc3 = Constraint(m.H, rule=function_list[function_i])  # Storage constraints-storage change
    function_i += 1

    def fun(m, i):
        return m.e[i] >= 0

    function_list.append(fun)
    m.soc1 = Constraint(m.H, rule=function_list[function_i])  # Storage constraints-nonnegative storage
    function_i += 1

    def fun(m, i):
        return m.e[i] - m.max_energy <= 0

    function_list.append(fun)
    m.soc2 = Constraint(m.H, rule=function_list[function_i])  # Storage constraints-maximum storage
    function_i += 1

    def fun(m, i):
        return m.pc[i] >= 0

    function_list.append(fun)
    m.var_b6 = Constraint(m.H, rule=function_list[function_i])  # Storage constraints-nonnegative charging
    function_i += 1

    def fun(m, i):
        return m.pd[i] >= 0

    function_list.append(fun)
    m.var_b7 = Constraint(m.H, rule=function_list[function_i])  # Storage constraints-nonnegative discharging
    function_i += 1

    def fun(m, i):
        return m.pc[i] <= m.max_power

    function_list.append(fun)
    m.var_b8 = Constraint(m.H, rule=function_list[function_i])  # Storage constraints-maximum charging power
    function_i += 1

    def fun(m, i):
        return m.pd[i] <= m.max_power

    function_list.append(fun)
    m.var_b9 = Constraint(m.H, rule=function_list[function_i])  # Storage constraints-maximum discharging power
    function_i += 1

    def fun(m, i):
        return m.curtail[i] >= 0

    function_list.append(fun)
    m.var_b3 = Constraint(m.H, rule=function_list[function_i])  # 
    function_i += 1

    def fun(m, i):
        return m.gen[i] - 0.1 * m.gen_cap * m.n[i] >= 0

    function_list.append(fun)
    m.var_gen1 = Constraint(m.H, rule=function_list[function_i])
    # Thermal generator constraints-minimum generation percentage(0.1)
    function_i += 1

    def fun(m, i):
        return m.gen[i] - m.n[i] * m.gen_cap <= 0

    function_list.append(fun)
    m.var_gen2 = Constraint(m.H, rule=function_list[function_i])
    # Thermal generator constraints-maximum generation percentage(1.0)
    function_i += 1

    def fun(m, i):
        return m.n[i] - m.N <= 0

    function_list.append(fun)
    m.var_gen3 = Constraint(m.H, rule=function_list[function_i])  # Thermal generator constraints-maximum units
    function_i += 1

    def fun(m, i):
        return m.n[i] >= 0

    function_list.append(fun)
    m.var_gen4 = Constraint(m.H, rule=function_list[function_i])  # Thermal generator constraints-minimum units
    function_i += 1

    def fun(m, i):
        if i == 0:
            return m.n[i] - m.n[m.horizon - 1] - (m.n_start[m.horizon - 1] - m.n_shut[m.horizon - 1]) == 0
        else:
            return m.n[i] - m.n[i - 1] - (m.n_start[i - 1] - m.n_shut[i - 1]) == 0

    function_list.append(fun)
    m.var_gen5 = Constraint(m.H, rule=function_list[function_i])  # Thermal generator constraints-units change
    function_i += 1

    def fun(m, i):
        return m.n_start[i] >= 0

    function_list.append(fun)
    m.var_gen6 = Constraint(m.H, rule=function_list[function_i])  # Thermal generator constraints-nonnegative start_up
    function_i += 1

    def fun(m, i):
        if i >= m.up_time:
            t_start_list = np.arange(i - m.up_time, i)
        else:
            t_start_list = np.arange(0, i)
        return m.n[i] - sum(m.n_start[k] for k in t_start_list) >= 0

    function_list.append(fun)
    m.var_gen7 = Constraint(m.H, rule=function_list[function_i])  # Thermal generator constraints-up_time
    function_i += 1

    def fun(m, i):
        if i >= m.down_time:
            t_shut_list = np.arange(i - m.down_time, i)
        else:
            t_shut_list = np.arange(0, i)
        return m.N - m.n[i] - sum(m.n_shut[k] for k in t_shut_list) >= 0

    function_list.append(fun)
    m.var_gen8 = Constraint(m.H, rule=function_list[function_i])  # Thermal generator constraints-down_time
    function_i += 1

    def fun(m, i):
        return m.n_shut[i] >= 0

    function_list.append(fun)
    m.var_gen9 = Constraint(m.H, rule=function_list[function_i])  # Thermal generator constraints-nonnegative shut_down
    function_i += 1

    def fun(m):
        return sum(m.gen[i]*m.weight[i] for i in m.H) / sum_load - (1 - m.rps) <= 0

    function_list.append(fun)
    m.rps_limit = Constraint(rule=function_list[function_i])  # Thermal generator constraints-total percentage
    function_i += 1

    def obj_value(m):
        return m.total_cost

    def obj_function(m):
        return m.total_cost == m.cost_var + m.cost_sto_inv + m.cost_pv_inv + m.cost_gen_inv

    m.OF = Constraint(rule=obj_function)

    def cost_gen_cal(m):
        return m.cost_var == sum(m.weight[i] * (m.pd[i] + m.pc[i]) for i in m.H) * m.mdc + \
               sum(m.weight[i] * m.gen[i] for i in m.H) * m.c_gen_b  # Weight of variable costs

    m.rev = Constraint(rule=cost_gen_cal)

    def cost_storage_cal(m):
        return m.cost_sto_inv == m.max_energy * m.c_bat + m.max_power * m.c_bat_power

    m.cost_bat = Constraint(rule=cost_storage_cal)  # Storage cost

    def cost_renewable_cal(m):
        return m.cost_pv_inv == sum(m.renewable_cap[r] * m.c_renewable[r] for r in m.R)

    m.cost_pv = Constraint(rule=cost_renewable_cal)  # Renewable cost

    def cost_gen_cal(m):
        return m.cost_gen_inv == m.gen_cap * m.c_gen_inv * m.N

    m.cost_gen = Constraint(rule=cost_gen_cal)  # Thermal cost

    m.OBJ = Objective(rule=obj_value, sense=minimize)

    # Solver
    if inputs["solver"] == 1:
        opt = SolverFactory('gurobi', executable="/usr/local/gurobi/gurobi903/linux64/bin/gurobi.sh")
        opt.options['timelimit'] = inputs["timelimit"]

        if inputs["print_log"] ==1:
            results = opt.solve(m,tee=True)
        else:
            results = opt.solve(m)

    else:

        opt = SolverFactory('cplex')
        opt.options['timelimit'] = inputs["timelimit"]

        if inputs["print_log"] ==1:
            results = opt.solve(m,tee=True)
        else:
            results = opt.solve(m)
    
    return(m)


def sim_features(config, wind_all, pv_all, load_all):
    """
    Run the optimization model for each period and generate a DataFrame containing the simulated features
    specified in settings from config.
    """
    inputs = config['inputs']
    settings = config['settings']

    inputs['c_renewable'] = [inputs['c_wind'], inputs['c_pv']]
   
    features = []
    feature_set = settings['feature_set']
    period = settings['period']
    
    day_num = settings['day_num']
    periods = settings['day_num']*24//period
    time_set = np.arange(24*day_num)  # Time horizon specified to hour
    
    renewable = [wind_all[time_set, settings['profile_id']], pv_all[time_set, settings['profile_id']]] if 'profile_id' in settings\
        else [wind_all[time_set], pv_all[time_set]]  # for NE
    
    nrenewable = len(renewable)
    period_renewable = [renewable[r].reshape(
        periods, period) for r in range(nrenewable)]
    period_load = load_all[time_set].reshape(periods, period)
    for w in range(periods):
        renewable = [r[w] for r in period_renewable]
        load = period_load[w]
        m = time_model_solve(inputs, renewable, load)
        results = {}
        for v in feature_set:
            var_object = getattr(m, v)
            if var_object.is_indexed():
                for t in range(len(var_object)):
                    results[v+'_'+str(t)] = var_object[t].value
            else:
                results[v] = var_object.value
                
        features.append(results)
    return pd.DataFrame(features)


def cluster(settings, data, cluster_log=False):
    """
    Given a dictionary of settings and a dataframe of data points to cluster, return the weight of 
    each representative week and the representative renewable and load values.

    If cluster_log=True, returns a dictionary mapping cluster labels to points in each cluster.
    """
    method = settings['method']
    # Used for kmeans random state
    init = settings['init'] if 'init' in settings else None
    connectivity = settings['connectivity'] if 'connectivity' in settings else False
    chronology = settings['chronology'] if 'chronology' in settings else False
    ncluster = settings['ncluster']

    df = data.copy()
    period = settings['period']
    periods = settings['periods']
    nrenewable = settings['nrenewable']
    period_df = settings['period_df']
    
    renewable_range = [np.arange(r*period, (r+1)*period)
                       for r in range(nrenewable)]
    load_range = np.arange(nrenewable*period, (nrenewable+1)*period)

    if method == 'kmeans':
        kmeans = KMeans(n_clusters=ncluster, random_state=init)
        kmeans.fit(df)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        df['cluster'] = labels
    else:
        if connectivity:
            # generate connectivity matrix
            connections = np.zeros([periods, periods])
            for i in range(periods-1):
                connections[i, i+1] = 1
        else:
            connections = None

        agglomerative = AgglomerativeClustering(
            n_clusters=ncluster, linkage=method, connectivity=connections)
        agglomerative.fit(df)
        labels = agglomerative.labels_
        n_features = len(df.columns)
        df['cluster'] = labels
        lens = {}
        centroids = {}
        for w in range(periods):
            label = df.loc[w, 'cluster']
            centroids.setdefault(label, [0]*n_features)
            centroids[label] += df.loc[w, df.columns != 'cluster']
            lens.setdefault(label, 0)
            lens[label] += 1
        for k in centroids:
            centroids[k] /= float(lens[k])

    weights = np.bincount(labels)  # per period
    weight = np.repeat(weights, period)
    clusters = {}
    for k in range(ncluster):
        clusters[k] = df.loc[df['cluster'] == k]

    # assuming only for combined case
    if 'centroid' in settings and settings['centroid'] and settings['trial'] == 'combined':
        rep_renewable = [np.concatenate([centroids[k][r] for k in range(ncluster)]) for r in renewable_range]
        rep_load = np.concatenate([centroids[k][load_range]
                                  for k in range(ncluster)])*100
    else:
        # Find representative points
        rep = [None]*ncluster
        for k in range(ncluster):
            dist = {}
            for j in range(weights[k]):
                dist[clusters[k].index[j]] = np.linalg.norm(
                    df.loc[clusters[k].index[j], df.columns != 'cluster']-centroids[k][:])
            rep[k] = min(dist, key=lambda k: dist[k])
        if chronology:
            rep.sort()
        #print('representative week indices:', rep)

        renewable = [period_df.loc[rep, r] for r in renewable_range]
        load = period_df.loc[rep, load_range]*100

        rep_renewable = [np.concatenate(
            [renewable[r].loc[j, :] for j in rep]) for r in range(nrenewable)]
        rep_load = np.concatenate([load.loc[j, :] for j in rep])

    if cluster_log:
        return weight, rep_renewable, rep_load, clusters

    return weight, rep_renewable, rep_load


def test_clustering(inputs, settings, expected, data):
    """
    Cluster the data using the settings, run the optimization model with the representative scenario generated, 
    and calculate the relative error with respect to the benchmark (expected). 
    """
    periods = settings['periods']
    
    feature_set = ['renewable_cap', 'N', 'max_energy', 'max_power', 'total_cost'] \
        if inputs['gen_cap'] == 1 else [
        'renewable_cap', 'max_energy', 'max_power', 'total_cost']
    error_terms = ['renewable_cap_0', 'renewable_cap_1', 'N', 'max_energy', 'max_power', 'total_cost'] \
        if inputs['gen_cap'] == 1 else [
        'renewable_cap_0', 'renewable_cap_1', 'max_energy', 'max_power', 'total_cost']

    weight, rep_renewable, rep_load = cluster(settings, data)

    m = time_model_solve(inputs, rep_renewable, rep_load, weight)
    opt_results = {}
    errors = {}
    # for v in m.component_objects(Var, active=True):
    for v in feature_set:
        var_object = getattr(m, str(v))
        if var_object.is_indexed():
            opt_results[str(v)] = []
            for t in range(len(var_object)):
                opt_results[v].append(var_object[t].value)
                
        elif len(var_object) == 1:
            opt_results[v] = var_object.value

    for re in range(len(opt_results['renewable_cap'])):
        opt_results['renewable_cap_{}'.format(re)] = opt_results['renewable_cap'][re]

    for e in error_terms:
        errors[e + '_err'] = abs(opt_results[e] - expected[e])/(expected[e]+0.0001)
        
    results = {**opt_results, **errors, 'mae': sum(value for value in errors.values()) / len(errors)}

    return results


def run_trials(config, wind_all, pv_all, load_all, expected, features):
    """
    Export a dataframe containing the results of clustering with the specified settings. 
    """
    inputs = config['inputs']
    settings = config['settings']
    ranges = config['ranges']
    
    day_num = settings['day_num']
    time_set = np.arange(24*day_num)

    if 'profile_id' in settings:
        profile_id = settings['profile_id']
        renewable = [wind_all[time_set, profile_id],pv_all[time_set, profile_id]]
    else:
        # Denote that first profile is used and no other profile exists.
        profile_id = -1
        renewable = [wind_all[time_set], pv_all[time_set]]
    load = load_all[time_set]*100
    
    nrenewable = len(renewable)
    settings['nrenewable'] = nrenewable
    
    period = settings['period']
    periods = settings['day_num']*24//period
    settings['periods'] = periods 

    period_renewable = [renewable[r].reshape(
        periods, period) for r in range(nrenewable)]
    period_load = load_all[time_set].reshape(periods, period)
    period_data = np.hstack(period_renewable+[period_load])
    period_df = pd.DataFrame(period_data)
    settings['period_df'] = period_df

    results = [expected]
    
    data = {'hourly': period_df, 'simulated': features,'combined': period_df.join(features, how='left')}
    
    i = 0
    nclusters = ranges['nclusters']
    trials = ranges['trials']
    methods = ranges['methods']
    
    if settings['normalize']:
        for trial in trials:
            for f in data[trial].columns:
                f_max = max(data[trial].loc[:, f])
                if f_max != 0:
                    for w in range(periods):
                        data[trial].loc[w, f] /= f_max
    
    for ncluster in nclusters:
        for trial in trials:
            for method in methods:
                i += 1
                data_c = data[trial].copy()
                temp = settings.copy()
                
                temp['ncluster'] = ncluster
                temp['trial'] = trial
                temp['method'] = method
                if method == 'kmeans':
                    temp['init'] = 0
                
                start = time.time()
                result = test_clustering(inputs,temp,expected,data_c)
                result['elapsed_time'] = time.time() - start
                result['description'] = 'clustered'
                result['ncluster'] = ncluster
                result['trial'] = trial
                result['method'] = method
                print('{}th trial: '.format(i),result['elapsed_time'])
                
                results.append(result)
        
    df = pd.DataFrame(results)
    return df