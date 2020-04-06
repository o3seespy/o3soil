import numpy as np
import sfsimodels as sm
import openseespy.opensees as opy
import o3seespy as o3
import o3seespy.extensions
import copy
import os


def site_response(sp, asig, freqs=(0.5, 10), xi=0.03, analysis_dt=0.001, dy=0.5, analysis_time=None, outs=None,
                  rec_dt=None, fixed_base=0, cache_path=None, pload=0.0):
    """
    Run seismic analysis of a soil profile - example based on:
    http://opensees.berkeley.edu/wiki/index.php/Site_Response_Analysis_of_a_Layered_Soil_Column_(Total_Stress_Analysis)

    Parameters
    ----------
    sp: sfsimodels.SoilProfile object
        A soil profile
    asig: eqsig.AccSignal object
        An acceleration signal

    Returns
    -------

    """
    if analysis_time is None:
        analysis_time = asig.time[-1]
    if outs is None:
        outs = {'ACCX': [0]}  # Export the horizontal acceleration at the surface
    if rec_dt is None:
        rec_dt = analysis_dt
    else:
        raise ValueError('This is causing an error')

    osi = o3.OpenSeesInstance(ndm=2, ndf=2, state=3)
    assert isinstance(sp, sm.SoilProfile)
    sp.gen_split(props=['shear_vel', 'unit_mass'], target=dy)
    thicknesses = sp.split["thickness"]
    n_node_rows = len(thicknesses) + 1
    node_depths = np.cumsum(sp.split["thickness"])
    node_depths = np.insert(node_depths, 0, 0)
    ele_depths = (node_depths[1:] + node_depths[:-1]) / 2
    unit_masses = sp.split["unit_mass"] / 1e3

    grav = 9.81
    omega_1 = 2 * np.pi * freqs[0]
    omega_2 = 2 * np.pi * freqs[1]
    a0 = 2 * xi * omega_1 * omega_2 / (omega_1 + omega_2)
    a1 = 2 * xi / (omega_1 + omega_2)

    k0 = 0.5
    pois = k0 / (1 + k0)

    newmark_gamma = 0.5
    newmark_beta = 0.25

    ele_width = 3 * min(thicknesses)

    # Define nodes and set boundary conditions for simple shear deformation
    # Start at top and build down?
    nx = 1
    sn = []
    # sn = [[o3.node.Node(osi, ele_width * j, 0) for j in range(nx + 1)]]
    for i in range(0, n_node_rows):
        # Establish left and right nodes
        sn.append([o3.node.Node(osi, ele_width * j, -node_depths[i]) for j in range(nx + 1)])
        # set x and y dofs equal for left and right nodes
        o3.EqualDOF(osi, sn[i][0], sn[i][-1], [o3.cc.X, o3.cc.Y])
    sn = np.array(sn)

    if fixed_base:
        # Fix base nodes
        for j in range(nx + 1):
            o3.Fix2DOF(osi, sn[-1][j], o3.cc.FIXED, o3.cc.FIXED)
    else:
        # Fix base nodes
        for j in range(nx + 1):
            o3.Fix2DOF(osi, sn[-1][j], o3.cc.FREE, o3.cc.FIXED)

        # Define dashpot nodes
        dashpot_node_l = o3.node.Node(osi, 0, -node_depths[-1])
        dashpot_node_2 = o3.node.Node(osi, 0, -node_depths[-1])
        o3.Fix2DOF(osi, dashpot_node_l, o3.cc.FIXED, o3.cc.FIXED)
        o3.Fix2DOF(osi, dashpot_node_2, o3.cc.FREE, o3.cc.FIXED)

        # define equal DOF for dashpot and soil base nodes
        o3.EqualDOF(osi, sn[-1][0], sn[-1][1], [o3.cc.X])
        o3.EqualDOF(osi, sn[-1][0], dashpot_node_2, [o3.cc.X])

    # define materials
    ele_thick = 1.0  # m
    soil_mats = []
    strains = np.logspace(-6, -0.5, 16)
    prev_args = []
    prev_kwargs = {}
    prev_sl_type = None
    eles = []
    for i in range(len(thicknesses)):
        y_depth = ele_depths[i]

        sl_id = sp.get_layer_index_by_depth(y_depth)
        sl = sp.layer(sl_id)

        app2mod = {}
        if y_depth > sp.gwl:
            umass = sl.unit_sat_mass / 1e3  # TODO: work out how to run in Pa, N, m, s
        else:
            umass = sl.unit_dry_mass / 1e3
        overrides = {'nu': pois, 'p_atm': 101,
                     'rho': umass,
                     'unit_moist_mass': umass,
                     'nd': 2.0,
                     # 'n_surf': 25
                     }
        # Define material
        if sl.type == 'pm4sand':
            sl_class = o3.nd_material.PM4Sand
            # overrides = {'nu': pois, 'p_atm': 101, 'unit_moist_mass': umass}
            app2mod = sl.app2mod
        elif sl.type == 'sdmodel':
            sl_class = o3.nd_material.StressDensity
            # overrides = {'nu': pois, 'p_atm': 101, 'unit_moist_mass': umass}
            app2mod = sl.app2mod
        elif sl.type in ['pimy', 'pdmy', 'pdmy02']:
            overrides['p_ref'] = sl.p_ref / 1e3
            overrides['g_mod_ref'] = sl.g_mod_ref / 1e3
            overrides['bulk_mod_ref'] = sl.bulk_mod_ref / 1e3
            overrides['d'] = sl.a
            if sl.type == 'pimy':
                overrides['cohesion'] = sl.cohesion / 1e3
                sl_class = o3.nd_material.PressureIndependMultiYield
            elif sl.type == 'pdmy':
                sl_class = o3.nd_material.PressureDependMultiYield
            elif sl.type == 'pdmy02':
                sl_class = o3.nd_material.PressureDependMultiYield02
        else:
            sl_class = o3.nd_material.ElasticIsotropic
            sl.e_mod = 2 * sl.g_mod * (1 - sl.poissons_ratio) / 1e3
            app2mod['rho'] = 'unit_moist_mass'
            overrides = {'nu': sl.poissons_ratio, 'unit_moist_mass': umass}

        args, kwargs = o3.extensions.get_o3_kwargs_from_obj(sl, sl_class, custom=app2mod, overrides=overrides)

        if o3.extensions.has_o3_model_changed(sl.type, prev_sl_type, args, prev_args, kwargs, prev_kwargs):
            mat = sl_class(osi, *args, **kwargs)
            prev_sl_type = sl.type
            prev_args = copy.deepcopy(args)
            prev_kwargs = copy.deepcopy(kwargs)

            soil_mats.append(mat)

        # def element
        for xx in range(nx):
            nodes = [sn[i+1][xx], sn[i+1][xx + 1], sn[i][xx + 1], sn[i][xx]]  # anti-clockwise
            # ele = o3.element.Quad(osi, nodes, ele_thick, o3.cc.PLANE_STRAIN, mat, b2=grav * unit_masses[i])
            # eles.append(ele)
            eles.append(o3.element.SSPquad(osi, nodes, mat, o3.cc.PLANE_STRAIN, ele_thick, 0.0, grav * unit_masses[i]))

    if not fixed_base:
        # define material and element for viscous dampers
        base_sl = sp.layer(sp.n_layers)
        c_base = ele_width * base_sl.unit_dry_mass / 1e3 * sp.get_shear_vel_at_depth(sp.height)
        dashpot_mat = o3.uniaxial_material.Viscous(osi, c_base, alpha=1.)
        o3.element.ZeroLength(osi, [dashpot_node_l, dashpot_node_2], mats=[dashpot_mat], dirs=[o3.cc.DOF2D_X])

    # Static analysis
    o3.constraints.Transformation(osi)
    o3.test.NormDispIncr(osi, tol=1.0e-6, max_iter=20, p_flag=0)
    # o3.test_check.EnergyIncr(osi, tol=1.0e-5, max_iter=20, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.ProfileSPD(osi)
    o3.integrator.Newmark(osi, newmark_gamma, newmark_beta)
    o3.analysis.Transient(osi)
    o3.analyze(osi, 40, 1.)

    for i in range(len(soil_mats)):
        if hasattr(soil_mats[i], 'update_to_nonlinear'):
            soil_mats[i].update_to_nonlinear(osi)
    o3.analyze(osi, 50, 0.5)

    o3.rayleigh.Rayleigh(osi, a0, a1, 0, 0)

    # reset time and analysis
    o3.set_time(osi, 0.0)  # TODO:
    o3.wipe_analysis(osi)

    coords = o3.get_all_node_coords(osi)
    ele_node_tags = o3.get_all_ele_node_tags_as_dict(osi)
    all_node_xdisp_rec = o3.recorder.NodesToArrayCache(osi, 'all', [o3.cc.DOF2D_X], 'disp', nsd=4)
    all_node_ydisp_rec = o3.recorder.NodesToArrayCache(osi, 'all', [o3.cc.DOF2D_Y], 'disp', nsd=4)

    # Define the dynamic analysis
    print('Here')
    o3.constraints.Transformation(osi)
    o3.test.NormDispIncr(osi, tol=1.0e-3, max_iter=15, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.system.SparseGeneral(osi)
    o3.numberer.RCM(osi)
    o3.integrator.Newmark(osi, newmark_gamma, newmark_beta)
    o3.analysis.Transient(osi)

    if pload:
        static_time = 100
        print('time: ', o3.get_time(osi))
        # Add static stress bias
        time_series = o3.time_series.Path(osi, time=[0, static_time / 2, static_time, 1e3], values=[0, 0.5, 1, 1], use_last=1)
        o3.pattern.Plain(osi, time_series)
        o3.Load(osi, sn[0][0], [pload * ele_width, 0])
        o3.Load(osi, sn[9][0], [-pload * ele_width, 0])
        if not fixed_base:
            o3.Load(osi, sn[-1][0], [-pload, 0])

        static_dt = 0.1
        o3.analyze(osi, int(static_time / static_dt) * 1.5, static_dt)
        o3.load_constant(osi, time=0)

    o3.set_time(osi, 0.0)  # TODO:
    init_time = o3.get_time(osi)
    o3sra_outs = O3SRAOutputs()
    o3sra_outs.start_recorders(osi, outs, sn, eles, rec_dt=rec_dt)

    # Define the dynamic input motion
    if fixed_base:
        acc_series = o3.time_series.Path(osi, dt=asig.dt, values=-asig.values)  # should be negative
        o3.pattern.UniformExcitation(osi, dir=o3.cc.X, accel_series=acc_series)
    else:
        ts_obj = o3.time_series.Path(osi, dt=asig.dt, values=asig.velocity * -1, factor=c_base)
        o3.pattern.Plain(osi, ts_obj)
        o3.Load(osi, sn[-1][0], [1., 0.])

    # Run the dynamic motion
    while o3.get_time(osi) - init_time < analysis_time:
        if o3.analyze(osi, 1, analysis_dt):
            print('failed')
            break
    o3.wipe(osi)
    out_dict = o3sra_outs.results_to_dict()

    if cache_path:
        import o3_plot
        o3sra_outs.cache_path = cache_path
        o3sra_outs.results_to_files()
        o3res = o3_plot.O3Results()
        o3res.cache_path = cache_path
        o3res.coords = coords
        o3res.ele_node_tags = ele_node_tags
        o3res.x_disp = all_node_xdisp_rec.collect()
        o3res.y_disp = all_node_ydisp_rec.collect()
        o3res.save_to_cache()

    return out_dict



class O3SRAOutputs(object):
    cache_path = ''
    out_dict = None
    area = 1.0
    outs = None

    def start_recorders(self, osi, outs, sn, eles, rec_dt, sn_xy=False):
        self.rec_dt = rec_dt
        self.eles = eles
        self.sn_xy = sn_xy
        if sn_xy:
            self.nodes = sn[0, :]
        else:
            self.nodes = sn[:, 0]
        self.outs = outs
        node_depths = np.array([node.y for node in sn[:, 0]])
        ele_depths = (node_depths[1:] + node_depths[:-1]) / 2
        ods = {}
        for otype in outs:
            if otype in ['ACCX', 'DISPX']:
                if isinstance(outs[otype], str) and outs[otype] == 'all':

                    if otype == 'ACCX':
                        ods['ACCX'] = o3.recorder.NodesToArrayCache(osi, nodes=self.nodes, dofs=[o3.cc.X], res_type='accel',
                                                                dt=rec_dt)
                    if otype == 'DISPX':
                        ods['DISPX'] = o3.recorder.NodesToArrayCache(osi, nodes=self.nodes, dofs=[o3.cc.X], res_type='disp',
                                                                dt=rec_dt)
                else:
                    ods['ACCX'] = []
                    for i in range(len(outs['ACCX'])):
                        ind = np.argmin(abs(node_depths - outs['ACCX'][i]))
                        ods['ACCX'].append(
                            o3.recorder.NodeToArrayCache(osi, node=sn[ind][0], dofs=[o3.cc.X], res_type='accel', dt=rec_dt))
            if otype == 'TAU':
                for ele in eles:
                    assert isinstance(ele, o3.element.SSPquad)
                ods['TAU'] = []
                if isinstance(outs['TAU'], str) and outs['TAU'] == 'all':
                    ods['TAU'] = o3.recorder.ElementsToArrayCache(osi, eles=eles, arg_vals=['stress'], dt=rec_dt)
                else:
                    for i in range(len(outs['TAU'])):
                        ind = np.argmin(abs(ele_depths - outs['TAU'][i]))
                        ods['TAU'].append(
                            o3.recorder.ElementToArrayCache(osi, ele=eles[ind], arg_vals=['stress'], dt=rec_dt))
            if otype == 'TAUX':
                if isinstance(outs['TAUX'], str) and outs['TAUX'] == 'all':
                    if sn_xy:
                        order = 'F'
                    else:
                        order = 'C'
                    ods['TAUX'] = o3.recorder.NodesToArrayCache(osi, nodes=sn.flatten(order), dofs=[o3.cc.X], res_type='reaction',
                                                                dt=rec_dt)
            if otype == 'STRS':
                ods['STRS'] = []
                if isinstance(outs['STRS'], str) and outs['STRS'] == 'all':
                    ods['STRS'] = o3.recorder.ElementsToArrayCache(osi, eles=eles, arg_vals=['strain'], dt=rec_dt)
                else:
                    for i in range(len(outs['STRS'])):
                        ind = np.argmin(abs(ele_depths - outs['STRS'][i]))
                        ods['STRS'].append(o3.recorder.ElementToArrayCache(osi, ele=eles[ind], arg_vals=['strain'], dt=rec_dt))
            if otype == 'STRSX':
                if isinstance(outs['STRSX'], str) and outs['STRSX'] == 'all':
                    if 'DISPX' in outs:
                        continue
                    if sn_xy:
                        nodes = sn[0, :]
                    else:
                        nodes = sn[:, 0]
                    ods['DISPX'] = o3.recorder.NodesToArrayCache(osi, nodes=nodes, dofs=[o3.cc.X], res_type='disp',
                                                                dt=rec_dt)

        self.ods = ods

    def results_to_files(self):
        od = self.results_to_dict()
        for item in od:
            ffp = self.cache_path + f'{item}.txt'
            if os.path.exists(ffp):
                os.remove(ffp)
            np.savetxt(ffp, od[item])

    def load_results_from_files(self, outs=None):
        if outs is None:
            outs = ['ACCX', 'TAU', 'STRS', 'time']
        od = {}
        for item in outs:
            od[item] = np.loadtxt(self.cache_path + f'{item}.txt')
        return od

    def results_to_dict(self):
        ro = o3.recorder.load_recorder_options()
        import pandas as pd
        df = pd.read_csv(ro)
        if self.outs is None:
            items = list(self.ods)
        else:
            items = list(self.outs)
        if self.out_dict is None:
            self.out_dict = {}
            for otype in items:
                if otype not in self.ods:
                    if otype == 'STRSX':
                        depths = []
                        for node in self.nodes:
                            depths.append(node.y)
                        depths = np.array(depths)
                        d_incs = depths[1:] - depths[:-1]
                        vals = self.ods['DISPX'].collect(unlink=False).T
                        self.out_dict[otype] = (vals[1:] - vals[:-1]) / d_incs[:, np.newaxis]
                elif isinstance(self.ods[otype], list):
                    self.out_dict[otype] = []
                    for i in range(len(self.ods[otype])):
                        if otype in ['TAU', 'STRS']:
                            self.out_dict[otype].append(self.ods[otype][i].collect()[2])
                        else:
                            self.out_dict[otype].append(self.ods[otype][i].collect())
                    self.out_dict[otype] = np.array(self.out_dict[otype])
                else:
                    vals = self.ods[otype].collect().T
                    cur_ind = 0
                    self.out_dict[otype] = []
                    if otype in ['TAU', 'STRS']:
                        for ele in self.eles:
                            mat_type = ele.mat.type
                            form = 'PlaneStrain'
                            dfe = df[(df['mat'] == mat_type) & (df['form'] == form)]
                            if otype == 'TAU':
                                dfe = dfe[dfe['recorder'] == 'stress']
                                ostr = 'sxy'
                            else:
                                dfe = dfe[dfe['recorder'] == 'strain']
                                ostr = 'gxy'
                            assert len(dfe) == 1
                            outs = dfe['outs'].iloc[0].split('-')
                            oind = outs.index(ostr)
                            self.out_dict[otype].append(vals[cur_ind + oind])
                            cur_ind += len(outs)
                        self.out_dict[otype] = np.array(self.out_dict[otype])
                        # if otype == 'STRS':
                        #     self.out_dict[otype] = vals[2::3]  # Assumes pimy
                        # elif otype == 'TAU':
                        #     self.out_dict[otype] = vals[3::5]  # Assumes pimy
                    elif otype == 'TAUX':
                        f_static = -np.cumsum(vals[::2, :] - vals[1::2, :], axis=0)[:-1]  # add left and right
                        f_dyn = vals[::2, :] + vals[1::2, :]  # add left and right
                        f_dyn_av = (f_dyn[1:] + f_dyn[:-1]) / 2
                        # self.out_dict[otype] = (f[1:, :] - f[:-1, :]) / area
                        self.out_dict[otype] = (f_dyn_av + f_static) / self.area
                    else:
                        self.out_dict[otype] = vals
            # Create time output
            if 'ACCX' in self.out_dict:
                self.out_dict['time'] = np.arange(0, len(self.out_dict['ACCX'][0])) * self.rec_dt
            elif 'TAU' in self.out_dict:
                self.out_dict['time'] = np.arange(0, len(self.out_dict['TAU'][0])) * self.rec_dt
        return self.out_dict


def site_response_w_pysra(soil_profile, asig, odepths):
    import liquepy as lq
    import pysra
    pysra_profile = lq.sra.sm_profile_to_pysra(soil_profile, d_inc=[0.5] * soil_profile.n_layers)
    # Should be input in g
    pysra_m = pysra.motion.TimeSeriesMotion(asig.label, None, time_step=asig.dt, accels=-asig.values / 9.8)

    calc = pysra.propagation.EquivalentLinearCalculator()

    od = {'ACCX': [], 'STRS': [], 'TAU': []}
    outs = []
    for i, depth in enumerate(odepths):
        od['ACCX'].append(len(outs))
        outs.append(pysra.output.AccelerationTSOutput(pysra.output.OutputLocation('within', depth=depth)))
        od['STRS'].append(len(outs))
        outs.append(pysra.output.StrainTSOutput(pysra.output.OutputLocation('within', depth=depth), in_percent=False))
        od['TAU'].append(len(outs))
        outs.append(pysra.output.StressTSOutput(pysra.output.OutputLocation('within', depth=depth),
                                                normalized=False))
    outputs = pysra.output.OutputCollection(outs)
    calc(pysra_m, pysra_profile, pysra_profile.location('outcrop', depth=soil_profile.height))
    outputs(calc)

    out_series = {}
    for mtype in od:
        out_series[mtype] = []
        for i in range(len(od[mtype])):
            out_series[mtype].append(outputs[od[mtype][i]].values[:asig.npts])
        out_series[mtype] = np.array(out_series[mtype])
        if mtype == 'ACCX':
            out_series[mtype] *= 9.8
    return out_series
