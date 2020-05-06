import numpy as np
import sfsimodels as sm
import openseespy.opensees as opy
import o3seespy as o3
import o3seespy.extensions
import copy
import os


class SRA1D(object):
    osi = None

    def __init__(self, sp, dy=0.5, k0=0.5, base_imp=0, cache_path=None, opfile=None):
        self.sp = sp
        sp.gen_split(props=['shear_vel', 'unit_mass'], target=dy)
        thicknesses = sp.split["thickness"]
        self.n_node_rows = len(thicknesses) + 1
        node_depths = np.cumsum(sp.split["thickness"])
        self.node_depths = np.insert(node_depths, 0, 0)
        self.ele_depths = (self.node_depths[1:] + self.node_depths[:-1]) / 2
        self.unit_masses = sp.split["unit_mass"] / 1e3

        self.grav = 9.81

        self.k0 = k0
        self.base_imp = base_imp

        self.ele_width = 3 * min(thicknesses)
        self.cache_path = cache_path
        self.opfile = opfile
        # Defined in static analysis
        self.soil_mats = None
        self.eles = None
        self.sn = None  # soil nodes

    def build_model(self):
        # Define nodes and set boundary conditions for simple shear deformation
        # Start at top and build down?
        if self.opfile:
            self.state = 3
        else:
            self.state = 0
        if self.osi is None:
            self.osi = o3.OpenSeesInstance(ndm=2, ndf=2, state=self.state)
        nx = 1
        sn = []
        # sn = [[o3.node.Node(osi, ele_width * j, 0) for j in range(nx + 1)]]
        for i in range(0, self.n_node_rows):
            # Establish left and right nodes
            sn.append([o3.node.Node(self.osi, self.ele_width * j, -self.node_depths[i]) for j in range(nx + 1)])
            # set x and y dofs equal for left and right nodes
            if i != self.n_node_rows - 1:
                o3.EqualDOF(self.osi, sn[i][0], sn[i][-1], [o3.cc.X, o3.cc.Y])
        sn = np.array(sn)

        if self.base_imp < 0:
            # Fix base nodes
            for j in range(nx + 1):
                o3.Fix2DOF(self.osi, sn[-1][j], o3.cc.FIXED, o3.cc.FIXED)
        else:
            # Fix base nodes
            for j in range(nx + 1):
                o3.Fix2DOF(self.osi, sn[-1][j], o3.cc.FREE, o3.cc.FIXED)

            # Define dashpot nodes
            dashpot_node_l = o3.node.Node(self.osi, 0, -self.node_depths[-1])
            dashpot_node_2 = o3.node.Node(self.osi, 0, -self.node_depths[-1])
            o3.Fix2DOF(self.osi, dashpot_node_l, o3.cc.FIXED, o3.cc.FIXED)
            o3.Fix2DOF(self.osi, dashpot_node_2, o3.cc.FREE, o3.cc.FIXED)

            # define equal DOF for dashpot and soil base nodes
            o3.EqualDOF(self.osi, sn[-1][0], sn[-1][1], [o3.cc.X])
            o3.EqualDOF(self.osi, sn[-1][0], dashpot_node_2, [o3.cc.X])

        # define materials
        pois = self.k0 / (1 + self.k0)
        ele_thick = 1.0  # m
        self.soil_mats = []
        strains = np.logspace(-6, -0.5, 16)
        prev_args = []
        prev_kwargs = {}
        prev_sl_class = None
        self.eles = []
        for i in range(len(self.ele_depths)):
            y_depth = self.ele_depths[i]

            sl_id = self.sp.get_layer_index_by_depth(y_depth)
            sl = self.sp.layer(sl_id)

            app2mod = {}
            if y_depth > self.sp.gwl:
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
                if hasattr(sl, 'get_g_mod_at_m_eff_stress'):
                    if hasattr(sl, 'g_mod_p0') and sl.g_mod_p0 != 0.0:
                        v_eff = self.sp.get_v_eff_stress_at_depth(y_depth)
                        k0 = sl.poissons_ratio / (1 - sl.poissons_ratio)
                        m_eff = v_eff * (1 + 2 * k0) / 3
                        p = m_eff  # Pa
                        overrides['d'] = 0.0
                    else:
                        p = 101.0e3  # Pa
                        overrides['d'] = sl.a
                    g_mod_r = sl.get_g_mod_at_m_eff_stress(p) / 1e3
                else:
                    p = 101.0e3  # Pa
                    overrides['d'] = 0.0
                    g_mod_r = sl.g_mod / 1e3

                b_mod = 2 * g_mod_r * (1 + sl.poissons_ratio) / (3 * (1 - 2 * sl.poissons_ratio))
                overrides['p_ref'] = p / 1e3
                overrides['g_mod_ref'] = g_mod_r
                overrides['bulk_mod_ref'] = b_mod
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
                overrides['nu'] = sl.poissons_ratio
                app2mod['rho'] = 'unit_moist_mass'
            args, kwargs = o3.extensions.get_o3_kwargs_from_obj(sl, sl_class, custom=app2mod, overrides=overrides)

            if o3.extensions.has_o3_model_changed(sl_class, prev_sl_class, args, prev_args, kwargs, prev_kwargs):
                mat = sl_class(self.osi, *args, **kwargs)
                prev_sl_class = sl_class
                prev_args = copy.deepcopy(args)
                prev_kwargs = copy.deepcopy(kwargs)
                mat.dynamic_poissons_ratio = sl.poissons_ratio
                self.soil_mats.append(mat)

            # def element
            for xx in range(nx):
                nodes = [sn[i + 1][xx], sn[i + 1][xx + 1], sn[i][xx + 1], sn[i][xx]]  # anti-clockwise
                # eles.append(o3.element.Quad(self.osi, nodes, ele_thick, o3.cc.PLANE_STRAIN, mat, b2=-grav * unit_masses[i]))
                self.eles.append(o3.element.SSPquad(self.osi, nodes, mat, o3.cc.PLANE_STRAIN, ele_thick, 0.0,
                                                    -self.grav * self.unit_masses[i]))
        self.sn = sn
        if self.base_imp >= 0:
            # define material and element for viscous dampers
            if self.base_imp == 0:
                sl = self.sp.get_soil_at_depth(self.sp.height)
                base_imp = sl.unit_dry_mass * self.sp.get_shear_vel_at_depth(self.sp.height)
            c_base = self.ele_width * base_imp / 1e3
            dashpot_mat = o3.uniaxial_material.Viscous(self.osi, c_base, alpha=1.)
            o3.element.ZeroLength(self.osi, [dashpot_node_l, dashpot_node_2], mats=[dashpot_mat], dirs=[o3.cc.DOF2D_X])
        import o3_plot
        self.o3res = o3.results.Results2D()
        self.o3res.cache_path = self.cache_path
        node_coords = o3.get_all_node_coords(self.osi)
        self.o3res.coords = node_coords
        self.o3res.ele2node_tags = o3.get_all_ele_node_tags_as_dict(self.osi)

    def execute_static(self):
        # Static analysis
        o3.constraints.Transformation(self.osi)
        o3.test.NormDispIncr(self.osi, tol=1.0e-5, max_iter=30, p_flag=0)
        o3.algorithm.Newton(self.osi)
        o3.numberer.RCM(self.osi)
        o3.system.ProfileSPD(self.osi)
        o3.integrator.Newmark(self.osi, gamma=0.5, beta=0.25)
        o3.analysis.Transient(self.osi)
        o3.analyze(self.osi, 10, 500.)

        for i in range(len(self.soil_mats)):
            if hasattr(self.soil_mats[i], 'update_to_nonlinear'):
                self.soil_mats[i].update_to_nonlinear(self.osi)
        for ele in self.eles:
            mat = ele.mat
            if hasattr(mat, 'set_nu'):
                mat.set_nu(mat.dynamic_poissons_ratio, eles=[ele])
        o3.analyze(self.osi, 40, 500.)

        # reset time and analysis
        o3.wipe_analysis(self.osi)
        self.o3res.coords = o3.get_all_node_coords(self.osi)

    def get_nearest_node_at_depth(self, depth):
        return int(np.round(np.interp(depth, self.node_depths, np.arange(len(self.node_depths)))))

    def apply_loads(self):
        o3.set_time(self.osi, 0.0)

        # Define the dynamic analysis
        o3.constraints.Transformation(self.osi)
        o3.test.NormDispIncr(self.osi, tol=1.0e-4, max_iter=30, p_flag=0)
        # o3.test_check.EnergyIncr(self.osi, tol=1.0e-6, max_iter=30)
        o3.algorithm.Newton(self.osi)
        o3.system.SparseGeneral(self.osi)
        o3.numberer.RCM(self.osi)
        o3.integrator.Newmark(self.osi, gamma=0.5, beta=0.25)
        o3.analysis.Transient(self.osi)
        # o3.rayleigh.Rayleigh(self.osi, self.a0, self.a1, 0, 0)

        static_time = 100
        print('time: ', o3.get_time(self.osi))
        # Add static stress bias
        time_series = o3.time_series.Path(self.osi, time=[0, static_time / 2, static_time, 1e3], values=[0, 0.5, 1, 1],
                                          use_last=True)
        o3.pattern.Plain(self.osi, time_series)
        net_hload = 0
        for i in range(len(self.sp.hloads)):
            pload = self.sp.hloads[i].p_x
            y = -self.sp.hloads[i].y
            ind = self.get_nearest_node_at_depth(y)
            if self.sp.loads_are_stresses:
                pload *= self.ele_width
            o3.Load(self.osi, self.sn[ind][0], [pload, 0])
            net_hload += pload
        if self.base_imp >= 0:
            o3.Load(self.osi, self.sn[-1][0], [-net_hload, 0])

        static_dt = 0.1
        o3.analyze(self.osi, int(static_time / static_dt) * 1.5, static_dt)
        o3.load_constant(self.osi, time=0)

    def execute_dynamic(self, asig, analysis_dt=0.001, ray_freqs=(0.5, 10), xi=0.03, analysis_time=None,
                        outs=None, rec_dt=None, playback=True):
        self.rec_dt = rec_dt
        if rec_dt is None:
            self.rec_dt = asig.dt
        o3.set_time(self.osi, 0.0)

        # Define the dynamic analysis
        o3.constraints.Transformation(self.osi)
        o3.test.NormDispIncr(self.osi, tol=1.0e-4, max_iter=30, p_flag=0)
        # o3.test_check.EnergyIncr(self.osi, tol=1.0e-6, max_iter=30)
        o3.algorithm.Newton(self.osi)
        o3.system.SparseGeneral(self.osi)
        o3.numberer.RCM(self.osi)
        o3.integrator.Newmark(self.osi, gamma=0.5, beta=0.25)
        o3.analysis.Transient(self.osi)
        # Rayleigh damping parameters
        omega_1 = 2 * np.pi * ray_freqs[0]
        omega_2 = 2 * np.pi * ray_freqs[1]
        a0 = 2 * xi * omega_1 * omega_2 / (omega_1 + omega_2)
        a1 = 2 * xi / (omega_1 + omega_2)
        o3.rayleigh.Rayleigh(self.osi, a0, a1, 0, 0)

        init_time = o3.get_time(self.osi)
        if playback:
            all_node_xdisp_rec = o3.recorder.NodesToArrayCache(self.osi, 'all', [o3.cc.DOF2D_X], 'disp', nsd=4)
            all_node_ydisp_rec = o3.recorder.NodesToArrayCache(self.osi, 'all', [o3.cc.DOF2D_Y], 'disp', nsd=4)
        self.o3sra_outs = O3SRAOutputs()
        self.o3sra_outs.start_recorders(self.osi, outs, self.sn, self.eles, rec_dt=self.rec_dt)

        # Define the dynamic input motion
        if self.base_imp < 0:  # fixed base
            acc_series = o3.time_series.Path(self.osi, dt=asig.dt, values=asig.values)
            o3.pattern.UniformExcitation(self.osi, dir=o3.cc.X, accel_series=acc_series)
        else:
            ts_obj = o3.time_series.Path(self.osi, dt=asig.dt, values=asig.velocity * 1, factor=self.c_base)
            o3.pattern.Plain(self.osi, ts_obj)
            o3.Load(self.osi, self.sn[-1][0], [1., 0.])
        if self.state == 3:
            o3.extensions.to_py_file(self.osi, self.opfile)
        # Run the dynamic motion
        while o3.get_time(self.osi) - init_time < analysis_time:
            if o3.analyze(self.osi, 1, analysis_dt):
                print('failed')
                break
        o3.wipe(self.osi)
        self.out_dict = self.o3sra_outs.results_to_dict()

        if playback:
            self.o3res.x_disp = all_node_xdisp_rec.collect()
            self.o3res.y_disp = all_node_ydisp_rec.collect()
        else:
            self.o3res.dynamic = False

        if self.cache_path:
            import o3_plot
            self.o3sra_outs.cache_path = self.cache_path
            self.o3sra_outs.results_to_files()
            self.o3res.save_to_cache()


def run_sra(sp, asig, ray_freqs=(0.5, 10), xi=0.03, analysis_dt=0.001, dy=0.5, analysis_time=None, outs=None,
                  base_imp=0, k0=0.5, cache_path=None, opfile=None, playback=False):
    sra_1d = SRA1D(sp, dy=dy, k0=k0, base_imp=base_imp, cache_path=cache_path, opfile=opfile)
    sra_1d.build_model()
    sra_1d.execute_static()
    if hasattr(sra_1d.sp, 'hloads'):
        sra_1d.apply_loads()
    sra_1d.execute_dynamic(asig, analysis_dt=analysis_dt, ray_freqs=ray_freqs, xi=xi, analysis_time=analysis_time,
                           outs=outs, playback=playback)
    return sra_1d



def site_response(sp, asig, freqs=(0.5, 10), xi=0.03, analysis_dt=0.001, dy=0.5, analysis_time=None, outs=None,
                  rec_dt=None, base_imp=0, cache_path=None, opfile=None):
    """
    Run seismic analysis of a soil profile - example based on:
    http://opensees.berkeley.edu/wiki/index.php/Site_Response_Analysis_of_a_Layered_Soil_Column_(Total_Stress_Analysis)

    Parameters
    ----------
    sp: sfsimodels.SoilProfile object
        A soil profile
    asig: eqsig.AccSignal object
        An acceleration signal
    base_imp: float
        If positive then use as impedence at base of model,
        If zero then use last soil layer
        If negative then use fixed base

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
    state = 0
    if opfile:
        state = 3
    osi = o3.OpenSeesInstance(ndm=2, ndf=2, state=state)
    assert isinstance(sp, sm.SoilProfile)
    sp.gen_split(props=['shear_vel', 'unit_mass'], target=dy)
    thicknesses = sp.split["thickness"]
    n_node_rows = len(thicknesses) + 1
    node_depths = np.cumsum(sp.split["thickness"])
    node_depths = np.insert(node_depths, 0, 0)
    ele_depths = (node_depths[1:] + node_depths[:-1]) / 2
    unit_masses = sp.split["unit_mass"] / 1e3

    grav = 9.81
    # Rayleigh damping parameters
    omega_1 = 2 * np.pi * freqs[0]
    omega_2 = 2 * np.pi * freqs[1]
    a0 = 2 * xi * omega_1 * omega_2 / (omega_1 + omega_2)
    a1 = 2 * xi / (omega_1 + omega_2)

    k0 = 0.5
    pois = k0 / (1 + k0)

    ele_width = 3 * min(thicknesses)

    # Define nodes and set boundary conditions for simple shear deformation
    # Start at top and build down?
    nx = 1
    sn = []
    # sn = [[o3.node.Node(self.osi, ele_width * j, 0) for j in range(nx + 1)]]
    for i in range(0, n_node_rows):
        # Establish left and right nodes
        sn.append([o3.node.Node(osi, ele_width * j, -node_depths[i]) for j in range(nx + 1)])
        # set x and y dofs equal for left and right nodes
        if i != n_node_rows -1:
            o3.EqualDOF(osi, sn[i][0], sn[i][-1], [o3.cc.X, o3.cc.Y])
    sn = np.array(sn)

    if base_imp < 0:
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
    prev_sl_class = None
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
            if hasattr(sl, 'get_g_mod_at_m_eff_stress'):
                if hasattr(sl, 'g_mod_p0') and sl.g_mod_p0 != 0.0:
                    v_eff = sp.get_v_eff_stress_at_depth(y_depth)
                    k0 = sl.poissons_ratio / (1 - sl.poissons_ratio)
                    m_eff = v_eff * (1 + 2 * k0) / 3
                    p = m_eff  # Pa
                    overrides['d'] = 0.0
                else:
                    p = 101.0e3  # Pa
                    overrides['d'] = sl.a
                g_mod_r = sl.get_g_mod_at_m_eff_stress(p) / 1e3
            else:
                p = 101.0e3  # Pa
                overrides['d'] = 0.0
                g_mod_r = sl.g_mod / 1e3

            b_mod = 2 * g_mod_r * (1 + sl.poissons_ratio) / (3 * (1 - 2 * sl.poissons_ratio))
            overrides['p_ref'] = p / 1e3
            overrides['g_mod_ref'] = g_mod_r
            overrides['bulk_mod_ref'] = b_mod
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
            overrides['nu'] = sl.poissons_ratio
            app2mod['rho'] = 'unit_moist_mass'
        args, kwargs = o3.extensions.get_o3_kwargs_from_obj(sl, sl_class, custom=app2mod, overrides=overrides)

        if o3.extensions.has_o3_model_changed(sl_class, prev_sl_class, args, prev_args, kwargs, prev_kwargs):
            mat = sl_class(osi, *args, **kwargs)
            prev_sl_class = sl_class
            prev_args = copy.deepcopy(args)
            prev_kwargs = copy.deepcopy(kwargs)
            mat.dynamic_poissons_ratio = sl.poissons_ratio
            soil_mats.append(mat)

        # def element
        for xx in range(nx):
            nodes = [sn[i+1][xx], sn[i+1][xx + 1], sn[i][xx + 1], sn[i][xx]]  # anti-clockwise
            #eles.append(o3.element.Quad(osi, nodes, ele_thick, o3.cc.PLANE_STRAIN, mat, b2=-grav * unit_masses[i]))
            eles.append(o3.element.SSPquad(osi, nodes, mat, o3.cc.PLANE_STRAIN, ele_thick, 0.0, -grav * unit_masses[i]))

    if base_imp >= 0:
        # define material and element for viscous dampers
        if base_imp == 0:
            sl = sp.get_soil_at_depth(sp.height)
            base_imp = sl.unit_dry_mass * sp.get_shear_vel_at_depth(sp.height)
        c_base = ele_width * base_imp / 1e3
        dashpot_mat = o3.uniaxial_material.Viscous(osi, c_base, alpha=1.)
        o3.element.ZeroLength(osi, [dashpot_node_l, dashpot_node_2], mats=[dashpot_mat], dirs=[o3.cc.DOF2D_X])

    # Static analysis
    o3.constraints.Transformation(osi)
    o3.test.NormDispIncr(osi, tol=1.0e-5, max_iter=30, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.ProfileSPD(osi)
    o3.integrator.Newmark(osi, gamma=0.5, beta=0.25)
    o3.analysis.Transient(osi)
    o3.analyze(osi, 10, 500.)

    for i in range(len(soil_mats)):
        if hasattr(soil_mats[i], 'update_to_nonlinear'):
            soil_mats[i].update_to_nonlinear(osi)
    for ele in eles:
        mat = ele.mat
        if hasattr(mat, 'set_nu'):
            mat.set_nu(mat.dynamic_poissons_ratio, eles=[ele])
    o3.analyze(osi, 40, 500.)

    # reset time and analysis
    o3.wipe_analysis(osi)
    o3.set_time(osi, 0.0)

    coords = o3.get_all_node_coords(osi)
    ele_node_tags = o3.get_all_ele_node_tags_as_dict(osi)
    all_node_xdisp_rec = o3.recorder.NodesToArrayCache(osi, 'all', [o3.cc.DOF2D_X], 'disp', nsd=4)
    all_node_ydisp_rec = o3.recorder.NodesToArrayCache(osi, 'all', [o3.cc.DOF2D_Y], 'disp', nsd=4)

    if hasattr(sp, 'hloads'):
        # Define the dynamic analysis
        o3.constraints.Transformation(osi)
        o3.test.NormDispIncr(osi, tol=1.0e-4, max_iter=30, p_flag=0)
        # o3.test_check.EnergyIncr(osi, tol=1.0e-6, max_iter=30)
        o3.algorithm.Newton(osi)
        o3.system.SparseGeneral(osi)
        o3.numberer.RCM(osi)
        o3.integrator.Newmark(osi, gamma=0.5, beta=0.25)
        o3.analysis.Transient(osi)
        # o3.rayleigh.Rayleigh(osi, a0, a1, 0, 0)
        pload = sp.hloads[0].p_x
        static_time = 100
        print('time: ', o3.get_time(osi))
        # Add static stress bias
        time_series = o3.time_series.Path(osi, time=[0, static_time / 2, static_time, 1e3], values=[0, 0.5, 1, 1], use_last=1)
        o3.pattern.Plain(osi, time_series)
        o3.Load(osi, sn[0][0], [pload * ele_width, 0])
        o3.Load(osi, sn[9][0], [-pload * ele_width, 0])
        if base_imp >= 0:
            o3.Load(osi, sn[-1][0], [-pload, 0])

        static_dt = 0.1
        o3.analyze(osi, int(static_time / static_dt) * 1.5, static_dt)
        o3.load_constant(osi, time=0)

        o3.wipe_analysis(osi)
    o3.set_time(osi, 0.0)  # TODO:
    # Define the dynamic analysis
    o3.constraints.Transformation(osi)
    o3.test.NormDispIncr(osi, tol=1.0e-4, max_iter=30, p_flag=0)
    # o3.test_check.EnergyIncr(osi, tol=1.0e-6, max_iter=30)
    o3.algorithm.Newton(osi)
    o3.system.SparseGeneral(osi)
    o3.numberer.RCM(osi)
    o3.integrator.Newmark(osi, gamma=0.5, beta=0.25)
    o3.analysis.Transient(osi)
    o3.rayleigh.Rayleigh(osi, a0, a1, 0, 0)

    init_time = o3.get_time(osi)
    o3sra_outs = O3SRAOutputs()
    o3sra_outs.start_recorders(osi, outs, sn, eles, rec_dt=rec_dt)

    # Define the dynamic input motion
    if base_imp < 0:  # fixed base
        acc_series = o3.time_series.Path(osi, dt=asig.dt, values=asig.values)
        o3.pattern.UniformExcitation(osi, dir=o3.cc.X, accel_series=acc_series)
    else:
        ts_obj = o3.time_series.Path(osi, dt=asig.dt, values=asig.velocity * 1, factor=c_base)
        o3.pattern.Plain(osi, ts_obj)
        o3.Load(osi, sn[-1][0], [1., 0.])
    if state == 3:
        o3.extensions.to_py_file(osi, opfile)
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
    print('site_response_w_pysra -> deprecated: use liquepy.sra.run_pysra')
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
