import eqsig
from collections import OrderedDict
import numpy as np
import sfsimodels as sm
import openseespy.opensees as opy
import o3seespy as o3
import o3seespy.extensions
import matplotlib.pyplot as plt
import copy

# for linear analysis comparison
import liquepy as lq
import pysra
from bwplot import cbox
import json


def site_response(sp, asig, linear=0, freqs=(0.5, 10), xi=0.03):
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

    osi = o3.OpenSeesInstance(ndm=2, ndf=2, state=3)
    assert isinstance(sp, sm.SoilProfile)
    sp.gen_split(props=['shear_vel', 'unit_mass', 'cohesion', 'phi', 'bulk_mod', 'poissons_ratio', 'strain_peak'])
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

    ele_width = min(thicknesses)
    total_soil_nodes = len(thicknesses) * 2 + 2

    # Define nodes and set boundary conditions for simple shear deformation
    # Start at top and build down?
    sn = [[o3.node.Node(osi, 0, 0), o3.node.Node(osi, ele_width, 0)]]
    for i in range(1, n_node_rows):
        # Establish left and right nodes
        sn.append([o3.node.Node(osi, 0, -node_depths[i]),
                    o3.node.Node(osi, ele_width, -node_depths[i])])
        # set x and y dofs equal for left and right nodes
        o3.EqualDOF(osi, sn[i][0], sn[i][1], [o3.cc.X, o3.cc.Y])

    # Fix base nodes
    o3.Fix2DOF(osi, sn[-1][0], o3.cc.FREE, o3.cc.FIXED)
    o3.Fix2DOF(osi, sn[-1][1], o3.cc.FREE, o3.cc.FIXED)

    # Define dashpot nodes
    dashpot_node_l = o3.node.Node(osi, 0, -node_depths[-1])
    dashpot_node_2 = o3.node.Node(osi, 0, -node_depths[-1])
    o3.Fix2DOF(osi, dashpot_node_l,  o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix2DOF(osi, dashpot_node_2, o3.cc.FREE, o3.cc.FIXED)

    # define equal DOF for dashpot and soil base nodes
    o3.EqualDOF(osi, sn[-1][0], sn[-1][1], [o3.cc.X])
    o3.EqualDOF(osi, sn[-1][0], dashpot_node_2, [o3.cc.X])

    # define materials
    ele_thick = 1.0  # m
    soil_mats = []
    strains = np.logspace(-6, -0.5, 16)
    ref_strain = 0.005
    rats = 1. / (1 + (strains / ref_strain) ** 0.91)
    prev_args = []
    prev_kwargs = {}
    prev_sl_type = None

    for i in range(len(thicknesses)):
        y_depth = ele_depths[i]

        sl_id = sp.get_layer_index_by_depth(y_depth)
        sl = sp.layer(sl_id)
        app2mod = {}
        if y_depth > sp.gwl:
            umass = sl.unit_sat_mass / 1e3
        else:
            umass = sl.unit_dry_mass / 1e3
        # Define material
        if sl.type == 'pm4sand':
            sl_class = o3.nd_material.PM4Sand
            overrides = {'nu': pois, 'p_atm': 101, 'unit_moist_mass': umass}
            app2mod = sl.app2mod
        elif sl.type == 'sdmodel':
            sl_class = o3.nd_material.StressDensity
            overrides = {'nu': pois, 'p_atm': 101, 'unit_moist_mass': umass}
            app2mod = sl.app2mod
        elif sl.type == 'pimy':
            sl_class = o3.nd_material.PressureIndependMultiYield
            overrides = {'nu': pois, 'p_atm': 101, 'unit_moist_mass': umass, 'bulk_mod': sl.bulk_mods / 1e3}
        else:
            sl_class = o3.nd_material.ElasticIsotropic
            sl.e_mod = 2 * sl.g_mod * (1 - sl.poissons_ratio) / 1e3
            app2mod['rho'] = 'unit_moist_mass'
            overrides = {'nu': sl.poissons_ratio, 'unit_moist_mass': umass}

        # opw.extensions.to_py_file(osi)
        args, kwargs = o3.extensions.get_o3_kwargs_from_obj(sl, sl_class, custom=app2mod, overrides=overrides)
        changed = 0
        if sl.type != prev_sl_type or len(args) != len(prev_args) or len(kwargs) != len(prev_kwargs):
            changed = 1
        else:
            for j, arg in enumerate(args):
                if not np.isclose(arg, prev_args[j]):
                    changed = 1
            for pm in kwargs:
                if pm not in prev_kwargs or not np.isclose(kwargs[pm], prev_kwargs[pm]):
                    changed = 1

        if changed:
            mat = sl_class(osi, *args, **kwargs)
            prev_sl_type = sl.type
            prev_args = copy.deepcopy(args)
            prev_kwargs = copy.deepcopy(kwargs)

            soil_mats.append(mat)

        # def element
        nodes = [sn[i+1][0], sn[i+1][1], sn[i][1], sn[i][0]]  # anti-clockwise
        ele = o3.element.Quad(osi, nodes, ele_thick, o3.cc.PLANE_STRAIN, mat, b2=grav * unit_masses[i])

    # define material and element for viscous dampers
    base_sl = sp.layer(sp.n_layers)
    c_base = ele_width * base_sl.unit_dry_mass / 1e3 * sp.get_shear_vel_at_depth(sp.height)
    dashpot_mat = o3.uniaxial_material.Viscous(osi, c_base, alpha=1.)
    o3.element.ZeroLength(osi, [dashpot_node_l, dashpot_node_2], mats=[dashpot_mat], dirs=[o3.cc.DOF2D_X])

    # Static analysis
    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-4, max_iter=30, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.ProfileSPD(osi)
    o3.integrator.Newmark(osi, newmark_gamma, newmark_beta)
    o3.analysis.Transient(osi)
    o3.analyze(osi, 40, 1.)

    for i in range(len(soil_mats)):
        if isinstance(soil_mats[i], o3.nd_material.PM4Sand):
            o3.update_material_stage(osi, soil_mats[i], 1)
    o3.analyze(osi, 50, 0.5)

    # reset time and analysis
    o3.set_time(osi, 0.0)
    o3.wipe_analysis(osi)

    # o3.recorder.NodeToFile(osi, 'sample_out.txt', node=nd["R0L"], dofs=[o3.cc.X], res_type='accel')
    na = o3.recorder.NodeToArrayCache(osi, node=sn[0][0], dofs=[o3.cc.X], res_type='accel')

    # Define the dynamic analysis
    ts_obj = o3.time_series.Path(osi, dt=asig.dt, values=asig.velocity * -1, factor=c_base)
    o3.pattern.Plain(osi, ts_obj)
    o3.Load(osi, sn[-1][0], [1., 0.])

    # Run the dynamic analysis
    o3.algorithm.Newton(osi)
    o3.system.SparseGeneral(osi)
    o3.numberer.RCM(osi)
    o3.constraints.Transformation(osi)
    o3.integrator.Newmark(osi, newmark_gamma, newmark_beta)
    o3.rayleigh.Rayleigh(osi, a0, a1, 0, 0)
    o3.analysis.Transient(osi)

    o3.test_check.EnergyIncr(osi, tol=1.0e-7, max_iter=10)
    analysis_time = asig.time[-1]
    analysis_dt = asig.dt / 10
    o3.extensions.to_py_file(osi)

    while opy.getTime() < analysis_time:
        print(opy.getTime())
        if o3.analyze(osi, 1, analysis_dt):
            print('failed')
            break
    opy.wipe()
    outputs = {
        "time": np.arange(0, analysis_time, analysis_dt),
        "rel_disp": [],
        "rel_accel": na.collect(),
    }

    return outputs

