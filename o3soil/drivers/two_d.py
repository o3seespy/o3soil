import o3seespy as o3

import liquepy as lq
import numpy as np
import eqsig
import sfsimodels as sm


def run_2d_stress_driver(osi, base_mat, esig_v0, forces, d_step=0.001, max_steps=10000, handle='silent', da_strain_max=0.05, max_cycles=200, srate=0.0001, esig_v_min=1.0, k0_init=1, verbose=0,
                   cyc_lim_fail=True):
    if k0_init != 1:
        raise ValueError('Only supports k0=1')
    max_steps_per_half_cycle = 50000

    nodes = [
        o3.node.Node(osi, 0.0, 0.0),
        o3.node.Node(osi, 1.0, 0.0),
        o3.node.Node(osi, 1.0, 1.0),
        o3.node.Node(osi, 0.0, 1.0)
    ]
    for node in nodes:
        o3.Fix2DOF(osi, node, 1, 1)

    mat = o3.nd_material.InitStressNDMaterial(osi, other=base_mat, init_stress=-esig_v0, n_dim=2)

    ele = o3.element.SSPquad(osi, nodes, mat, 'PlaneStrain', 1, 0.0, 0.0)

    # create analysis
    o3.constraints.Penalty(osi, 1.0e15, 1.0e15)
    o3.algorithm.Linear(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    o3.analysis.Static(osi)

    d_init = 0.0
    d_max = 0.1  # element height is 1m
    max_time = (d_max - d_init) / srate

    ts0 = o3.time_series.Linear(osi, factor=1)
    o3.pattern.Plain(osi, ts0)
    o3.Load(osi, nodes[2], [1.0, 0.0])
    o3.Load(osi, nodes[3], [1.0, 0.0])

    o3.analyze(osi, 1)
    o3.set_parameter(osi, value=1, eles=[ele], args=['materialState'])
    o3.update_material_stage(osi, base_mat, 1)
    o3.analyze(osi, 1)

    exit_code = None
    print('hhh')
    # loop through the total number of cycles
    react = 0
    strain = [0]
    stresses = o3.get_ele_response(osi, ele, 'stress')
    stress = [stresses[2]]
    v_eff = [stresses[1]]
    h_eff = [stresses[0]]
    diffs = np.diff(forces, prepend=0)
    orys = np.where(diffs >= 0, 1, -1)
    for i in range(len(forces)):
        print('i: ', i, d_step)
        ory = orys[i]
        o3.integrator.DisplacementControl(osi, nodes[2], o3.cc.DOF2D_X, -d_step * ory)
        o3.Load(osi, nodes[2], [ory * 1.0, 0.0])
        o3.Load(osi, nodes[3], [ory * 1.0, 0.0])
        for j in range(max_steps):
            if react * ory < forces[i] * ory:
                o3.analyze(osi, 1)
            else:
                print('reached!')
                break
            o3.gen_reactions(osi)
            # react = o3.get_ele_response(osi, ele, 'force')[0]
            stresses = o3.get_ele_response(osi, ele, 'stress')
            # print(stresses)
            tau = stresses[2]
            print(tau, forces[i], ory)
            react = -tau
            v_eff.append(stresses[1])
            h_eff.append(stresses[0])
            stress.append(tau)
            end_strain = -o3.get_node_disp(osi, nodes[2], dof=o3.cc.DOF2D_X)
            strain.append(end_strain)
        if j == max_steps - 1:
            if handle == 'silent':
                break
            if handle == 'warn':
                print(f'Target force not reached: force={react:.4g}, target: {forces[i]:.4g}')
            else:
                raise ValueError()

    return np.array(stress), np.array(strain), np.array(v_eff), np.array(h_eff), exit_code


def run_2d_strain_driver_iso(osi, base_mat, esig_v0, disps, target_d_inc=0.00001, max_steps=10000, handle='silent', da_strain_max=0.05, max_cycles=200, srate=0.0001, esig_v_min=1.0, k0_init=1, verbose=0,
                   cyc_lim_fail=True):
    if not np.isclose(k0_init, 1., rtol=0.05):
        raise ValueError(f'Only supports k0=1, current k0={k0_init:.3f}')
    max_steps_per_half_cycle = 50000

    nodes = [
        o3.node.Node(osi, 0.0, 0.0),
        o3.node.Node(osi, 1.0, 0.0),
        o3.node.Node(osi, 1.0, 1.0),
        o3.node.Node(osi, 0.0, 1.0)
    ]
    for node in nodes:
        o3.Fix2DOF(osi, node, 1, 1)

    mat = o3.nd_material.InitStressNDMaterial(osi, other=base_mat, init_stress=-esig_v0, n_dim=2)

    ele = o3.element.SSPquad(osi, nodes, mat, 'PlaneStrain', 1, 0.0, 0.0)

    # create analysis
    o3.constraints.Penalty(osi, 1.0e15, 1.0e15)
    o3.algorithm.Linear(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    o3.analysis.Static(osi)

    d_init = 0.0
    d_max = 0.1  # element height is 1m
    max_time = (d_max - d_init) / srate

    ts0 = o3.time_series.Linear(osi, factor=1)
    o3.pattern.Plain(osi, ts0)
    o3.Load(osi, nodes[2], [1.0, 0.0])
    o3.Load(osi, nodes[3], [1.0, 0.0])

    o3.analyze(osi, 1)
    o3.set_parameter(osi, value=1, eles=[ele], args=['materialState'])
    o3.update_material_stage(osi, base_mat, 1)
    o3.analyze(osi, 1)

    exit_code = None
    # loop through the total number of cycles
    react = 0
    strain = [0]
    stresses = o3.get_ele_response(osi, ele, 'stress')
    stress = [stresses[2]]
    v_eff = [stresses[1]]
    h_eff = [stresses[0]]
    d_incs = np.diff(disps, prepend=0)
    # orys = np.where(diffs >= 0, 1, -1)
    for i in range(len(disps)):
        d_inc_i = d_incs[i]
        if target_d_inc < abs(d_inc_i):
            n = int(abs(d_inc_i / target_d_inc))
            d_step = d_inc_i / n
        else:
            n = 1
            d_step = d_inc_i
        for j in range(n):
            o3.integrator.DisplacementControl(osi, nodes[2], o3.cc.DOF2D_X, -d_step)
            o3.Load(osi, nodes[2], [1.0, 0.0])
            o3.Load(osi, nodes[3], [1.0, 0.0])
            o3.analyze(osi, 1)
            o3.gen_reactions(osi)
            # react = o3.get_ele_response(osi, ele, 'force')[0]
            stresses = o3.get_ele_response(osi, ele, 'stress')
            v_eff.append(stresses[1])
            h_eff.append(stresses[0])
            force0 = o3.get_node_reaction(osi, nodes[0], o3.cc.DOF2D_X)
            force1 = o3.get_node_reaction(osi, nodes[1], o3.cc.DOF2D_X)
            stress.append(-force0 - force1)
            # stress.append(stresses[2])
            end_strain = -o3.get_node_disp(osi, nodes[2], dof=o3.cc.DOF2D_X)
            strain.append(end_strain)

    return -np.array(stress), np.array(strain), np.array(v_eff), np.array(h_eff), exit_code


def run_2d_strain_driver(osi, mat, esig_v0, disps, target_d_inc=0.00001, handle='silent', verbose=0):
    k0 = 1.0
    pois = k0 / (1 + k0)
    damp = 0.05
    omega0 = 0.2
    omega1 = 20.0
    a1 = 2. * damp / (omega0 + omega1)
    a0 = a1 * omega0 * omega1

    # Establish nodes
    h_ele = 1.
    nodes = [
        o3.node.Node(osi, 0.0, 0.0),
        o3.node.Node(osi, h_ele, 0.0),
        o3.node.Node(osi, h_ele, h_ele),
        o3.node.Node(osi, 0.0, h_ele)
    ]

    # Fix bottom node
    o3.Fix2DOF(osi, nodes[0], o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix2DOF(osi, nodes[1], o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix2DOF(osi, nodes[2], o3.cc.FREE, o3.cc.FREE)
    o3.Fix2DOF(osi, nodes[3], o3.cc.FREE, o3.cc.FREE)
    # Set out-of-plane DOFs to be slaved
    o3.EqualDOF(osi, nodes[2], nodes[3], [o3.cc.X, o3.cc.Y])

    ele = o3.element.SSPquad(osi, nodes, mat, 'PlaneStrain', 1, 0.0, 0.0)

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    o3.integrator.Newmark(osi, gamma=5./6, beta=4./9)
    o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Transient(osi)
    o3.update_material_stage(osi, mat, stage=0)

    # Add static vertical pressure and stress bias
    time_series = o3.time_series.Path(osi, time=[0, 100, 1e10], values=[0, 1, 1])
    o3.pattern.Plain(osi, time_series)
    o3.Load(osi, nodes[2], [0, -esig_v0 / 2])
    o3.Load(osi, nodes[3], [0, -esig_v0 / 2])

    o3.analyze(osi, num_inc=110, dt=1)

    ts2 = o3.time_series.Path(osi, time=[110, 80000, 1e10], values=[1., 1., 1.], factor=1)
    o3.pattern.Plain(osi, ts2, fact=1.)
    y_vert = o3.get_node_disp(osi, nodes[2], o3.cc.Y)
    o3.SP(osi, nodes[3], dof=o3.cc.Y, dof_values=[y_vert])
    o3.SP(osi, nodes[2], dof=o3.cc.Y, dof_values=[y_vert])

    o3.analyze(osi, 25, dt=1)

    o3.wipe_analysis(osi)
    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    o3.analysis.Static(osi)

    o3.update_material_stage(osi, mat, stage=1)
    o3.analyze(osi, 25, dt=1)
    # o3.set_parameter(osi, value=sl.poissons_ratio, eles=[ele], args=['poissonRatio', 1])

    o3.extensions.to_py_file(osi)

    exit_code = None
    strain = [0]
    stresses = o3.get_ele_response(osi, ele, 'stress')
    force0 = o3.get_node_reaction(osi, nodes[0], o3.cc.DOF2D_X)
    force1 = o3.get_node_reaction(osi, nodes[1], o3.cc.DOF2D_X)
    stress = [-force0 - force1]
    v_eff = [stresses[1]]
    h_eff = [stresses[0]]
    d_incs = np.diff(disps, prepend=0)
    for i in range(len(disps)):
        d_inc_i = d_incs[i]
        if target_d_inc < abs(d_inc_i):
            n = int(abs(d_inc_i / target_d_inc))
            d_step = d_inc_i / n
        else:
            n = 1
            d_step = d_inc_i
        for j in range(n):
            o3.integrator.DisplacementControl(osi, nodes[2], o3.cc.DOF2D_X, -d_step)
            o3.Load(osi, nodes[2], [1.0, 0.0])
            o3.Load(osi, nodes[3], [1.0, 0.0])
            o3.analyze(osi, 1)
            o3.gen_reactions(osi)
            stresses = o3.get_ele_response(osi, ele, 'stress')
            print(stresses)
            v_eff.append(stresses[1])
            h_eff.append(stresses[0])
            force0 = o3.get_node_reaction(osi, nodes[0], o3.cc.DOF2D_X)
            force1 = o3.get_node_reaction(osi, nodes[1], o3.cc.DOF2D_X)
            stress.append(-force0 - force1)
            end_strain = o3.get_node_disp(osi, nodes[2], dof=o3.cc.DOF2D_X)
            strain.append(end_strain)

    return -np.array(stress), -np.array(strain), np.array(v_eff), np.array(h_eff), exit_code


def _set_hyperbolic_params_from_op_pimy_model(sl, esig_v0, strain_max):
    k0 = 1 - np.sin(sl.phi_r)
    p_eff = (esig_v0 * (1 + 1 * k0) / 2)
    # Octahedral shear stress
    tau_f = (2 * np.sqrt(2.) * np.sin(sl.phi_r)) / (3 - np.sin(sl.phi_r)) * p_eff + 2 * np.sqrt(2.) / 3 * sl.cohesion
    if hasattr(sl, 'get_g_mod_at_v_eff_stress'):
        g_mod_r = sl.g0_mod * sl.p_atm
        d = sl.a
        p_r = sl.p_atm
    else:
        g_mod_r = sl.g_mod
        d = 0.0
        p_r = 1

    strain_r = strain_max * tau_f / (g_mod_r * strain_max - tau_f)
    sdf = (p_r / p_eff) ** d
    sl.strain_curvature = 1.0
    sl.xi_min = 0.01
    sl.sra_type = "hyperbolic"
    sl.strain_ref = strain_r / sdf


def get_pimy_soil():
    sl = sm.Soil()
    vs = 200.
    unit_mass = 1700.0
    sl.cohesion = 68.0e3
    sl.phi = 0.0
    sl.g_mod = vs ** 2 * unit_mass
    print('G_mod: ', sl.g_mod)
    sl.unit_dry_weight = unit_mass * 9.8
    sl.specific_gravity = 2.65
    sl.poissons_ratio = 0.3
    # sl.e_curr = 0.7
    sl.id = 1
    assert np.isclose(vs, sl.get_shear_vel(saturated=False))
    strain_max = 0.05
    strains = np.logspace(-4, -1.5, 40)
    _set_hyperbolic_params_from_op_pimy_model(sl, 1, strain_max)
    return sl


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    sl = get_pimy_soil()

    vert_sig_eff = 100.0
    csr_level = 2.0 / abs(vert_sig_eff)
    tau = np.array([30, 5, 55, -10, 40, -30, 30])
    # forces = np.array([4., -3, 5])
    osi = o3.OpenSeesInstance(ndm=2, ndf=2, state=3)
    base_mat = o3.nd_material.PressureIndependMultiYield(osi,
                                                         nd=2,
                                                         rho=sl.unit_sat_mass / 1e3,
                                                         g_mod_ref=sl.g_mod / 1e3,
                                                         bulk_mod_ref=sl.bulk_mod / 1e3,
                                                         peak_strain=0.05,
                                                         cohesion=sl.cohesion / 1e3,
                                                         phi=sl.phi,
                                                         p_ref=vert_sig_eff,
                                                         d=0.0,
                                                         n_surf=20
                                                         )

    disps = np.array([0.0, 0.00003, -0.00003, 0.0004, 0.0001, 0.0009, -0.0012]) * 10
    disps = np.array([0.0, 0.001, 0.001])
    stress, strain, v_eff, h_eff, exit_code = run_2d_strain_driver(osi, base_mat, esig_v0=vert_sig_eff, disps=disps,
                                                                   handle='warn', verbose=1)
    print(exit_code)
    pis = eqsig.get_peak_array_indices(stress)
    n_cycs = 0.5 * np.arange(len(pis)) - 0.25
    n_cycs[0] = 0
    n_cycles = np.interp(np.arange(len(stress)), pis, n_cycs)
    bf, sps = plt.subplots(nrows=2)
    sps[0].plot(strain, stress)
    sps[0].plot(strain[pis], stress[pis], 'o')
    sps[0].axhline(sl.cohesion / 1e3, c='k')
    sps[1].plot(stress)
    sps[1].plot(v_eff)
    sps[1].plot(h_eff)

    plt.show()

