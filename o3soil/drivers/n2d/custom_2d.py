import o3seespy as o3
import numpy as np
import math


def run_ts_custom_strain(mat, esig_v0, strains, osi=None, nu_dyn=None, target_d_inc=0.00001, k0=None,
                         handle='silent', verbose=0, opyfile=None, dss=False, plain_strain=True, min_n=10):
    if dss:
        raise ValueError('dss option is not working')
    damp = 0.05
    omega0 = 0.2
    omega1 = 20.0
    a1 = 2. * damp / (omega0 + omega1)  # beta
    a0 = a1 * omega0 * omega1  # alpha
    # or
    # alpha = xi_min * omega_min
    # beta = xi_min / omega_min

    if osi is None:
        osi = o3.OpenSeesInstance(ndm=2, ndf=2)
        mat.build(osi)

    # Establish nodes
    h_ele = 1.
    nodes = [
        o3.node.Node(osi, 0.0, 0.0),
        o3.node.Node(osi, h_ele, 0.0),
        o3.node.Node(osi, h_ele, h_ele),
        o3.node.Node(osi, 0.0, h_ele)
    ]

    # Fix bottom node
    if k0 is None:
        o3.Fix2DOF(osi, nodes[0], o3.cc.FIXED, o3.cc.FIXED)
        o3.Fix2DOF(osi, nodes[1], o3.cc.FIXED, o3.cc.FIXED)
        # Set out-of-plane DOFs to be slaved
        o3.EqualDOF(osi, nodes[2], nodes[3], [o3.cc.X, o3.cc.Y])
    else:  # control k0 with node forces
        o3.Fix2DOF(osi, nodes[0], o3.cc.FIXED, o3.cc.FIXED)
        o3.Fix2DOF(osi, nodes[1], o3.cc.FIXED, o3.cc.FREE)

    if plain_strain:
        oop = 'PlaneStrain'
    else:
        oop = 'PlaneStress'

    ele = o3.element.SSPquad(osi, nodes, mat, oop, 1, 0.0, 0.0)

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    o3.integrator.LoadControl(osi, 1)
    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Static(osi)
    o3.update_material_stage(osi, mat, stage=0)

    # Add static vertical pressure and stress bias
    time_series = o3.time_series.Path(osi, time=[0, 100, 1e10], values=[0, 1, 1])
    o3.pattern.Plain(osi, time_series)

    if k0:
        o3.Load(osi, nodes[2], [-esig_v0 / 2 * k0, -esig_v0 / 2])
        o3.Load(osi, nodes[3], [esig_v0 / 2 * k0, -esig_v0 / 2])
        o3.Load(osi, nodes[1], [-esig_v0 / 2 * k0, 0])
        # node 0 is fixed
    else:
        o3.Load(osi, nodes[2], [0, -esig_v0 / 2])
        o3.Load(osi, nodes[3], [0, -esig_v0 / 2])

    o3.analyze(osi, num_inc=200)
    stresses = o3.get_ele_response(osi, ele, 'stress')
    print('init_stress0: ', stresses)
    o3.load_constant(osi, 100)

    # o3.wipe_analysis(osi)
    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    o3.analysis.Static(osi)

    if hasattr(mat, 'update_to_nonlinear'):
        print('set to nonlinear')
        mat.update_to_nonlinear()
        o3.analyze(osi, 25, dt=1)
    if nu_dyn is not None:
        mat.set_nu(nu_dyn, eles=[ele])
        o3.analyze(osi, 25, dt=1)

    # o3.extensions.to_py_file(osi)
    stresses = o3.get_ele_response(osi, ele, 'stress')
    print('init_stress1: ', stresses)

    # Prepare for reading results
    exit_code = None
    stresses = o3.get_ele_response(osi, ele, 'stress')
    if dss:
        o3.gen_reactions(osi)
        force0 = o3.get_node_reaction(osi, nodes[2], o3.cc.DOF2D_X)
        force1 = o3.get_node_reaction(osi, nodes[3], o3.cc.DOF2D_X)
        # force2 = o3.get_node_reaction(osi, nodes[0], o3.cc.DOF2D_X)
        stress = [force1 + force0]
        strain = [o3.get_node_disp(osi, nodes[2], dof=o3.cc.DOF2D_X)]
        sxy_ind = None
        gxy_ind = None
        # iforce0 = o3.get_node_reaction(osi, nodes[0], o3.cc.DOF2D_X)
        # iforce1 = o3.get_node_reaction(osi, nodes[1], o3.cc.DOF2D_X)
        # iforce2 = o3.get_node_reaction(osi, nodes[2], o3.cc.DOF2D_X)
        # iforce3 = o3.get_node_reaction(osi, nodes[3], o3.cc.DOF2D_X)
        # print(iforce0, iforce1, iforce2, iforce3, stresses[2])
    else:
        ro = o3.recorder.load_recorder_options()
        import pandas as pd
        df = pd.read_csv(ro)
        mat_type = ele.mat.type
        dfe = df[(df['mat'] == mat_type) & (df['form'] == oop)]
        df_sxy = dfe[dfe['recorder'] == 'stress']
        outs = df_sxy['outs'].iloc[0].split('-')
        sxy_ind = outs.index('sxy')

        df_gxy = dfe[dfe['recorder'] == 'strain']
        outs = df_gxy['outs'].iloc[0].split('-')
        gxy_ind = outs.index('gxy')
        stress = [stresses[sxy_ind]]
        cur_strains = o3.get_ele_response(osi, ele, 'strain')
        strain = [cur_strains[gxy_ind]]
    v_eff = [stresses[1]]
    h_eff = [stresses[0]]
    d_incs = np.diff(strains, prepend=0)
    for i in range(len(strains)):
        d_inc_i = d_incs[i]
        n = int(max(abs(d_inc_i / target_d_inc), min_n))
        d_step = d_inc_i / n
        o3.integrator.DisplacementControl(osi, nodes[2], o3.cc.DOF2D_X, -d_step)
        o3.Load(osi, nodes[2], [1.0, 0.0])
        o3.Load(osi, nodes[3], [1.0, 0.0])
        for j in range(n):
            o3.analyze(osi, 1)

            stresses = o3.get_ele_response(osi, ele, 'stress')
            if opyfile:
                import o3seespy.extensions
                o3.extensions.to_py_file(osi, opyfile)
                opyfile = None
            v_eff.append(stresses[1])
            h_eff.append(stresses[0])
            if dss:
                o3.gen_reactions(osi)
                force0 = o3.get_node_reaction(osi, nodes[2], o3.cc.DOF2D_X)
                force1 = o3.get_node_reaction(osi, nodes[3], o3.cc.DOF2D_X)
                stress.append(force1 + force0)
                strain.append(o3.get_node_disp(osi, nodes[2], dof=o3.cc.DOF2D_X))
            else:
                stress.append(stresses[sxy_ind])
                cur_strains = o3.get_ele_response(osi, ele, 'strain')
                strain.append(cur_strains[gxy_ind])

    return -np.array(stress), -np.array(strain), np.array(v_eff), np.array(h_eff), exit_code


def run_ud_custom_strain(mat, esig_v0, strains, osi=None, nu_dyn=None, target_d_inc=0.00001, handle='silent',
                         verbose=0, opyfile=None, dss=False, damp_freqs=(0.5, 10)):
    """must be in kPa"""

    damp = 0.02
    omega0 = 0.2
    omega1 = 20.0
    a1 = 2. * damp / (omega0 + omega1)
    a0 = a1 * omega0 * omega1

    if osi is None:
        osi = o3.OpenSeesInstance(ndm=2, ndf=3)
        mat.build(osi)

    # Establish nodes
    h_ele = 1.
    nodes = [
        o3.node.Node(osi, 0.0, 0.0),
        o3.node.Node(osi, h_ele, 0.0),
        o3.node.Node(osi, h_ele, h_ele),
        o3.node.Node(osi, 0.0, h_ele)
    ]

    # Fix bottom node
    o3.Fix3DOF(osi, nodes[0], o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix3DOF(osi, nodes[1], o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix3DOF(osi, nodes[2], o3.cc.FREE, o3.cc.FREE, o3.cc.FIXED)
    o3.Fix3DOF(osi, nodes[3], o3.cc.FREE, o3.cc.FREE, o3.cc.FIXED)
    # Set out-of-plane DOFs to be slaved
    o3.EqualDOF(osi, nodes[2], nodes[3], [o3.cc.X, o3.cc.Y])

    water_bulk_mod = 2.2e6
    ele = o3.element.SSPquadUP(osi, nodes, mat, 1.0, water_bulk_mod, 1.,
                               1.0e-4, 1.0e-4, 0.6, alpha=1.0e-5, b1=0.0, b2=0.0)

    if hasattr(mat, 'update_to_linear'):
        mat.update_to_linear()

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    o3.integrator.Newmark(osi, gamma=5. / 6, beta=4. / 9)
    o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Transient(osi)

    # Add static vertical pressure and stress bias
    time_series = o3.time_series.Path(osi, time=[0, 200, 1e10], values=[0, 1, 1])
    o3.pattern.Plain(osi, time_series)
    o3.Load(osi, nodes[2], [0, -esig_v0 / 2, 0])
    o3.Load(osi, nodes[3], [0, -esig_v0 / 2, 0])

    o3.analyze(osi, num_inc=2100, dt=0.1)
    init_stresses = o3.get_ele_response(osi, ele, 'stress')
    print('init_stress0: ', init_stresses)

    ts2 = o3.time_series.Path(osi, time=[210, 80000, 1e10], values=[1., 1., 1.], factor=1)
    o3.pattern.Plain(osi, ts2, fact=1.)
    y_vert = o3.get_node_disp(osi, nodes[3], o3.cc.Y)
    o3.SP(osi, nodes[2], dof=o3.cc.Y, dof_values=[y_vert])
    o3.SP(osi, nodes[3], dof=o3.cc.Y, dof_values=[y_vert])

    # Close the drainage valves
    for node in nodes:
        o3.remove_sp(osi, node, dof=3)
    o3.analyze(osi, 250, dt=0.1)
    print('here3: ', o3.get_ele_response(osi, ele, 'stress'), esig_v0)

    if hasattr(mat, 'update_to_nonlinear'):
        mat.update_to_nonlinear()
        # o3.analyze(osi, 250, dt=0.001)
    curr_stresses = o3.get_ele_response(osi, ele, 'stress')
    print('init_stress1: ', curr_stresses)
    if hasattr(mat, 'set_first_call'):
        o3.update_material_stage(osi, ele.mat, stage=1)
        mat.set_first_call(value=0, ele=ele)
        o3.analyze(osi, 250, dt=0.001)
    curr_stresses = o3.get_ele_response(osi, ele, 'stress')
    print('init_stress1: ', curr_stresses)
    if nu_dyn is not None:
        mat.set_nu(nu_dyn, ele=ele)
        o3.analyze(osi, 25, dt=0.1)

    o3.extensions.to_py_file(osi)
    curr_stresses = o3.get_ele_response(osi, ele, 'stress')
    print('init_stress1: ', curr_stresses)
    if dss:
        sxy_ind = None
        gxy_ind = None
        force0 = o3.get_node_reaction(osi, nodes[0], o3.cc.DOF2D_X)
        force1 = o3.get_node_reaction(osi, nodes[1], o3.cc.DOF2D_X)
        stress = [-force0 - force1]
    else:
        ro = o3.recorder.load_recorder_options()
        import pandas as pd
        df = pd.read_csv(ro)
        mat_type = ele.mat.type
        oop = o3.cc.PLANE_STRAIN
        dfe = df[(df['mat'] == mat_type) & (df['form'] == oop)]
        df_sxy = dfe[dfe['recorder'] == 'stress']
        outs = df_sxy['outs'].iloc[0].split('-')
        sxy_ind = outs.index('sxy')
        df_gxy = dfe[dfe['recorder'] == 'strain']
        outs = df_gxy['outs'].iloc[0].split('-')
        gxy_ind = outs.index('gxy')
        stress = [curr_stresses[sxy_ind]]
        # cur_strains = o3.get_ele_response(osi, ele, 'strain')
        # strain = [cur_strains[gxy_ind]]
    strain = [0.0]
    exit_code = None

    v_eff = [curr_stresses[1]]
    h_eff = [curr_stresses[0]]
    ihd = o3.get_node_disp(osi, nodes[2], dof=o3.cc.X)  # initial horizontal displacement
    target_disps = np.array(strains) + ihd

    for i in range(len(target_disps)):
        h_disp = o3.get_node_disp(osi, nodes[2], o3.cc.X)
        curr_time = o3.get_time(osi)
        steps = int(abs(target_disps[i] - h_disp) / target_d_inc)
        ts0 = o3.time_series.Path(osi, time=[curr_time, curr_time + steps, 1e10],
                                  values=[h_disp, target_disps[i], target_disps[i]],
                                  factor=1)
        pat0 = o3.pattern.Plain(osi, ts0)
        o3.SP(osi, nodes[2], dof=o3.cc.X, dof_values=[1.0])
        for j in range(steps):
            o3.analyze(osi, 1, dt=1)
            curr_stresses = o3.get_ele_response(osi, ele, 'stress')
            if opyfile:
                import o3seespy.extensions
                o3.extensions.to_py_file(osi, opyfile)
                opyfile = None
            v_eff.append(curr_stresses[1])
            h_eff.append(curr_stresses[0])
            if dss:
                o3.gen_reactions(osi)
                force0 = o3.get_node_reaction(osi, nodes[0], o3.cc.DOF2D_X)
                force1 = o3.get_node_reaction(osi, nodes[1], o3.cc.DOF2D_X)
                stress.append(-force0 - force1)
                strain.append(o3.get_node_disp(osi, nodes[2], dof=o3.cc.DOF2D_X) - ihd)
            else:
                stress.append(curr_stresses[sxy_ind])
                cur_strains = o3.get_ele_response(osi, ele, 'strain')
                strain.append(cur_strains[gxy_ind])

        o3.remove_load_pattern(osi, pat0)
        o3.remove(osi, ts0)
        o3.remove_sp(osi, nodes[2], dof=o3.cc.X)

    return np.array(stress), np.array(strain), -np.array(v_eff), -np.array(h_eff), exit_code


def run_ud_custom_stress(mat, esig_v0, stresses, osi=None, nu_dyn=None, target_d_inc=0.00001, handle='silent', verbose=0, opyfile=None):

    damp = 0.05
    omega0 = 0.2
    omega1 = 20.0
    a1 = 2. * damp / (omega0 + omega1)
    a0 = a1 * omega0 * omega1

    if osi is None:
        osi = o3.OpenSeesInstance(ndm=2, ndf=3)
        mat.build(osi)

    # Establish nodes
    h_ele = 1.
    nodes = [
        o3.node.Node(osi, 0.0, 0.0),
        o3.node.Node(osi, h_ele, 0.0),
        o3.node.Node(osi, h_ele, h_ele),
        o3.node.Node(osi, 0.0, h_ele)
    ]

    # Fix bottom node
    o3.Fix3DOF(osi, nodes[0], o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix3DOF(osi, nodes[1], o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix3DOF(osi, nodes[2], o3.cc.FREE, o3.cc.FREE, o3.cc.FIXED)
    o3.Fix3DOF(osi, nodes[3], o3.cc.FREE, o3.cc.FREE, o3.cc.FIXED)
    # Set out-of-plane DOFs to be slaved
    o3.EqualDOF(osi, nodes[2], nodes[3], [o3.cc.X, o3.cc.Y])

    # ele = o3.element.SSPquad(osi, nodes, mat, 'PlaneStrain', 1, 0.0, 0.0)
    water_bulk_mod = 2.2e6
    ele = o3.element.SSPquadUP(osi, nodes, mat, 1.0, water_bulk_mod, 1.,
                               1.0e-4, 1.0e-4, 0.6, alpha=1.0e-5, b1=0.0, b2=0.0)

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    o3.integrator.Newmark(osi, gamma=5. / 6, beta=4. / 9)
    o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Transient(osi)

    # Add static vertical pressure and stress bias
    time_series = o3.time_series.Path(osi, time=[0, 100, 1e10], values=[0, 1, 1])
    o3.pattern.Plain(osi, time_series)
    o3.Load(osi, nodes[2], [0, -esig_v0 / 2, 0])
    o3.Load(osi, nodes[3], [0, -esig_v0 / 2, 0])

    o3.analyze(osi, num_inc=110, dt=1)
    init_stresses = o3.get_ele_response(osi, ele, 'stress')
    print('init_stress0: ', init_stresses)

    ts2 = o3.time_series.Path(osi, time=[110, 80000, 1e10], values=[1., 1., 1.], factor=1)
    o3.pattern.Plain(osi, ts2, fact=1.)
    y_vert = o3.get_node_disp(osi, nodes[3], o3.cc.Y)
    o3.SP(osi, nodes[2], dof=o3.cc.Y, dof_values=[y_vert])
    o3.SP(osi, nodes[3], dof=o3.cc.Y, dof_values=[y_vert])

    # Close the drainage valves
    for node in nodes:
        o3.remove_sp(osi, node, dof=3)
    o3.analyze(osi, 25, dt=1)
    print('here3: ', o3.get_ele_response(osi, ele, 'stress'), esig_v0)

    if hasattr(mat, 'update_to_nonlinear'):
        mat.update_to_nonlinear()
        o3.analyze(osi, 25, dt=1)
    if hasattr(mat, 'set_first_call'):
        mat.set_first_call(value=0, ele=ele)
    if nu_dyn is not None:
        mat.set_nu(nu_dyn, ele=ele)
        o3.analyze(osi, 25, dt=1)

    o3.extensions.to_py_file(osi)
    curr_stresses = o3.get_ele_response(osi, ele, 'stress')
    print('init_stress1: ', curr_stresses)

    exit_code = None
    strain = [0]
    curr_stresses = o3.get_ele_response(osi, ele, 'stress')
    force0 = o3.get_node_reaction(osi, nodes[0], o3.cc.DOF2D_X)
    force1 = o3.get_node_reaction(osi, nodes[1], o3.cc.DOF2D_X)
    stress = [-force0 - force1]
    v_eff = [curr_stresses[1]]
    h_eff = [curr_stresses[0]]
    prev_stress = stress[0]
    o3.record(osi)
    for i in range(len(stresses)):
        if stresses[i] >= prev_stress:
            target_disp = 0.1
        else:
            target_disp = -0.1
        if stresses[i] >= 0:
            sgn_f = 1
        else:
            sgn_f = -1

        h_disp = o3.get_node_disp(osi, nodes[2], o3.cc.X)
        curr_time = o3.get_time(osi)
        steps = 1000
        ts0 = o3.time_series.Path(osi, time=[curr_time, curr_time + steps, 1e10], values=[h_disp, target_disp, target_disp],
                                  factor=1)
        pat0 = o3.pattern.Plain(osi, ts0)
        o3.SP(osi, nodes[2], dof=o3.cc.X, dof_values=[1.0])
        curr_stress = o3.get_ele_response(osi, ele, 'stress')[2]
        if math.isnan(curr_stress):
            raise ValueError

        if opyfile:
            o3.extensions.to_py_file(osi, opyfile)
            opyfile = None
        n = 0
        while curr_stress * sgn_f < stresses[i] * sgn_f and n < 1000:  # TODO: change the strain controlled version to be like this
            n += 1
            o3.analyze(osi, 1, dt=1)
            o3.gen_reactions(osi)
            curr_stresses = o3.get_ele_response(osi, ele, 'stress')
            if opyfile:
                import o3seespy.extensions
                o3.extensions.to_py_file(osi, opyfile)
                opyfile = None
            v_eff.append(curr_stresses[1])
            h_eff.append(curr_stresses[0])
            force0 = o3.get_node_reaction(osi, nodes[0], o3.cc.DOF2D_X)
            force1 = o3.get_node_reaction(osi, nodes[1], o3.cc.DOF2D_X)
            stress.append(-force0 - force1)
            curr_stress = stress[-1]
            end_strain = o3.get_node_disp(osi, nodes[2], dof=o3.cc.DOF2D_X)
            strain.append(end_strain)
            print(curr_stress, stresses[i])
        o3.remove_load_pattern(osi, pat0)
        o3.remove(osi, ts0)
        o3.remove_sp(osi, nodes[2], dof=o3.cc.X)

    return -np.array(stress), -np.array(strain), -np.array(v_eff), -np.array(h_eff), exit_code


def example_of_run_ud_custom_strain(show=0):
    import o3seespy as o3
    import liquepy as lq

    esig_v0 = 69.131
    gravity = 9.8

    sl = lq.num.o3.PM4Sand(liq_mass_density=1.0)
    sl.relative_density = 0.35
    sl.g0_mod = 476.0
    sl.h_po = 0.53
    sl.unit_sat_weight = 1.42 * gravity

    sl.e_min = 0.5
    sl.e_max = 0.8
    sl.poissons_ratio = 0.3
    sl.phi = 33.

    sl.permeability = 1.0e-9
    sl.p_atm = 101.0

    peak_strains = [0.0001, -0.001, 0.0002]

    nu_init = sl.poissons_ratio
    pm4sand = o3.nd_material.PM4Sand(None, sl.relative_density, sl.g0_mod, sl.h_po, sl.unit_sat_mass,
                                     p_atm=101.3, nu=nu_init)

    stress, strain, vstress, hstress, ecode = run_ud_custom_strain(pm4sand, esig_v0, peak_strains)
    if show:
        import matplotlib.pyplot as plt
        bf, sps = plt.subplots(nrows=3)
        sps[0].plot(stress, label='shear')
        sps[0].plot(vstress[0] - vstress, label='PPT')
        sps[1].plot(strain, stress, label='o3seespy')
        sps[2].plot(strain, label='o3seespy')
        for ps in peak_strains:
            sps[2].axhline(ps, c='k')
        w_dss = 1
        if w_dss:
            pm4sand = o3.nd_material.PM4Sand(None, sl.relative_density, sl.g0_mod, sl.h_po, sl.unit_sat_mass,
                                             p_atm=101.3, nu=nu_init)
    
            stress, strain, vstress, hstress, ecode = run_ud_custom_strain(pm4sand, esig_v0, peak_strains, dss=True)
            sps[0].plot(stress, label='shear')
            sps[0].plot(vstress[0] - vstress, label='PPT')
            sps[1].plot(strain, stress, label='o3seespy')
            sps[2].plot(strain, label='o3seespy')

        sps[0].set_xlabel('Time [s]')
        sps[0].set_ylabel('Stress [kPa]')
        sps[1].set_xlabel('Strain')
        sps[1].set_ylabel('Stress [kPa]')
        sps[0].legend()
        sps[1].legend()
        plt.show()


def example_of_run_ts_custom_strain(show=0):
    osi = o3.OpenSeesInstance(ndm=2, ndf=2, state=3)

    esig_v0 = 50.0e3
    poissons_ratio = 0.3
    g_mod = 1.0e6
    b_mod = 2 * g_mod * (1 + poissons_ratio) / (3 * (1 - 2 * poissons_ratio))

    # mat = o3.nd_material.PressureIndependMultiYield(osi, 2, 1600, g_mod, b_mod, 1800.0, 0.1, 0.0, 101000.0, 0.0, 25)
    mat = o3.nd_material.PM4Sand(osi, 0.45, 587, 0.3, 1.6e3, 101.3e3)
    # mat = o3.nd_material.ElasticIsotropic(osi, e_mod=1.0e6, nu=0.3)
    peak_strains = [0.001, -0.005, 0.01]
    ss, es, vp, hp, error = run_ts_custom_strain(mat, esig_v0=esig_v0, strains=peak_strains, osi=osi, target_d_inc=1.0e-4)
    if show:
        import matplotlib.pyplot as plt
        plt.plot(es, ss)
        plt.show()


def example_of_run_ts_custom_strain_w_dss(show=0):
    osi = o3.OpenSeesInstance(ndm=2, ndf=2, state=3)

    esig_v0 = 50.0e3
    poissons_ratio = 0.3
    g_mod = 1.0e6
    b_mod = 2 * g_mod * (1 + poissons_ratio) / (3 * (1 - 2 * poissons_ratio))

    # mat = o3.nd_material.PressureIndependMultiYield(osi, 2, 1600, g_mod, b_mod, 1800.0, 0.1, 0.0, 101000.0, 0.0, 25)
    # mat = o3.nd_material.PM4Sand(osi, 0.45, 587, 0.3, 1.6e3, 101.3e3)
    mat = o3.nd_material.ElasticIsotropic(osi, e_mod=1.0e6, nu=0.3)
    peak_strains = [0.001, -0.005, 0.01]
    ss, es, vp, hp, error = run_ts_custom_strain(mat, esig_v0=esig_v0, strains=peak_strains, osi=osi, target_d_inc=5.0e-4, dss=True)
    if show:
        import matplotlib.pyplot as plt
        plt.plot(es, ss)
        o3.wipe(osi)
        osi = o3.OpenSeesInstance(ndm=2, ndf=2, state=3)
        # mat = o3.nd_material.PressureIndependMultiYield(osi, 2, 1600, g_mod, b_mod, 1800.0, 0.1, 0.0, 101000.0, 0.0, 25)
        mat = o3.nd_material.ElasticIsotropic(osi, e_mod=1.0e6, nu=0.3)
        ss, es, vp, hp, error = run_ts_custom_strain(mat, esig_v0=esig_v0, strains=peak_strains, osi=osi,
                                                     target_d_inc=1.0e-4)
        plt.plot(es, ss)
        plt.show()


def example_compare_ts_loader_and_ud_loader(show=0):

    esig_v0 = 50.0e3
    poissons_ratio = 0.3
    g_mod = 1.0e6
    b_mod = 2 * g_mod * (1 + poissons_ratio) / (3 * (1 - 2 * poissons_ratio))

    mat = o3.nd_material.PressureIndependMultiYield(None, 2, 1600, g_mod, b_mod, 1800.0, 0.1, 0.0, 101000.0, 0.0, 25)
    peak_strains = [0.001, -0.005, 0.01, -0.01, 0.01, -0.01]
    ts_ss, ts_es, ts_vp, ts_hp, ts_error = run_ts_custom_strain(mat, esig_v0=esig_v0, strains=peak_strains,
                                                 target_d_inc=5.0e-4)
    mat = o3.nd_material.PressureIndependMultiYield(None, 2, 1600, g_mod, b_mod, 1800.0, 0.1, 0.0, 101000.0, 0.0, 25)
    ud_ss, ud_es, ud_vp, ud_hp, ud_error = run_ud_custom_strain(mat, esig_v0=esig_v0, strains=peak_strains,
                                                 target_d_inc=5.0e-4)
    if show:
        import matplotlib.pyplot as plt
        plt.plot(ts_es, ts_ss, label='TS')
        plt.plot(ud_es, ud_ss, label='UD')
        plt.xlabel('Shear strain')
        plt.xlabel('Shear stress [Pa]')
        plt.legend()
        plt.show()


def run_2d_strain_driver_vcon(osi, mat, esig_v0, disps, target_d_inc=0.00001, handle='silent', verbose=0, min_n=10):
    if osi is None:
        osi = o3.OpenSeesInstance(ndm=2, ndf=2, state=3)
        mat.build(osi)

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
    o3.test_check.NormDispIncr(osi, tol=1.0e-3, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    # o3.integrator.DisplacementControl(osi, nodes[2], o3.cc.DOF2D_Y, 0.005)
    o3.integrator.LoadControl(osi, incr=0.01)
    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Static(osi)

    all_stresses_cache = o3.recorder.ElementToArrayCache(osi, ele, arg_vals=['stress'])
    all_strains_cache = o3.recorder.ElementToArrayCache(osi, ele, arg_vals=['strain'])
    all_params_cache = o3.recorder.ElementToArrayCache(osi, ele, arg_vals=['state'])

    # Add static vertical pressure and stress bias
    # time_series = o3.time_series.Path(osi, time=[0, 100, 1e10], values=[0, 1, 1])
    # o3.pattern.Plain(osi, time_series)
    ts0 = o3.time_series.Linear(osi, factor=1)
    o3.pattern.Plain(osi, ts0)
    o3.Load(osi, nodes[2], [0, -esig_v0 / 2])
    o3.Load(osi, nodes[3], [0, -esig_v0 / 2])
    v_load0 = esig_v0 / 2
    v_load1 = esig_v0 / 2

    o3.analyze(osi, num_inc=100)
    stresses = o3.get_ele_response(osi, ele, 'stress')
    print('init_stress0: ', stresses)

    o3.load_constant(osi)

    o3.wipe_analysis(osi)
    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    o3.analysis.Static(osi)

    o3.analyze(osi, 25, dt=1)
    # o3.set_parameter(osi, value=sl.poissons_ratio, eles=[ele], args=['poissonRatio', 1])

    # o3.extensions.to_py_file(osi)
    stresses = o3.get_ele_response(osi, ele, 'stress')
    print('init_stress1: ', stresses)

    exit_code = None
    strain = [0]
    stresses = o3.get_ele_response(osi, ele, 'stress')
    force0 = o3.get_node_reaction(osi, nodes[0], o3.cc.DOF2D_X)
    force1 = o3.get_node_reaction(osi, nodes[1], o3.cc.DOF2D_X)
    stress = [-force0 - force1]
    v_eff = [stresses[1]]
    h_eff = [stresses[0]]
    d_incs = np.diff(disps, prepend=0)
    o3.Load(osi, nodes[2], [1.0, 0.0])
    o3.Load(osi, nodes[3], [1.0, 0.0])

    ts_vload = o3.time_series.Constant(osi, factor=1)
    lp_vload = o3.pattern.Plain(osi, ts_vload)
    v_curr = 0
    o3.Load(osi, nodes[2], [0, v_curr])
    o3.Load(osi, nodes[3], [0, v_curr])
    for i in range(len(disps)):
        d_inc_i = d_incs[i]
        n = int(max(abs(d_inc_i / target_d_inc), min_n))
        d_step = d_inc_i / n

        for j in range(n):
            o3.integrator.DisplacementControl(osi, nodes[2], o3.cc.DOF2D_X, -d_step)
            # o3.SP(osi, nodes[2], dof=o3.cc.X, dof_values=[1])
            # o3.SP(osi, nodes[3], dof=o3.cc.X, dof_values=[1])
            fail = o3.analyze(osi, 1)
            if fail:
                print('Model failed')
                break
            o3.gen_reactions(osi)
            stresses = o3.get_ele_response(osi, ele, 'stress')
            if verbose:
                print('stresses: ', stresses)
            v_eff.append(stresses[1])
            h_eff.append(stresses[0])
            force0 = o3.get_node_reaction(osi, nodes[0], o3.cc.DOF2D_X)
            force1 = o3.get_node_reaction(osi, nodes[1], o3.cc.DOF2D_X)
            # stress.append(-force0 - force1)
            stress.append(stresses[2])
            end_strain = o3.get_node_disp(osi, nodes[2], dof=o3.cc.DOF2D_X)
            print('y_disp: ', o3.get_node_disp(osi, nodes[2], dof=o3.cc.DOF2D_Y))
            print('y_force: ', o3.get_node_reaction(osi, nodes[2], o3.cc.Y), o3.get_node_reaction(osi, nodes[3], o3.cc.Y))
            strain.append(end_strain)
            o3.remove_load_pattern(osi, lp_vload)
            ts_vload = o3.time_series.Constant(osi, factor=1)
            lp_vload = o3.pattern.Plain(osi, ts_vload)
            v_curr += (esig_v0 + v_eff[-1]) / 2
            print('v_eff: ', v_eff[-1], (esig_v0 + v_eff[-1]), v_curr)
            o3.Load(osi, nodes[2], [0, v_curr])
            o3.Load(osi, nodes[3], [0, v_curr])

    o3.wipe()
    all_stresses = all_stresses_cache.collect()
    all_strains = all_strains_cache.collect()
    all_params = all_params_cache.collect()
    return -all_stresses[:, 2], -all_strains[:, 2], all_stresses[:, 0], all_stresses[:, 1], all_params, exit_code



if __name__ == '__main__':
    # example_of_run_ud_custom_strain(show=1)
    # example_of_run_ts_custom_strain(show=1)
    # example_of_run_ts_custom_strain_w_dss(show=1)
    example_of_run_ts_custom_strain(show=1)
    # example_compare_ts_loader_and_ud_loader(show=1)
