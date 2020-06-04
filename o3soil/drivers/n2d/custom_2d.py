import o3seespy as o3
import numpy as np
import math


def run_ts_custom_strain(mat, esig_v0, strains, osi=None, nu_dyn=None, target_d_inc=0.00001, handle='silent', verbose=0, opyfile=None):
    k0 = 1.0
    pois = k0 / (1 + k0)
    damp = 0.05
    omega0 = 0.2
    omega1 = 20.0
    a1 = 2. * damp / (omega0 + omega1)
    a0 = a1 * omega0 * omega1
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
    o3.integrator.DisplacementControl(osi, nodes[2], o3.cc.DOF2D_Y, 0.005)
    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Static(osi)
    o3.update_material_stage(osi, mat, stage=0)

    # Add static vertical pressure and stress bias
    # time_series = o3.time_series.Path(osi, time=[0, 100, 1e10], values=[0, 1, 1])
    # o3.pattern.Plain(osi, time_series)
    ts0 = o3.time_series.Linear(osi, factor=1)
    o3.pattern.Plain(osi, ts0)
    o3.Load(osi, nodes[2], [0, -esig_v0 / 2])
    o3.Load(osi, nodes[3], [0, -esig_v0 / 2])

    o3.analyze(osi, num_inc=100)
    stresses = o3.get_ele_response(osi, ele, 'stress')
    print('init_stress0: ', stresses)

    o3.wipe_analysis(osi)
    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    o3.analysis.Static(osi)

    o3.update_material_stage(osi, mat, stage=1)
    o3.analyze(osi, 25, dt=1)
    if nu_dyn is not None:
        mat.set_nu(osi, nu_dyn, eles=[ele])

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
    d_incs = np.diff(strains, prepend=0)
    for i in range(len(strains)):
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
            if opyfile:
                import o3seespy.extensions
                o3.extensions.to_py_file(osi, opyfile)
                opyfile = None
            print(stresses, d_step)
            v_eff.append(stresses[1])
            h_eff.append(stresses[0])
            force0 = o3.get_node_reaction(osi, nodes[0], o3.cc.DOF2D_X)
            force1 = o3.get_node_reaction(osi, nodes[1], o3.cc.DOF2D_X)
            stress.append(-force0 - force1)
            end_strain = o3.get_node_disp(osi, nodes[2], dof=o3.cc.DOF2D_X)
            strain.append(end_strain)

    return -np.array(stress), -np.array(strain), np.array(v_eff), np.array(h_eff), exit_code


def run_custom_strain(osi, mat, esig_v0, disps, nu_dyn=None, target_d_inc=0.00001, handle='silent', verbose=0, opyfile=None):
    # deprecated
    return run_ts_custom_strain(mat, esig_v0, disps, osi, nu_dyn, target_d_inc, handle=handle, verbose=verbose,
                             opyfile=opyfile)


def run_ud_custom_strain(mat, esig_v0, disps, osi=None, nu_dyn=None, target_d_inc=0.00001, handle='silent', verbose=0, opyfile=None):

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
    o3.integrator.DisplacementControl(osi, nodes[2], o3.cc.DOF2D_Y, 0.005)
    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Static(osi)
    o3.update_material_stage(osi, mat, stage=0)

    # Add static vertical pressure and stress bias
    # time_series = o3.time_series.Path(osi, time=[0, 100, 1e10], values=[0, 1, 1])
    # o3.pattern.Plain(osi, time_series)
    time_series = o3.time_series.Path(osi, time=[0, 100, 1e10], values=[0, 1, 1])
    o3.pattern.Plain(osi, time_series)
    o3.Load(osi, nodes[2], [0, -esig_v0 / 2, 0])
    o3.Load(osi, nodes[3], [0, -esig_v0 / 2, 0])

    o3.analyze(osi, num_inc=110, dt=1)
    stresses = o3.get_ele_response(osi, ele, 'stress')
    print('init_stress0: ', stresses)

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

    o3.load_constant(osi, 0.0)

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
    return
    for i in range(len(disps)):
        h_disp = o3.get_node_disp(osi, nodes[2], o3.cc.X)
        curr_time = o3.get_time(osi)
        steps = 100
        ts0 = o3.time_series.Path(osi, time=[curr_time, curr_time + steps, 1e10],
                                  values=[h_disp, disps[i], disps[i]],
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

    return -np.array(stress), -np.array(strain), np.array(v_eff), np.array(h_eff), exit_code


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
    # time_series = o3.time_series.Path(osi, time=[0, 100, 1e10], values=[0, 1, 1])
    # o3.pattern.Plain(osi, time_series)
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
    return
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

    return -np.array(stress), -np.array(strain), np.array(v_eff), np.array(h_eff), exit_code