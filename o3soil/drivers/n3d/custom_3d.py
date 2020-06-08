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
        o3.node.Node(osi, 0.0, 0.0, h_ele), o3.node.Node(osi, h_ele, 0.0, h_ele),  # left-bot-front -> right-bot-front
        o3.node.Node(osi, h_ele, 0.0, 0.0), o3.node.Node(osi, 0.0, 0.0, 0.0),  # right-bot-back -> left-bot-back
        o3.node.Node(osi, 0.0, h_ele, h_ele), o3.node.Node(osi, h_ele, h_ele, h_ele),  # left-top-ft -> right-top-front
        o3.node.Node(osi, h_ele, h_ele, 0.0), o3.node.Node(osi, 0.0, h_ele, 0.0)  # right-top-back -> left-top-back
    ]

    # Fix bottom nodes
    o3.Fix3DOFMulti(osi, nodes[:4], o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)

    # Set out-of-plane DOFs to be slaved
    o3.EqualDOFMulti(osi, nodes[4], nodes[5:], [o3.cc.X, o3.cc.Y, o3.cc.DOF3D_Z])

    ele = o3.element.SSPbrick(osi, nodes, mat, 0.0, 0.0, 0.0)

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    o3.integrator.DisplacementControl(osi, nodes[4], o3.cc.DOF2D_Y, 0.005)
    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Static(osi)
    o3.update_material_stage(osi, mat, stage=0)

    # Add static vertical pressure and stress bias
    ts0 = o3.time_series.Path(osi, time=[0, 1000, 1e10],
                              values=[0, 1, 1], factor=1)
    pat0 = o3.pattern.Plain(osi, ts0)
    o3.Load(osi, nodes[4], [0, -esig_v0 * h_ele / 4, 0])
    o3.Load(osi, nodes[5], [0, -esig_v0 * h_ele / 4, 0])
    o3.Load(osi, nodes[6], [0, -esig_v0 * h_ele / 4, 0])
    o3.Load(osi, nodes[7], [0, -esig_v0 * h_ele / 4, 0])
    o3.record(osi)

    o3.analyze(osi, num_inc=1000)
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
            o3.integrator.DisplacementControl(osi, nodes[4], o3.cc.DOF2D_X, -d_step)
            o3.Load(osi, nodes[2], [1.0, 0.0, 0.0])
            o3.Load(osi, nodes[3], [1.0, 0.0, 0.0])
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
            stress.append(stresses[2])
            end_strain = o3.get_node_disp(osi, nodes[4], dof=o3.cc.DOF2D_X)
            strain.append(end_strain)

    return -np.array(stress), -np.array(strain), np.array(v_eff), np.array(h_eff), exit_code


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
    ihd = o3.get_node_disp(osi, nodes[2], dof=o3.cc.DOF2D_X)  # initial horizontal displacement
    strain = [0]
    stresses = o3.get_ele_response(osi, ele, 'stress')
    force0 = o3.get_node_reaction(osi, nodes[0], o3.cc.DOF2D_X)
    force1 = o3.get_node_reaction(osi, nodes[1], o3.cc.DOF2D_X)
    stress = [-force0 - force1]
    v_eff = [stresses[1]]
    h_eff = [stresses[0]]
    target_disps = np.array(disps) + ihd

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
            stress.append(curr_stresses[2])
            end_strain = o3.get_node_disp(osi, nodes[2], dof=o3.cc.DOF2D_X)
            strain.append(end_strain - ihd)

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


def example_of_run_ts_custom_strain(show=0):
    osi = o3.OpenSeesInstance(ndm=3, ndf=3, state=3)

    esig_v0 = 50.0e3
    poissons_ratio = 0.3
    g_mod = 1.0e6
    b_mod = 2 * g_mod * (1 + poissons_ratio) / (3 * (1 - 2 * poissons_ratio))

    mat = o3.nd_material.PressureIndependMultiYield(osi, 3, 2058.49, g_mod, b_mod, 68000.0, 0.1, 0.0, 100000.0, 0.0, 25)
    # mat = o3.nd_material.ElasticIsotropic(osi, e_mod=1.0e6, nu=0.3)
    ss, es = run_vload(mat, v_pressure=esig_v0, osi=osi)
    if show:
        import matplotlib.pyplot as plt
        plt.plot(es[:, 1], ss[:, 1])
        plt.show()



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

        sps[0].set_xlabel('Time [s]')
        sps[0].set_ylabel('Stress [kPa]')
        sps[1].set_xlabel('Strain')
        sps[1].set_ylabel('Stress [kPa]')
        sps[0].legend()
        sps[1].legend()
        plt.show()


if __name__ == '__main__':
    example_of_run_ud_custom_strain(show=1)