import o3seespy as o3
import math


def run_ud_cdss(mat, esig_v0, csr, osi=None, static_bias=0.0, n_lim=100, nu_dyn=None, opyfile=None,
                strain_limit=0.03, strain_inc=5.0e-6, verbose=0):
    """Undrained cyclic simple shear test for 2d element"""
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
    bl_node = o3.node.Node(osi, 0, 0)
    br_node = o3.node.Node(osi, h_ele, 0)
    tr_node = o3.node.Node(osi, h_ele, h_ele)
    tl_node = o3.node.Node(osi, 0, h_ele)
    all_nodes = [bl_node, br_node, tr_node, tl_node]

    # Fix bottom node
    o3.Fix3DOF(osi, bl_node, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix3DOF(osi, br_node, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
    o3.Fix3DOF(osi, tr_node, o3.cc.FREE, o3.cc.FREE, o3.cc.FIXED)
    o3.Fix3DOF(osi, tl_node, o3.cc.FREE, o3.cc.FREE, o3.cc.FIXED)
    # Set out-of-plane DOFs to be slaved
    o3.EqualDOF(osi, tr_node, tl_node, [o3.cc.X, o3.cc.Y])

    # Note water bulk modulus, permeability, void ratio are irrelevant, since constant volume test
    # - so as soil skeleton contracts
    # the bulk modulus of the soil skeleton controls the change in effective stress
    water_bulk_mod = 2.2e6
    ele = o3.element.SSPquadUP(osi, all_nodes, mat, 1.0, water_bulk_mod, 1.,
                                1.0e-4, 1.0e-4, 0.6, alpha=1.0e-5, b1=0.0, b2=0.0)

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    o3.integrator.Newmark(osi, gamma=5./6, beta=4./9)
    o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Transient(osi)

    o3.update_material_stage(osi, mat, stage=0)
    # print('here1: ', o3.get_ele_response(osi, ele, 'stress'), esig_v0, csr)

    all_stresses_cache = o3.recorder.ElementToArrayCache(osi, ele, arg_vals=['stress'])
    all_strains_cache = o3.recorder.ElementToArrayCache(osi, ele, arg_vals=['strain'])
    nodes_cache = o3.recorder.NodesToArrayCache(osi, all_nodes, dofs=[1, 2, 3], res_type='disp')
    o3.recorder.NodesToFile(osi, 'node_disp.txt', all_nodes, dofs=[1, 2, 3], res_type='disp')

    # Add static vertical pressure and stress bias
    time_series = o3.time_series.Path(osi, time=[0, 100, 1e10], values=[0, 1, 1])
    o3.pattern.Plain(osi, time_series)
    o3.Load(osi, tl_node, [0, -esig_v0 / 2, 0])
    o3.Load(osi, tr_node, [0, -esig_v0 / 2, 0])

    o3.analyze(osi, num_inc=110, dt=1)

    ts2 = o3.time_series.Path(osi, time=[110, 80000, 1e10], values=[1., 1., 1.], factor=1)
    o3.pattern.Plain(osi, ts2, fact=1.)
    y_vert = o3.get_node_disp(osi, tr_node, o3.cc.Y)
    o3.SP(osi, tl_node, dof=o3.cc.Y, dof_values=[y_vert])
    o3.SP(osi, tr_node, dof=o3.cc.Y, dof_values=[y_vert])

    # Close the drainage valves
    for node in all_nodes:
        o3.remove_sp(osi, node, dof=3)
    o3.analyze(osi, 25, dt=1)
    if verbose:
        print('Finished applying vert load: ', o3.get_ele_response(osi, ele, 'stress'))

    if hasattr(mat, 'update_to_nonlinear'):
        mat.update_to_nonlinear()
        o3.analyze(osi, 25, dt=1)
    if hasattr(mat, 'set_first_call'):
        mat.set_first_call(value=0, ele=ele)
    # o3.set_parameter(osi, value=0, eles=[ele], args=['FirstCall', mat.tag])
    o3.analyze(osi, 25, dt=1)
    if nu_dyn is not None:
        mat.set_nu(nu_dyn, ele=ele)

    ro = o3.recorder.load_recorder_options()
    import pandas as pd
    df = pd.read_csv(ro)
    mat_type = ele.mat.type
    oop = o3.cc.PLANE_STRAIN
    dfe = df[(df['mat'] == mat_type) & (df['form'] == oop)]
    df_sxy = dfe[dfe['recorder'] == 'stress']
    outs = df_sxy['outs'].iloc[0].split('-')
    sxy_ind = outs.index('sxy')

    n_cyc = 0.0
    target_strain = 1.1 * strain_limit
    target_disp = target_strain * h_ele
    limit_reached = 0
    o3.record(osi)
    while n_cyc < n_lim:
        if verbose:
            print('n_cyc: ', n_cyc)
        h_disp = o3.get_node_disp(osi, tr_node, o3.cc.X)
        curr_time = o3.get_time(osi)
        steps = target_strain / strain_inc
        ts0 = o3.time_series.Path(osi, time=[curr_time, curr_time + steps, 1e10], values=[h_disp, target_disp, target_disp], factor=1)
        pat0 = o3.pattern.Plain(osi, ts0)
        o3.SP(osi, tr_node, dof=o3.cc.X, dof_values=[1.0])
        curr_stress = o3.get_ele_response(osi, ele, 'stress')[sxy_ind]
        if math.isnan(curr_stress):
            raise ValueError

        if opyfile:
            o3.extensions.to_py_file(osi, opyfile)
            opyfile = None
        while curr_stress < (csr - static_bias) * esig_v0:
            o3.analyze(osi, 1, dt=1)
            curr_stress = o3.get_ele_response(osi, ele, 'stress')[sxy_ind]
            h_disp = o3.get_node_disp(osi, tr_node, o3.cc.X)

            if h_disp >= target_disp:
                if verbose:
                    print('STRAIN LIMIT REACHED - on load')
                limit_reached = 1
                break
        if limit_reached:
            break
        n_cyc += 0.25
        if verbose:
            print('load reversal, n_cyc: ', n_cyc)
        curr_time = o3.get_time(osi)
        o3.remove_load_pattern(osi, pat0)
        o3.remove(osi, ts0)
        o3.remove_sp(osi, tr_node, dof=o3.cc.X)
        # Reverse cycle
        steps = (h_disp + target_disp) / (strain_inc * h_ele)
        ts0 = o3.time_series.Path(osi, time=[curr_time, curr_time + steps, 1e10],
                                   values=[h_disp, -target_disp, -target_disp], factor=1)
        pat0 = o3.pattern.Plain(osi, ts0)
        o3.SP(osi, tr_node, dof=o3.cc.X, dof_values=[1.0])
        i = 0
        while curr_stress > -(csr + static_bias) * esig_v0:
            o3.analyze(osi, 1, dt=1)
            curr_stress = o3.get_ele_response(osi, ele, 'stress')[2]
            h_disp = o3.get_node_disp(osi, tr_node, o3.cc.X)

            if -h_disp >= target_disp:
                if verbose:
                    print('STRAIN LIMIT REACHED - on reverse')
                limit_reached = 1
                break
            i += 1
            if i > steps:
                break
        if limit_reached:
            break
        n_cyc += 0.5
        if verbose:
            print('reload, n_cyc: ', n_cyc)
        curr_time = o3.get_time(osi)
        o3.remove_load_pattern(osi, pat0)
        o3.remove(osi, ts0)
        o3.remove_sp(osi, tr_node, dof=o3.cc.X)
        # reload cycle
        steps = (-h_disp + target_disp) / (strain_inc * h_ele)
        ts0 = o3.time_series.Path(osi, time=[curr_time, curr_time + steps, 1e10],
                                   values=[h_disp, target_disp, target_disp], factor=1)
        pat0 = o3.pattern.Plain(osi, ts0)
        o3.SP(osi, tr_node, dof=o3.cc.X, dof_values=[1.0])
        while curr_stress < static_bias * esig_v0:
            o3.analyze(osi, 1, dt=1)
            curr_stress = o3.get_ele_response(osi, ele, 'stress')[sxy_ind]
            h_disp = o3.get_node_disp(osi, tr_node, o3.cc.X)

            if h_disp >= target_disp:
                if verbose:
                    print('STRAIN LIMIT REACHED - on reload')
                limit_reached = 1
                break
        if limit_reached:
            break
        o3.remove_load_pattern(osi, pat0)
        o3.remove(osi, ts0)
        o3.remove_sp(osi, tr_node, dof=o3.cc.X)
        n_cyc += 0.25

    o3.wipe(osi)
    all_stresses = all_stresses_cache.collect()
    all_strains = all_strains_cache.collect()
    disps = nodes_cache.collect()
    stress = all_stresses[:, 2]
    strain = all_strains[:, 2]
    ppt = all_stresses[:, 1]

    return stress, strain, ppt, disps


def run_example(show=0):
    import o3seespy as o3

    esig_v0 = 101.3

    # PM4Sand properties
    relative_density = 0.35
    g0_mod = 476.0
    h_po = 0.53
    unit_sat_mass = 1.42
    k0 = 0.5
    nu_dyn = 0.3
    p_atm = 101.0e3
    strain_inc = 5.e-6
    csr = 0.16

    # Initialise OpenSees instance
    osi = o3.OpenSeesInstance(ndm=2, ndf=3, state=3)

    # Define material
    nu_init = k0 / (1 + k0)
    mat = o3.nd_material.PM4Sand(osi, relative_density, g0_mod, h_po, unit_sat_mass, p_atm, nu=nu_init)
    # mat = o3.nd_material.ElasticIsotropic(osi, e_mod=1.0e10, nu=0.3)  # TODO: not working with the elastic model!!!
    # nu_dyn = None
    stress, strain, ppt, disps = run_ud_cdss(mat, csr=csr, osi=osi, n_lim=20, strain_limit=0.03, nu_dyn=nu_dyn,
                                                esig_v0=esig_v0, strain_inc=strain_inc, opyfile='ss.py', verbose=0)

    if show:
        import matplotlib.pyplot as plt
        bf, sps = plt.subplots(nrows=2)
        sps[0].plot(stress, label='shear')
        sps[0].plot(ppt, label='PPT')
        sps[1].plot(strain, stress, label='o3seespy')

        sps[0].set_xlabel('Time [s]')
        sps[0].set_ylabel('Stress [kPa]')
        sps[1].set_xlabel('Strain')
        sps[1].set_ylabel('Stress [kPa]')
        sps[0].legend()
        sps[1].legend()

        plt.show()


if __name__ == '__main__':
    run_example(show=1)
