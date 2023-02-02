import eqsig
import numpy as np

import o3plot
import os

import sfsimodels as sm
import o3soil


def run(out_folder, dytime=None):

    xi = 0.05

    sl = sm.Soil()
    sl.type = 'pimy'
    vs = 160.
    unit_mass = 1700.0
    sl.cohesion = 58.0e3
    sl.phi = 0.0
    sl.g_mod = vs ** 2 * unit_mass
    sl.poissons_ratio = 0.0
    sl.phi = 0.0
    sl.unit_dry_weight = unit_mass * 9.8
    sl.specific_gravity = 2.65
    sl.peak_strain = 0.01  # set additional parameter required for PIMY model
    ref_press = 100.e3
    sl.xi = 0.03  # for linear analysis
    sl.sra_type = 'hyperbolic'
    o3soil.backbone.set_params_from_op_pimy_model(sl, ref_press)
    sl.inputs += ['strain_curvature', 'xi_min', 'sra_type', 'strain_ref', 'peak_strain']
    assert np.isclose(vs, sl.get_shear_vel(saturated=False))
    sp = sm.SoilProfile()
    sp.add_layer(0, sl)

    sl = sm.Soil()
    sl.type = 'pimy'
    vs = 350.
    unit_mass = 1700.0
    sl.g_mod = vs ** 2 * unit_mass
    sl.poissons_ratio = 0.0
    sl.cohesion = 395.0e3
    sl.phi = 0.0
    sl.unit_dry_weight = unit_mass * 9.8
    sl.specific_gravity = 2.65
    sl.peak_strain = 0.1  # set additional parameter required for PIMY model
    sl.xi = 0.03  # for linear analysis
    sl.sra_type = 'hyperbolic'
    o3soil.backbone.set_params_from_op_pimy_model(sl, ref_press)
    sl.inputs += ['strain_curvature', 'xi_min', 'sra_type', 'strain_ref', 'peak_strain']
    sp.add_layer(8.5, sl)
    sp.height = 14.0

    ecp_out = sm.Output()
    ecp_out.add_to_dict(sp)
    ecp_out.to_file('ecp_sp_w_hload.json')
    import tests.conftest
    record_filename = 'short_motion_dt0p01.txt'
    asig = eqsig.load_asig(tests.conftest.TEST_DATA_DIR + record_filename, m=2.5)
    if dytime is None:
        ind = None
    else:
        ind = int(dytime / asig.dt)
    asig = eqsig.AccSignal(asig.values[:ind], asig.dt)

    outs = {
        'ACCX': 'all',
        'TAU': 'all',
        'STRS': 'all'
    }
    # cache = 0
    # if not cache:
    #     outputs = site_response(soil_profile, asig, xi=xi, out_folder=out_folder, outs=outs, rec_dt=asig.dt, analysis_time=dytime, fixed_base=1)
    # else:
    #     o3sra_outs = o3ptools.O3SRAOutputs()
    #     outputs = o3sra_outs.load_results_from_files()

    show = 1
    if show:
        import matplotlib.pyplot as plt
        bf, sps = plt.subplots(nrows=3)
        # TODO: Show loads on playback
        # TODO: add material to playback, and show legend with material type, set material.__str__ as basetype, params[2:]
        sra1d = o3soil.sra.run_sra(sp, asig, xi=xi, cache_path=out_folder, outs=outs,
                                   analysis_time=dytime, base_imp=-1, playback=True, opfile='run_pimy.py')
        outputs = sra1d.out_dict
        import pandas as pd
        df = pd.DataFrame.from_dict(outputs['TAU'])
        df.to_csv('tau.csv', index=False)
        ind_3m = sra1d.get_nearest_ele_layer_at_depth(3)
        ind_6m = sra1d.get_nearest_ele_layer_at_depth(6)
        ind_12m = sra1d.get_nearest_ele_layer_at_depth(12)
        sps[0].plot(outputs["time"], outputs["TAU"][ind_3m], ls='--')
        sps[0].plot(outputs["time"], outputs["TAU"][ind_6m], ls='--', c='r')
        sps[0].plot(outputs["time"], outputs["TAU"][ind_12m], ls='--')
        sps[2].plot(outputs["time"], outputs["STRS"][ind_6m], ls='--', c='r')
        sps[1].plot(outputs['STRS'][ind_6m], outputs['TAU'][ind_6m], c='r')
        # sps[1].plot(outputs['STRS'][3], sl.g_mod / 1e3 * outputs['STRS'][3], ls='--')
        # sps[2].plot(outputs["time"], outputs["ACCX"][5], ls='--')
        plt.show()

# TODO: how does ele2node know which element it is referring to? it doesnt
if __name__ == '__main__':
    name = __file__.replace('.py', '')
    name = name.split("example_")[-1]
    OP_PATH = "temp/"
    out_folder = OP_PATH + name + '/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    run(dytime=2, out_folder=out_folder)
    import o3seespy as o3
    # o3res = o3.results.Results2D(cache_path=out_folder, dynamic=True)
    # o3res.load_from_cache()
    # o3plot.replot(o3res, xmag=0.5, t_scale=1)
