import eqsig
import numpy as np

import o3plot
import os

import sfsimodels as sm
import o3soil
import o3seespy as o3


def run(out_folder, dytime=None):

    xi = 0.05

    sl = o3.nd_material.PressureDependMultiYield02(None, 2, 1.8, 9.0e4, 2.2e5, 32, peak_strain=0.1,
                                      p_ref=101.0, d=0.5, pt_ang=26, con_rates=[0.067, 5.0, 0.23],
                                      dil_rates=[0.06, 3.0, 0.27], n_surf=20, liquefac=[1.0, 0.0],
                                                   e_init=0.77, cs_params=[0.9, 0.02, 0.7, 101.0])
    sl.is_o3_mat = True
    sl.unit_mass = 0
    sl.permeability = 1e-6
    sl.e_curr = sl.e_init
    sl.dynamic_poissons_ratio = 0.3

    sp = sm.SoilProfile()
    sp.add_layer(0, sl)

    sp.height = 15.0

    # ecp_out = sm.Output()
    # ecp_out.add_to_dict(sp)
    # ecp_out.to_file('ecp_sp_w_hload.json')
    import tests.conftest
    record_filename = 'short_motion_dt0p01.txt'
    asig = eqsig.load_asig(tests.conftest.TEST_DATA_DIR + record_filename, m=.5)
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

    show = 1
    if show:
        import matplotlib.pyplot as plt
        bf, sps = plt.subplots(nrows=3)
        # TODO: Show loads on playback
        # TODO: add material to playback, and show legend with material type, set material.__str__ as basetype, params[2:]
        sra1d = o3soil.sra.run_eff_sra(sp, asig, xi=xi, cache_path=out_folder, outs=outs, opfile='run_pdmy2.py',
                                   analysis_time=dytime, base_imp=-1, playback=True)
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
        sps[2].plot(outputs["time"], outputs["ACCX"][0], ls='--', c='r')
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
    run(dytime=10, out_folder=out_folder)
    import o3seespy as o3
    o3res = o3.results.Results2D(cache_path=out_folder, dynamic=True)
    o3res.load_from_cache()
    o3plot.replot(o3res, xmag=2.0, t_scale=1)
