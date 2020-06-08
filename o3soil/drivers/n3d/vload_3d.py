import o3seespy as o3
import numpy as np


def run_vload(mat, v_pressure, osi=None, nu_dyn=None):
    if osi is None:
        osi = o3.OpenSeesInstance(ndm=3, ndf=3)
        mat.build(osi)
    h_ele = 1.
    nodes = [
        o3.node.Node(osi, 0.0, 0.0, h_ele), o3.node.Node(osi, h_ele, 0.0, h_ele),   # left-bot-front -> right-bot-front
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
    o3.test_check.NormDispIncr(osi, tol=1.0e-3, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    # o3.integrator.DisplacementControl(osi, nodes[5], o3.cc.DOF2D_Y, 0.005)
    o3.integrator.LoadControl(osi, 1)
    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Static(osi)
    from o3seespy import extensions
    o3.extensions.to_py_file(osi, 'ele_3d.py')
    if hasattr(mat, 'update_to_nonlinear'):
        mat.update_to_nonlinear(osi)
    if nu_dyn is not None:
        mat.set_poissons_ratio(osi, nu_dyn, ele=ele)

    ods = {'stresses': o3.recorder.ElementToArrayCache(osi, ele=ele, arg_vals=['stress'], fname='stress.txt'),
           'strains': o3.recorder.ElementToArrayCache(osi, ele=ele, arg_vals=['strain'])
           }

    ts0 = o3.time_series.Path(osi, time=[0, 1000, 1e10],
                              values=[0, 1, 1], factor=1)
    pat0 = o3.pattern.Plain(osi, ts0)
    o3.Load(osi, nodes[4], [0, -v_pressure * h_ele / 4, 0])
    o3.Load(osi, nodes[5], [0, -v_pressure * h_ele / 4, 0])
    o3.Load(osi, nodes[6], [0, -v_pressure * h_ele / 4, 0])
    o3.Load(osi, nodes[7], [0, -v_pressure * h_ele / 4, 0])
    o3.record(osi)
    stresses = o3.get_ele_response(osi, ele, 'stress')
    print(stresses)
    for i in range(1000):
        o3.analyze(osi, num_inc=1)
        stresses = o3.get_ele_response(osi, ele, 'stress')
        print(stresses)
        if stresses[1] >= v_pressure:
            break
    o3.wipe(osi)
    for item in ods:
        ods[item] = ods[item].collect()
    return ods['stresses'], ods['strains']


def run_example(show=0):
    osi = o3.OpenSeesInstance(ndm=3, ndf=3, state=3)

    esig_v0 = 50.0e3
    poissons_ratio = 0.3
    g_mod = 1.0e6
    b_mod = 2 * g_mod * (1 + poissons_ratio) / (3 * (1 - 2 * poissons_ratio))

    mat = o3.nd_material.PressureIndependMultiYield(osi, 3, 2058.49, g_mod, b_mod, 68000.0, 0.1, 0.0, 100000.0, 0.0, 25)
    mat = o3.nd_material.ElasticIsotropic(osi, e_mod=1.0e6, nu=0.3)
    ss, es = run_vload(mat, v_pressure=esig_v0, osi=osi)
    if show:
        import matplotlib.pyplot as plt
        plt.plot(es[:, 1], ss[:, 1])
        plt.show()


if __name__ == '__main__':
    run_example(show=1)
