import o3seespy as o3
import numpy as np


def load_element(osi, mat, v_pressure, nu_dyn=None):

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
    o3.integrator.DisplacementControl(osi, nodes[2], o3.cc.DOF2D_Y, 0.005)
    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Static(osi)
    if hasattr(mat, 'update_to_nonlinear'):
        mat.update_to_nonlinear(osi)
    if nu_dyn is not None:
        mat.set_poissons_ratio(osi, nu_dyn, ele=ele)

    ods = {'stresses': o3.recorder.ElementToArrayCache(osi, ele=ele, arg_vals=['stress']),
           'strains': o3.recorder.ElementToArrayCache(osi, ele=ele, arg_vals=['strain'])
           }

    ts0 = o3.time_series.Path(osi, time=[0, 1000, 1e10],
                              values=[0, 1, 1], factor=1)
    pat0 = o3.pattern.Plain(osi, ts0)
    o3.Load(osi, nodes[2], [0, -v_pressure * h_ele / 2])
    o3.Load(osi, nodes[3], [0, -v_pressure * h_ele / 2])
    stresses = o3.get_ele_response(osi, ele, 'stress')
    for i in range(1000):
        o3.analyze(osi, num_inc=1)
        stresses = o3.get_ele_response(osi, ele, 'stress')
        if stresses[1] >= v_pressure:
            break
    o3.wipe(osi)
    for item in ods:
        ods[item] = ods[item].collect()
    return ods['stresses'], ods['strains']


def run_example():
    osi = o3.OpenSeesInstance(ndm=2, ndf=2, state=3)

    import matplotlib.pyplot as plt
    esig_v0 = 50.0e3
    poissons_ratio = 0.3
    g_mod = 1.0e6
    b_mod = 2 * g_mod * (1 + poissons_ratio) / (3 * (1 - 2 * poissons_ratio))

    mat = o3.nd_material.PressureIndependMultiYield(osi, 2, 2058.49, g_mod, b_mod, 68000.0, 0.1, 0.0, 100000.0, 0.0, 25)
    # mat = o3.nd_material.ElasticIsotropic(osi, e_mod=1.0e6, nu=0.3)
    ss, es = load_element(osi, mat, v_pressure=esig_v0)
    print(ss[-1])
    plt.plot(es[:, 1], ss[:, 1])
    plt.show()


if __name__ == '__main__':
    run_example()
