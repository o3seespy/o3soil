import pysra
import numpy as np
import matplotlib.pyplot as plt
import sfsimodels as sm


def set_hyperbolic_params_from_op_pimy_model(sl, esig_v0, strain_max):
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


def calc_hyperbolic_params_for_op_pimy_model(sl, esig_v0, strain_max, strains):
    k0 = 1 - np.sin(sl.phi_r)
    p_eff = (esig_v0 * (1 + 1 * k0) / 2)
    # Octahedral shear stress
    tau_f = (2 * np.sqrt(2.) * np.sin(sl.phi_r)) / (3 - np.sin(sl.phi_r)) * p_eff + 2 * np.sqrt(2.) / 3 * sl.cohesion
    if hasattr(sl, 'get_g_mod_at_v_eff_stress'):
        g_init = sl.get_g_mod_at_v_eff_stress(esig_v0)
        g_mod_r = sl.g0_mod * sl.p_atm
        d = sl.a
        p_r = sl.p_atm
    else:
        g_init = sl.g_mod
        g_mod_r = sl.g_mod
        d = 0.0
        p_r = 1

    strain_r = strain_max * tau_f / (g_mod_r * strain_max - tau_f)
    sdf = (p_r / p_eff) ** d
    return strain_r / sdf, 1.0


def calc_backbone_op_pimy_model(sl, esig_v0, strain_max, strains):
    k0 = 1 - np.sin(sl.phi_r)
    p_eff = (esig_v0 * (1 + 1 * k0) / 2)
    # Octahedral shear stress
    tau_f = (2 * np.sqrt(2.) * np.sin(sl.phi_r)) / (3 - np.sin(sl.phi_r)) * p_eff + 2 * np.sqrt(2.) / 3 * sl.cohesion
    if hasattr(sl, 'get_g_mod_at_v_eff_stress'):
        g_init = sl.get_g_mod_at_v_eff_stress(esig_v0)
        g_mod_r = sl.g0_mod * sl.p_atm
        d = sl.a
        p_r = sl.p_atm
    else:
        g_init = sl.g_mod
        g_mod_r = sl.g_mod
        d = 0.0
        p_r = 1

    strain_r = strain_max * tau_f / (g_mod_r * strain_max - tau_f)
    tau_back = g_init * strains / ((1 + strains / strain_r) * (p_r / p_eff) ** d)
    return tau_back


def create():
    sl = sm.Soil()
    vs = 200.
    unit_mass = 1700.0
    sl.cohesion = 68.0e3
    sl.phi = 0.0
    sl.g_mod = vs ** 2 * unit_mass
    print('G_mod: ', sl.g_mod)
    sl.unit_dry_weight = unit_mass * 9.8
    sl.id = 1
    assert np.isclose(vs, sl.get_shear_vel(saturated=False))
    strain_max = 0.05
    strains = np.logspace(-4, -1.5, 40)
    set_hyperbolic_params_from_op_pimy_model(sl, 1, strain_max)
    sl.inputs += ['strain_curvature', 'xi_min', 'sra_type', 'strain_ref']

    sp = sm.SoilProfile()
    sp.id = 1
    sp.add_layer(0, sl)
    sp.height = 30.0
    ecp_out = sm.Output()
    ecp_out.add_to_dict(sp)
    ofile = open('ecp.json', 'w')
    # ofile.write(json.dumps(ecp_out.to_dict(), indent=4))
    ofile.close()

    compare_backbone = 1
    if compare_backbone:
        pysra_sl = pysra.site.ModifiedHyperbolicSoilType("", 1, strain_ref=sl.strain_ref,
                                                         curvature=sl.strain_curvature,
                                                         damping_min=sl.xi_min,
                                                         strains=strains)
        pysra_tau = pysra_sl.mod_reduc.values * sl.g_mod * pysra_sl.mod_reduc.strains
        taus = calc_backbone_op_pimy_model(sl, 1, strain_max, strains)
        plt.plot(strains, taus, label='approx PIMY')
        plt.plot(pysra_sl.mod_reduc.strains, pysra_tau, ls='--', label='PySRA Hyperbolic')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    create()