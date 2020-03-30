import numpy as np


def set_params_from_op_pimy_model(sl, p_ref=100.0e3, hyp=True):
    # Octahedral shear stress
    tau_f = (2 * np.sqrt(2.) * np.sin(sl.phi_r)) / (3 - np.sin(sl.phi_r)) * p_ref + 2 * np.sqrt(2.) / 3 * sl.cohesion
    if hasattr(sl, 'get_g_mod_at_m_eff_stress'):
        g_mod_r = sl.get_g_mod_at_m_eff_stress(p_ref)
        if sl.phi == 0.0:
            d = 0.
        else:
            d = sl.a
    else:
        g_mod_r = sl.g_mod
        d = 0.0
    print('tau_f: ', tau_f)
    print('cohesion: ', sl.cohesion)
    strain_r = sl.peak_strain * tau_f / (g_mod_r * sl.peak_strain - tau_f)
    sdf = (p_ref / p_ref) ** d
    if hyp:  # hyperbolic model parameters
        sl.strain_curvature = 1.0
        sl.xi_min = 0.01
        dss_eq = 1.  # np.sqrt(3. / 2)  # correct to direct simple shear equivalent
        sl.strain_ref = strain_r / sdf / dss_eq
        sl.sra_type = "hyperbolic"

    sl.p_ref = p_ref
    sl.g_mod_ref = g_mod_r
    b_mod = 2 * g_mod_r * (1 + sl.poissons_ratio) / (3 * (1 - 2 * sl.poissons_ratio))
    sl.bulk_mod_ref = b_mod


def calc_backbone_op_pimy_model(sl, strains, p_ref=100.0e3, esig_v0=100., ndm=2):
    k0 = sl.poissons_ratio / (1. - sl.poissons_ratio)
    p_eff = (esig_v0 * (1 + 2 * k0) / 3)
    # Octahedral shear stress
    tau_f = (2 * np.sqrt(2.) * np.sin(sl.phi_r)) / (3 - np.sin(sl.phi_r)) * p_eff + 2 * np.sqrt(2.) / 3 * sl.cohesion
    tau_f_ref = (2 * np.sqrt(2.) * np.sin(sl.phi_r)) / (3 - np.sin(sl.phi_r)) * p_ref + 2 * np.sqrt(2.) / 3 * sl.cohesion
    if hasattr(sl, 'get_g_mod_at_v_eff_stress'):
        g_init = sl.get_g_mod_at_v_eff_stress(esig_v0)

        if sl.phi == 0.0:
            d = 0.
        else:
            d = sl.a
        g_mod_r = sl.g0_mod * (p_eff / p_ref) ** d
    else:
        g_init = sl.g_mod
        g_mod_r = sl.g_mod
        d = 0.0

    dss_eq = np.sqrt(3. / 2)  # correct to direct simple shear equivalent
    strain_r = sl.peak_strain * tau_f_ref / (g_mod_r * sl.peak_strain - tau_f_ref) * dss_eq
    tau_back = g_init * strains / ((1 + strains / strain_r) * (p_ref / p_eff) ** d)
    return tau_back