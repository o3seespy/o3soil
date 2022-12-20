import o3seespy as o3


def get_o3_class_and_args_from_soil_obj(sl, saturated=False, esig_v0=None, f_order=1e3, overrides=None):
    import numpy as np
    if overrides is None:
        overrides = {}

    if saturated:
        umass = sl.unit_sat_mass / f_order  # TODO: work out how to run in Pa, N, m, s
    else:
        umass = sl.unit_dry_mass / f_order
    if 'p_atm' not in overrides:
        if not hasattr(sl, 'p_atm'):
            p_atm = 101.0e3
        else:
            p_atm = sl.p_atm
        overrides['p_atm'] = p_atm / f_order
    if 'rho' not in overrides:
        overrides['rho'] = umass
    if 'unit_moist_mass' not in overrides:
        overrides['unit_moist_mass'] = umass
    if 'nd' not in overrides:
        overrides['nd'] = 2

    app2mod = {}
    # Define material
    if not hasattr(sl, 'o3_type'):
        sl.o3_type = sl.type  # for backward compatibility
    if sl.o3_type == 'pm4sand':
        sl_class = o3.nd_material.PM4Sand
        app2mod = sl.app2mod
    elif sl.o3_type == 'sdmodel':
        sl_class = o3.nd_material.StressDensity
        # overrides = {'nu': pois, 'p_atm': 101, 'unit_moist_mass': umass}
        app2mod = sl.app2mod
    elif sl.o3_type in ['pimy', 'pdmy', 'pdmy02']:
        if hasattr(sl, 'get_g_mod_at_m_eff_stress'):
            if hasattr(sl, 'g_mod_p0') and sl.g_mod_p0 != 0.0:
                k0 = sl.poissons_ratio / (1 - sl.poissons_ratio)
                m_eff = esig_v0 * (1 + 2 * k0) / 3
                p = m_eff  # Pa
                overrides['d'] = 0.0
            else:
                p = 101.0e3  # Pa
                overrides['d'] = sl.a
            g_mod_r = sl.get_g_mod_at_m_eff_stress(p)
        else:
            p = 101.0e3  # Pa
            overrides['d'] = 0.0
            g_mod_r = sl.g_mod

        b_mod = 2 * g_mod_r * (1 + sl.poissons_ratio) / (3 * (1 - 2 * sl.poissons_ratio))
        overrides['p_ref'] = p / f_order
        overrides['g_mod_ref'] = g_mod_r / f_order
        overrides['bulk_mod_ref'] = b_mod / f_order
        if sl.o3_type == 'pimy':
            if sl.phi != 0:
                raise ValueError('need to deal with strength correction')
            tau_f = 2 * np.sqrt(2.) / 3 * sl.cohesion
            sf = sl.peak_strain * sl.g_mod / (sl.g_mod * sl.peak_strain - tau_f) * np.sqrt(3. / 2) * tau_f / sl.cohesion
            overrides['cohesion'] = sl.cohesion / f_order / sf
            sl_class = o3.nd_material.PressureIndependMultiYield
        elif sl.o3_type == 'pdmy':
            sl_class = o3.nd_material.PressureDependMultiYield
        elif sl.o3_type == 'pdmy02':
            sl_class = o3.nd_material.PressureDependMultiYield02
    else:
        if hasattr(sl, 'get_g_mod_at_v_eff_stress'):
            g_mod = sl.get_g_mod_at_v_eff_stress(esig_v0)
        else:
            g_mod = sl.g_mod
        sl_class = o3.nd_material.ElasticIsotropic
        sl.e_mod = 2 * g_mod * (1 - sl.poissons_ratio)
        overrides['nu'] = sl.poissons_ratio
        app2mod['rho'] = 'unit_moist_mass'
    args, kwargs = o3.extensions.get_o3_kwargs_from_obj(sl, sl_class, custom=app2mod, overrides=overrides)
    return sl_class, args, kwargs
