import pysra
import numpy as np
import matplotlib.pyplot as plt
import sfsimodels as sm
import o3seespy as o3
import o3soil.drivers.two_d as d2d


def set_params_from_op_pimy_model(sl, strain_max, p_ref=100.0e3, hyp=True):
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

    strain_r = strain_max * tau_f / (g_mod_r * strain_max - tau_f)
    sdf = (p_ref / p_ref) ** d
    if hyp:  # hyperbolic model parameters
        sl.strain_curvature = 1.0
        sl.xi_min = 0.01
        sl.strain_ref = strain_r / sdf
        sl.sra_type = "hyperbolic"

    sl.p_ref = p_ref
    sl.g_mod_ref = g_mod_r
    b_mod = 2 * g_mod_r * (1 + sl.poissons_ratio) / (3 * (1 - 2 * sl.poissons_ratio))
    sl.bulk_mod_ref = b_mod



# def calc_hyperbolic_params_for_op_pimy_model(sl, strain_max, p_ref=100.0, esig_v0=100., ndm=2):
#     if ndm == 2:
#         k0 = 1 - np.sin(sl.phi_r)
#         p_eff = (esig_v0 * (1 + 1 * k0) / 2)
#     else:
#         k0 = sl.poissons_ratio / (1. - sl.poissons_ratio)
#         p_eff = (esig_v0 * (1 + 2 * k0) / 3)
#     # Octahedral shear stress
#     tau_f = (2 * np.sqrt(2.) * np.sin(sl.phi_r)) / (3 - np.sin(sl.phi_r)) * p_eff + 2 * np.sqrt(2.) / 3 * sl.cohesion
#     if hasattr(sl, 'get_g_mod_at_v_eff_stress'):
#         d = sl.a
#         g_mod_r = sl.g0_mod * (p_eff / p_ref) ** d
#     else:
#         g_mod_r = sl.g_mod
#         d = 0.0
#
#     strain_r = strain_max * tau_f / (g_mod_r * strain_max - tau_f)
#     sdf = (p_ref / p_eff) ** d
#     return strain_r / sdf, 1.0


def calc_backbone_op_pimy_model(sl, strain_max, strains, p_ref=100.0e3, esig_v0=100., ndm=2):
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
    strain_r = strain_max * tau_f_ref / (g_mod_r * strain_max - tau_f_ref) * dss_eq
    tau_back = g_init * strains / ((1 + strains / strain_r) * (p_ref / p_eff) ** d)
    return tau_back


def create():
    sl = sm.Soil()
    vs = 200.
    unit_mass = 1700.0
    sl.cohesion = 68.0e3
    # sl.cohesion = 0.0
    sl.phi = 0.0
    sl.g_mod = 68.0e6
    print('G_mod: ', sl.g_mod)
    sl.unit_dry_weight = unit_mass * 9.8
    sl.specific_gravity = 2.65
    k0 = 1.0
    # sl.poissons_ratio = k0 / (1 + k0) - 0.01
    sl.poissons_ratio = 0.3
    sl.peak_strain = 0.01
    strains = np.logspace(-4, -1.5, 40)
    esig_v0 = 100.0e3
    ref_press = 100.e3
    ndm = 2
    # TODO: phi and cohesion are not used as you would expect for user defined surfaces !!! recalculated: tau_max calculated the
    set_params_from_op_pimy_model(sl, sl.peak_strain, ref_press)
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
        dss_eq = np.sqrt(3. / 2)
        pysra_sl = pysra.site.ModifiedHyperbolicSoilType("", 1, strain_ref=sl.strain_ref * dss_eq,
                                                         curvature=sl.strain_curvature,
                                                         damping_min=sl.xi_min,
                                                         strains=strains)
        pysra_tau = pysra_sl.mod_reduc.values * sl.g_mod * pysra_sl.mod_reduc.strains
        taus = calc_backbone_op_pimy_model(sl, sl.peak_strain, strains, ref_press, ndm=ndm, esig_v0=esig_v0)
        osi = o3.OpenSeesInstance(ndm=2, ndf=2, state=3)
        # See example: https://opensees.berkeley.edu/wiki/index.php/PressureIndependMultiYield_Material
        base_mat = o3.nd_material.PressureIndependMultiYield(osi,
                                                             nd=2,
                                                             rho=sl.unit_sat_mass,
                                                             g_mod_ref=sl.g_mod_ref,
                                                             bulk_mod_ref=sl.bulk_mod_ref,
                                                             peak_strain=sl.peak_strain,
                                                             cohesion=sl.cohesion,
                                                             phi=sl.phi,
                                                             p_ref=sl.p_ref,
                                                             d=0.0,
                                                             n_surf=25
                                                             )

        disps = np.array([0.0, 0.00003, -0.00003, 0.0004, 0.0001, 0.0009, -0.0009]) * 10
        disps = np.linspace(0.0, 0.02, 100)
        k0_init = sl.poissons_ratio / (1 - sl.poissons_ratio)
        print(k0_init)
        stress, strain, v_eff, h_eff, exit_code = d2d.run_2d_strain_driver_iso(osi, base_mat, esig_v0=esig_v0, disps=disps,
                                                                       handle='warn', verbose=1, target_d_inc=0.00001)
        bf, sps = plt.subplots(nrows=2)
        sps[0].plot(strain, stress, c='r')
        sps[0].plot(strains[:-1], taus[:-1], label='approx PIMY')
        sps[0].plot(pysra_sl.mod_reduc.strains[:-1], pysra_tau[:-1], ls='--', label='PySRA Hyperbolic')
        # sps[1].plot(stress)
        sps[1].plot(v_eff)
        sps[1].plot(h_eff)
        sps[0].axhline(sl.cohesion, c='k')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    create()