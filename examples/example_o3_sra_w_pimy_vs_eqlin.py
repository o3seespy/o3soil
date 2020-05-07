import o3soil.sra
import sfsimodels as sm
import numpy as np
import eqsig
from tests.conftest import TEST_DATA_DIR

# for linear analysis comparison
import liquepy as lq


def run():
    # Define the soil layers using the sfsimodels package
    sl = sm.Soil()
    sl.type = 'pimy'  # set type to be recognised by o3soil package
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
    soil_profile = sm.SoilProfile()
    soil_profile.add_layer(0, sl)

    sl = sm.Soil()
    sl.type = 'pimy'
    vs = 400.
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
    soil_profile.add_layer(9.5, sl)
    soil_profile.height = 20.0

    # Export the soil profile to json
    ecp_out = sm.Output()
    ecp_out.add_to_dict(soil_profile)
    ecp_out.to_file('ecp.json')
    # Reload the json file to demonstrate save/load capability
    mods = sm.load_json('ecp.json', default_to_base=True)
    soil_profile = mods['soil_profile'][1]

    # Load the ground motion as an eqsig.AccSignal object
    in_sig = eqsig.load_asig(TEST_DATA_DIR + 'short_motion_dt0p01.txt', m=0.2)

    # run equivalent linear analysis with pysra - using the liquepy package
    od = lq.sra.run_pysra(soil_profile, in_sig, odepths=np.array([0.0, 2.0]))
    # save surface acceleration as an eqsig.AccSignal object
    pysra_sig = eqsig.AccSignal(od['ACCX'][0], in_sig.dt)

    # run a nonlinear analysis with o3seespy - using the o3soil package
    outputs = o3soil.sra.site_response(soil_profile, in_sig, xi=0.02, freqs=(0.5, 5))
    # save surface acceleration as an eqsig.AccSignal object
    resp_dt = outputs['time'][2] - outputs['time'][1]
    surf_sig = eqsig.AccSignal(outputs['ACCX'][0], resp_dt)

    show = 1

    if show:
        import matplotlib.pyplot as plt
        from bwplot import cbox

        bf, sps = plt.subplots(nrows=3)

        sps[0].plot(in_sig.time, in_sig.values, c='k', label='Input')
        # sps[0].plot(pysra_sig.time, o3_surf_vals, c=cbox(0), label='o3')
        sps[0].plot(outputs['time'], outputs['ACCX'][0], c=cbox(3), label='o3')
        sps[0].plot(pysra_sig.time, pysra_sig.values, c=cbox(1), label='pysra', ls='--')

        sps[1].plot(in_sig.fa_frequencies, abs(in_sig.fa_spectrum), c='k')
        sps[1].plot(surf_sig.fa_frequencies, abs(surf_sig.fa_spectrum), c=cbox(0))
        sps[1].plot(pysra_sig.fa_frequencies, abs(pysra_sig.fa_spectrum), c=cbox(1), ls='--')
        sps[1].set_xlim([0, 20])
        in_sig.smooth_fa_frequencies = in_sig.fa_frequencies
        surf_sig.smooth_fa_frequencies = in_sig.fa_frequencies
        pysra_sig.smooth_fa_frequencies = in_sig.fa_frequencies
        h = surf_sig.smooth_fa_spectrum / in_sig.smooth_fa_spectrum
        sps[2].plot(surf_sig.smooth_fa_frequencies, h, c=cbox(0))
        pysra_h = pysra_sig.smooth_fa_spectrum / in_sig.smooth_fa_spectrum
        sps[2].plot(pysra_sig.smooth_fa_frequencies, pysra_h, c=cbox(1), ls='--')
        sps[2].axhline(1, c='k', ls='--')
        sps[0].legend()
        plt.show()

    o3_surf_vals = np.interp(pysra_sig.time, surf_sig.time, surf_sig.values)
    assert np.isclose(o3_surf_vals, pysra_sig.values, atol=0.01, rtol=100).all()


if __name__ == '__main__':
    run()

