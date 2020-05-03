import geofound as gf
import o3seespy as o3
import numpy as np


def set_bnwf2d_via_harden_2009(osi, sl, fd, soil_node, bd_node, axis, dettach=True, soil_nl=True):
    """
    Set a Beam on nonlinear Winker Foundation between two nodes

    Parameters
    ----------
    osi
    sl: sfsimodels.Soil object
    fd: sfsimodels.Foundation object
    soil_node: o3.Node
        The soil node
    bd_node: o3.Node
        The base of the building node
    axis: str
        The axis which the foundation would rotate around
    dettach
    soil_nl

    Returns
    -------

    """
    # TODO: account for foundation height
    end_zone_ratio = 0.3  # TODO: currently based on 0.3, but Harden et al. (2005) showed this to be a ratio of B/L
    k_rot = gf.stiffness.calc_rotational_via_gazetas_1991(sl, fd, axis=axis)
    k_vert = gf.stiffness.calc_vert_via_gazetas_1991(sl, fd)
    if axis == 'length':  # rotation around the length axis
        k_vert_i = k_vert / fd.width / fd.length
    else:
        k_vert_i = k_vert / fd.length / fd.length
    if fd.i_ww >= fd.i_ll:
        len_dominant = True
        l = fd.length * 0.5
        b = fd.width * 0.5
    else:
        len_dominant = False
        l = fd.width * 0.5
        b = fd.length * 0.5
    if (axis == 'length' and len_dominant) or (axis == 'width' and not len_dominant):
        # rotation around x-axis
        r_k = (3 * k_rot / (4 * k_vert_i * b ** 3 * l)) - (1 - end_zone_ratio) ** 3 / (1 - (1 - end_zone_ratio) ** 3)
    else:
        # rotation around y-axis
        r_k = (3 * k_rot / (4 * k_vert_i * b * l ** 3)) - (1 - end_zone_ratio) ** 3 / (1 - (1 - end_zone_ratio) ** 3)

    n_springs = 10
    k_spring = k_vert / n_springs
    # use 10 springs - have the exterior spring
    w = fd.width
    pos = np.linspace(w / 20, fd.width - w / 20, n_springs) - fd.width / 2
    if not soil_nl:
        if dettach:
            k_ten = 1.0e-5 * k_spring
        else:
            k_ten = k_spring
        int_spring_mat = o3.uniaxial_material.Elastic(osi, k_spring, eneg=k_ten)
        ext_spring_mat = o3.uniaxial_material.Elastic(osi, r_k * k_spring, eneg=r_k * k_ten)
    else:
        q_ult = gf.capacity_salgado_2008(sl, fd)
        f_ult = q_ult * fd.area
        f_spring = f_ult / n_springs / 2  # TODO: should exterior be different? -should be divide by n_springs
        int_spring_mat_1 = o3.uniaxial_material.Steel02(osi, f_spring, k_spring, b=0.05, params=[5, 0.925, 0.15])
        ext_spring_mat_1 = o3.uniaxial_material.Steel02(osi, f_spring, r_k * k_spring, b=0.05, params=[5, 0.925, 0.15])
        if dettach:
            mat_obj2 = o3.uniaxial_material.Elastic(osi, 1000 * k_spring, eneg=0.0001 * k_spring)
            int_spring_mat = o3.uniaxial_material.Series(osi, [int_spring_mat_1, mat_obj2])
            ext_spring_mat = o3.uniaxial_material.Series(osi, [ext_spring_mat_1, mat_obj2])
        else:
            int_spring_mat = int_spring_mat_1
            ext_spring_mat = ext_spring_mat_1
    spring_mats = 3 * [ext_spring_mat] + 4 * [int_spring_mat] + 3 * [ext_spring_mat]
    scaler = np.ones_like(pos)
    # scaler[:3] = r_k
    # scaler[-3:] = r_k
    # k_rot_bnwf = np.sum(k_spring * scaler * pos ** 2)
    # print(k_rot, k_rot_bnwf, k_rot_bnwf / k_rot)
    from o3seespy.command.element.soil_foundation import gen_shallow_foundation_bnwf
    fd_area = fd.width * fd.height
    fd_emod = 30.0e9
    fd_iz = fd.width * fd.height ** 3 / 12
    return gen_shallow_foundation_bnwf(osi, soil_node, bd_node, sf_mats=spring_mats, pos=pos, fd_area=fd_area,
                                fd_e_mod=fd_emod, fd_iz=fd_iz)


def generate_example_ssi_system():
    import sfsimodels as sm
    sl = sm.StressDependentSoil()
    sl.e_max = 0.877
    sl.e_min = 0.511
    sl.specific_gravity = 2.65
    sl.relative_density = 0.38
    sl.phi = 32.
    sl.cohesion = 0.0
    cf = 0.77  # Arulmoli et al. (1992)
    sl.g0_mod = 625 * cf * np.sqrt(100e3) / (0.3 + 0.7 * sl.e_curr) * np.sqrt(sl.p_atm) / sl.p_atm
    sl.a = 0.5
    sl.g_mod = 23.5e6
    print('g_below f', sl.get_g_mod_at_m_eff_stress(13.7e3))
    sl.poissons_ratio = 0.3
    sp = sm.SoilProfile()
    sp.add_layer(0, sl)
    # Add steel base
    h_base = 0.183 * 49  # note N=49g
    sp.height = h_base + 0.1
    steel = sm.Soil()
    steel.g_mod = 80e9
    steel.e_curr = 0.0
    steel.xi = 0.01
    steel.poissons_ratio = 0.28
    steel.unit_sat_weight = 76930.0
    sp.add_layer(h_base, steel)

    fd = sm.RaftFoundation()
    fd.width = 7.35
    fd.length = 4.7
    fd.height = 1.24
    fd.depth = 2.24  # m
    fd.mass = 79.0e3  # kg

    bd = sm.SDOFBuilding()
    bd.h_eff = 12.1
    bd.mass_eff = 553.0e3
    bd.xi = 0.05
    bd.t_fixed = 0.05  # s
    bd.inputs.append('xi')
    bd.set_foundation(fd, x=0)
    return bd, sl


def run_example():
    osi = o3.OpenSeesInstance(ndm=2, state=3)
    bd, sl = generate_example_ssi_system()
    fd = bd.fd
    height = bd.h_eff
    # Establish nodes
    bot_node = o3.node.Node(osi, 0, 0)
    top_node = o3.node.Node(osi, 0, height)
    sl_node = o3.node.Node(osi, 0, 0)  # TODO: add fd height

    # Fix bottom node
    o3.Fix3DOF(osi, top_node, o3.cc.FREE, o3.cc.FREE, o3.cc.FREE)
    o3.Fix3DOF(osi, bot_node, o3.cc.FREE, o3.cc.FREE, o3.cc.FREE)
    o3.Fix3DOF(osi, sl_node, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)
    # Set out-of-plane DOFs to be slaved
    # o3.EqualDOF(osi, top_node, bot_node, [o3.cc.Y])

    # nodal mass (weight / g):
    o3.Mass(osi, top_node, bd.mass_eff, 0., 0.)
    o3.Mass(osi, bot_node, fd.mass, 0., 0.)  # TODO: add

    # Define material
    transf = o3.geom_transf.Linear2D(osi, [])
    area = 1.0
    e_mod = 100.0e6
    iz = bd.k_eff * height ** 3 / (3 * e_mod)
    ele_nodes = [bot_node, top_node]

    vert_ele = o3.element.ElasticBeamColumn2D(osi, ele_nodes, area=area, e_mod=e_mod, iz=iz, transf=transf)

    # TODO: if sl.g_mod is stress dependent, then account for foundation load and depth increase using pg2-18 of NIST
    # TODO: Implement the stiffness using soil_profile into geofound that accounts for fd.q_load
    k_shear = gf.stiffness.calc_shear_via_gazetas_1991(sl, fd, axis='length')
    shear_mat = o3.uniaxial_material.Elastic(osi, k_shear)
    # sl.override('g_mod', sl.g_mod)
    soil_fd_ele = o3.element.ZeroLength(osi, [sl_node, bot_node], mats=[shear_mat], dirs=[o3.cc.DOF2D_X])
    bnwf = set_bnwf_via_harden_2009(osi, sl, fd, sl_node, bot_node, axis='width', soil_nl=True, dettach=True)
    import o3seespy.extensions
    o3.extensions.to_py_file(osi)

    # set damping based on first eigen mode
    angular_freq = o3.get_eigen(osi, solver='fullGenLapack', n=1)[0] ** 0.5
    response_period = 2 * np.pi / angular_freq
    print('response_period: ', response_period)
    beta_k = 2 * bd.xi / angular_freq
    o3.rayleigh.Rayleigh(osi, alpha_m=0.0, beta_k=beta_k, beta_k_init=0.0, beta_k_comm=0.0)
    ts0 = o3.time_series.Linear(osi, factor=1)
    o3.pattern.Plain(osi, ts0)
    o3.Load(osi, top_node, [0, -bd.mass_eff * 9.8, 0])
    o3.Load(osi, bot_node, [0, -fd.mass * 9.8, 0])

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    n_steps_gravity = 10
    d_gravity = 1. / n_steps_gravity
    o3.integrator.LoadControl(osi, d_gravity, num_iter=10)
    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Static(osi)
    o3.analyze(osi, num_inc=n_steps_gravity)
    o3.load_constant(osi, time=0.0)
    print('init_disp: ', o3.get_node_disp(osi, bot_node, o3.cc.DOF2D_Y))

    # Start horizontal load
    ts0 = o3.time_series.Linear(osi, factor=1)
    o3.pattern.Plain(osi, ts0)
    o3.Load(osi, top_node, [bd.mass_eff * 2.5, 0, 0])
    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    n_steps_hload = 100
    d_hload = 1. / n_steps_hload
    o3.integrator.LoadControl(osi, d_hload, num_iter=10)
    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Static(osi)
    rot = []
    mom = []
    for i in range(n_steps_hload):
        o3.analyze(osi, num_inc=1)
        rot.append(o3.get_node_disp(osi, bot_node, o3.cc.DOF2D_ROTZ))
        o3.gen_reactions(osi)
        mom.append(o3.get_ele_response(osi, vert_ele, 'force')[2])

    o3.gen_reactions(osi)
    print('node disps')
    for node in bnwf.top_nodes:
        print(o3.get_node_disp(osi, node, o3.cc.DOF2D_Y))
    print('node react')
    for node in bnwf.bot_nodes:
        print(o3.get_node_reaction(osi, node, o3.cc.DOF2D_Y))

    import matplotlib.pyplot as plt
    plt.plot(rot, mom)

    plt.show()

if __name__ == '__main__':
    run_example()