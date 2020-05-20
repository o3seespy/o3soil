import os

import numpy as np
import o3seespy as o3

ecp2o3_type_dict = {'TAU': ['stress', 'sxy'],
                    'ESIGY': ['stress', 'syy'],
                    'ESIGX': ['stress', 'sxx'],
                    'STRS': ['strain', 'gxy']}


class O3SRAOutputs(object):
    cache_path = ''
    out_dict = None
    area = 1.0
    outs = None

    def start_recorders(self, osi, outs, sn, eles, rec_dt, sn_xy=False):
        self.rec_dt = rec_dt
        self.eles = eles
        self.sn_xy = sn_xy
        if sn_xy:
            self.nodes = sn[0, :]
            f_order = 'F'
        else:
            self.nodes = sn[:, 0]
            f_order = 'C'
        self.outs = outs
        node_depths = np.array([node.y for node in sn[:, 0]])
        ele_depths = (node_depths[1:] + node_depths[:-1]) / 2
        rd = {}
        srd = {}
        for otype in outs:
            if otype in ['ACCX', 'DISPX', 'PP']:
                if isinstance(outs[otype], str) and outs[otype] == 'all':

                    if otype == 'ACCX':
                        rd['ACCX'] = o3.recorder.NodesToArrayCache(osi, nodes=self.nodes, dofs=[o3.cc.DOF2D_Y], res_type='accel',
                                                                dt=rec_dt)
                    if otype == 'DISPX':
                        rd['DISPX'] = o3.recorder.NodesToArrayCache(osi, nodes=self.nodes, dofs=[o3.cc.DOF2D_X], res_type='disp',
                                                                dt=rec_dt)
                    if otype == 'PP':
                        rd['PP'] = o3.recorder.NodesToArrayCache(osi, nodes=self.nodes, dofs=[o3.cc.DOF2D_PP], res_type='vel',
                                                                dt=rec_dt)
                else:
                    rd['ACCX'] = []
                    for i in range(len(outs['ACCX'])):
                        ind = np.argmin(abs(node_depths - outs['ACCX'][i]))
                        rd['ACCX'].append(
                            o3.recorder.NodeToArrayCache(osi, node=sn[ind][0], dofs=[o3.cc.X], res_type='accel', dt=rec_dt))
            if otype in ecp2o3_type_dict:
                rname = ecp2o3_type_dict[otype][0]  # recorder name
                for ele in eles:
                    assert isinstance(ele, o3.element.SSPquad) or isinstance(ele, o3.element.SSPquadUP)

                if isinstance(outs[otype], str) and outs[otype] == 'all':
                    if rname not in srd:
                        srd[rname] = o3.recorder.ElementsToArrayCache(osi, eles=eles, arg_vals=[rname], dt=rec_dt)
                else:
                    raise ValueError('Currently not supported')
                    # rd['STRESS'] = []
                    # for i in range(len(outs['TAU'])):
                    #     ind = np.argmin(abs(ele_depths - outs['TAU'][i]))
                    #     rd['STRESS'].append(o3.recorder.ElementToArrayCache(osi, ele=eles[ind], arg_vals=['stress'], dt=rec_dt))
            # if otype == 'ESIGY':
            #     if isinstance(outs['TAU'], str) and outs['TAU'] == 'all':
            #         if 'stress' not in srd:
            #             srd['stress'] = o3.recorder.ElementsToArrayCache(osi, eles=eles, arg_vals=['stress'], dt=rec_dt)
            if otype == 'TAUX':
                if isinstance(outs['TAUX'], str) and outs['TAUX'] == 'all':
                    rd['TAUX'] = o3.recorder.NodesToArrayCache(osi, nodes=sn.flatten(f_order), dofs=[o3.cc.X],
                                                               res_type='reaction',
                                                               dt=rec_dt)
            if otype == 'STRSX':
                if isinstance(outs['STRSX'], str) and outs['STRSX'] == 'all':
                    if 'DISPX' in outs:
                        continue
                    if sn_xy:
                        nodes = sn[0, :]
                    else:
                        nodes = sn[:, 0]
                    rd['DISPX'] = o3.recorder.NodesToArrayCache(osi, nodes=nodes, dofs=[o3.cc.X], res_type='disp',
                                                                dt=rec_dt)

        self.rd = rd
        self.srd = srd

    def results_to_files(self):
        od = self.results_to_dict()
        for item in od:
            ffp = self.cache_path + f'{item}.txt'
            if os.path.exists(ffp):
                os.remove(ffp)
            np.savetxt(ffp, od[item])

    def load_results_from_files(self, outs=None):
        if outs is None:
            outs = ['ACCX', 'TAU', 'STRS', 'time']
        od = {}
        for item in outs:
            od[item] = np.loadtxt(self.cache_path + f'{item}.txt')
        return od

    def results_to_dict(self):
        ro = o3.recorder.load_recorder_options()
        import pandas as pd
        df = pd.read_csv(ro)
        if self.outs is None:
            raise ValueError('outs is None')
            # items = list(self.rd)
        else:
            items = list(self.outs)

        if self.out_dict is None:
            self.out_dict = {}
            for item in self.srd:
                self.srd[item] = self.srd[item].collect().T
            for otype in items:
                if otype in self.rd:
                    vals = self.rd[otype].collect().T
                    self.out_dict[otype] = []
                    if otype == 'TAUX':
                        f_static = -np.cumsum(vals[::2, :] - vals[1::2, :], axis=0)[:-1]  # add left and right
                        f_dyn = vals[::2, :] + vals[1::2, :]  # add left and right
                        f_dyn_av = (f_dyn[1:] + f_dyn[:-1]) / 2
                        # self.out_dict[otype] = (f[1:, :] - f[:-1, :]) / area
                        self.out_dict[otype] = (f_dyn_av + f_static) / self.area
                    else:
                        self.out_dict[otype] = vals
                else:
                    if otype in ecp2o3_type_dict:
                        rname = ecp2o3_type_dict[otype][0]
                        ostr = ecp2o3_type_dict[otype][1]
                        dfe = df[df['recorder'] == rname]
                        vals = self.srd[rname]
                        cur_ind = 0
                        self.out_dict[otype] = []
                        for ele in self.eles:
                            mat_type = ele.mat.type
                            form = 'PlaneStrain'
                            dfm = dfe[(dfe['mat'] == mat_type) & (dfe['form'] == form)]
                            assert len(dfm) == 1, len(dfm)
                            outs = dfm['outs'].iloc[0].split('-')
                            oind = outs.index(ostr)
                            self.out_dict[otype].append(vals[cur_ind + oind])
                            cur_ind += len(outs)
                        self.out_dict[otype] = np.array(self.out_dict[otype])
                    if otype == 'STRSX':
                        depths = []
                        for node in self.nodes:
                            depths.append(node.y)
                        depths = np.array(depths)
                        d_incs = depths[1:] - depths[:-1]
                        vals = self.rd['DISPX'].collect(unlink=False).T
                        self.out_dict[otype] = (vals[1:] - vals[:-1]) / d_incs[:, np.newaxis]

            # Create time output
            if 'ACCX' in self.out_dict:
                self.out_dict['time'] = np.arange(0, len(self.out_dict['ACCX'][0])) * self.rec_dt
            elif 'TAU' in self.out_dict:
                self.out_dict['time'] = np.arange(0, len(self.out_dict['TAU'][0])) * self.rec_dt
        return self.out_dict