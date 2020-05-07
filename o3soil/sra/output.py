import os

import numpy as np
import o3seespy as o3


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
        else:
            self.nodes = sn[:, 0]
        self.outs = outs
        node_depths = np.array([node.y for node in sn[:, 0]])
        ele_depths = (node_depths[1:] + node_depths[:-1]) / 2
        ods = {}
        for otype in outs:
            if otype in ['ACCX', 'DISPX']:
                if isinstance(outs[otype], str) and outs[otype] == 'all':

                    if otype == 'ACCX':
                        ods['ACCX'] = o3.recorder.NodesToArrayCache(osi, nodes=self.nodes, dofs=[o3.cc.X], res_type='accel',
                                                                dt=rec_dt)
                    if otype == 'DISPX':
                        ods['DISPX'] = o3.recorder.NodesToArrayCache(osi, nodes=self.nodes, dofs=[o3.cc.X], res_type='disp',
                                                                dt=rec_dt)
                else:
                    ods['ACCX'] = []
                    for i in range(len(outs['ACCX'])):
                        ind = np.argmin(abs(node_depths - outs['ACCX'][i]))
                        ods['ACCX'].append(
                            o3.recorder.NodeToArrayCache(osi, node=sn[ind][0], dofs=[o3.cc.X], res_type='accel', dt=rec_dt))
            if otype == 'TAU':
                for ele in eles:
                    assert isinstance(ele, o3.element.SSPquad) or isinstance(ele, o3.element.SSPquadUP)
                ods['TAU'] = []
                if isinstance(outs['TAU'], str) and outs['TAU'] == 'all':
                    ods['TAU'] = o3.recorder.ElementsToArrayCache(osi, eles=eles, arg_vals=['stress'], dt=rec_dt)
                else:
                    for i in range(len(outs['TAU'])):
                        ind = np.argmin(abs(ele_depths - outs['TAU'][i]))
                        ods['TAU'].append(
                            o3.recorder.ElementToArrayCache(osi, ele=eles[ind], arg_vals=['stress'], dt=rec_dt))
            if otype == 'TAUX':
                if isinstance(outs['TAUX'], str) and outs['TAUX'] == 'all':
                    if sn_xy:
                        order = 'F'
                    else:
                        order = 'C'
                    ods['TAUX'] = o3.recorder.NodesToArrayCache(osi, nodes=sn.flatten(order), dofs=[o3.cc.X], res_type='reaction',
                                                                dt=rec_dt)
            if otype == 'STRS':
                ods['STRS'] = []
                if isinstance(outs['STRS'], str) and outs['STRS'] == 'all':
                    ods['STRS'] = o3.recorder.ElementsToArrayCache(osi, eles=eles, arg_vals=['strain'], dt=rec_dt)
                else:
                    for i in range(len(outs['STRS'])):
                        ind = np.argmin(abs(ele_depths - outs['STRS'][i]))
                        ods['STRS'].append(o3.recorder.ElementToArrayCache(osi, ele=eles[ind], arg_vals=['strain'], dt=rec_dt))
            if otype == 'STRSX':
                if isinstance(outs['STRSX'], str) and outs['STRSX'] == 'all':
                    if 'DISPX' in outs:
                        continue
                    if sn_xy:
                        nodes = sn[0, :]
                    else:
                        nodes = sn[:, 0]
                    ods['DISPX'] = o3.recorder.NodesToArrayCache(osi, nodes=nodes, dofs=[o3.cc.X], res_type='disp',
                                                                dt=rec_dt)

        self.ods = ods

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
            items = list(self.ods)
        else:
            items = list(self.outs)
        if self.out_dict is None:
            self.out_dict = {}
            for otype in items:
                if otype not in self.ods:
                    if otype == 'STRSX':
                        depths = []
                        for node in self.nodes:
                            depths.append(node.y)
                        depths = np.array(depths)
                        d_incs = depths[1:] - depths[:-1]
                        vals = self.ods['DISPX'].collect(unlink=False).T
                        self.out_dict[otype] = (vals[1:] - vals[:-1]) / d_incs[:, np.newaxis]
                elif isinstance(self.ods[otype], list):
                    self.out_dict[otype] = []
                    for i in range(len(self.ods[otype])):
                        if otype in ['TAU', 'STRS']:
                            self.out_dict[otype].append(self.ods[otype][i].collect()[2])
                        else:
                            self.out_dict[otype].append(self.ods[otype][i].collect())
                    self.out_dict[otype] = np.array(self.out_dict[otype])
                else:
                    vals = self.ods[otype].collect().T
                    cur_ind = 0
                    self.out_dict[otype] = []
                    if otype in ['TAU', 'STRS']:
                        for ele in self.eles:
                            mat_type = ele.mat.type
                            form = 'PlaneStrain'
                            dfe = df[(df['mat'] == mat_type) & (df['form'] == form)]
                            if otype == 'TAU':
                                dfe = dfe[dfe['recorder'] == 'stress']
                                ostr = 'sxy'
                            else:
                                dfe = dfe[dfe['recorder'] == 'strain']
                                ostr = 'gxy'
                            assert len(dfe) == 1, len(dfe)
                            outs = dfe['outs'].iloc[0].split('-')
                            oind = outs.index(ostr)
                            self.out_dict[otype].append(vals[cur_ind + oind])
                            cur_ind += len(outs)
                        self.out_dict[otype] = np.array(self.out_dict[otype])
                        # if otype == 'STRS':
                        #     self.out_dict[otype] = vals[2::3]  # Assumes pimy
                        # elif otype == 'TAU':
                        #     self.out_dict[otype] = vals[3::5]  # Assumes pimy
                    elif otype == 'TAUX':
                        f_static = -np.cumsum(vals[::2, :] - vals[1::2, :], axis=0)[:-1]  # add left and right
                        f_dyn = vals[::2, :] + vals[1::2, :]  # add left and right
                        f_dyn_av = (f_dyn[1:] + f_dyn[:-1]) / 2
                        # self.out_dict[otype] = (f[1:, :] - f[:-1, :]) / area
                        self.out_dict[otype] = (f_dyn_av + f_static) / self.area
                    else:
                        self.out_dict[otype] = vals
            # Create time output
            if 'ACCX' in self.out_dict:
                self.out_dict['time'] = np.arange(0, len(self.out_dict['ACCX'][0])) * self.rec_dt
            elif 'TAU' in self.out_dict:
                self.out_dict['time'] = np.arange(0, len(self.out_dict['TAU'][0])) * self.rec_dt
        return self.out_dict