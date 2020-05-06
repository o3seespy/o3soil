from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
from bwplot import cbox, colors
import os
import o3seespy as o3
from o3seespy.results import Results2D


class Window(pg.GraphicsWindow):  # TODO: consider switching to pandas.read_csv(ffp, engine='c')
    started = 0

    def __init__(self, parent=None):
        self.app = QtWidgets.QApplication([])
        super().__init__(parent=parent)
        #
        # pg.setConfigOptions(antialias=False)  # True seems to work as well
        # self.app.aboutToQuit.connect(self.stop)
        self.mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainLayout)
        self.timer = QtCore.QTimer(self)
        self.x_coords = None
        self.y_coords = None
        self.x = None
        self.y = None
        self.time = None
        self.i = 0
        self.plotItem = self.addPlot(title="Nodes")
        self.node_points_plot = None
        self.ele_lines_plot = {}
        self.ele_node_tags = {2: [], 3: [], 4: [], 8: [], 9: [], 20: []}
        self.ele_x_coords = {}
        self.ele_y_coords = {}
        self.ele_connects = {}

    def get_reverse_ele_node_tags(self):
        return list(self.ele_node_tags)[::-1]


    def init_model(self, coords, ele2node_tags=None):
        self.x_coords = np.array(coords)[:, 0]
        self.y_coords = np.array(coords)[:, 1]

        if ele2node_tags is not None:

            self.ele2node_tags = ele2node_tags
            rnt = self.get_reverse_ele2node_tags()
            for nl in rnt:
                self.ele2node_tags[nl] = np.array(self.ele2node_tags[nl], dtype=int)
                ele_x_coords = self.x_coords[self.ele2node_tags[nl] - 1]
                ele_y_coords = self.y_coords[self.ele2node_tags[nl] - 1]
                ele_x_coords = np.insert(ele_x_coords, len(ele_x_coords[0]), ele_x_coords[:, 0], axis=1)
                ele_y_coords = np.insert(ele_y_coords, len(ele_y_coords[0]), ele_y_coords[:, 0], axis=1)
                connect = np.ones_like(ele_x_coords, dtype=np.ubyte)
                connect[:, -1] = 0
                self.ele_x_coords = ele_x_coords.flatten()
                self.ele_y_coords = ele_y_coords.flatten()
                self.ele_connects[nl] = connect.flatten()
                if nl == 2:
                    pen = 'b'
                else:
                    pen = 'w'
                self.ele_lines_plot[nl] = self.plotItem.plot(self.ele_x_coords, self.ele_y_coords, pen=pen, connect=self.ele_connects[nl])

        self.node_points_plot = self.plotItem.plot([], pen=None,
                                                   symbolBrush=(255, 0, 0), symbolSize=5, symbolPen=None)
        self.node_points_plot.setData(self.x_coords, self.y_coords)
        self.plotItem.autoRange(padding=0.05)  # TODO: depends on xmag
        self.plotItem.disableAutoRange()

    def start(self):
        if not self.started:
            self.started = 1
            self.raise_()
            self.app.exec_()

    def plot(self, x, y, dt, xmag=10.0, ymag=10.0, node_c=None, t_scale=1):
        self.timer.setInterval(1000. * dt * t_scale)  # in milliseconds
        self.timer.start()
        self.node_c = node_c
        self.x = np.array(x) * xmag
        self.y = np.array(y) * ymag
        if self.x_coords is not None:
            self.x += self.x_coords
            self.y += self.y_coords

        self.time = np.arange(len(self.x)) * dt

        # Prepare node colors
        if self.node_c is not None:
            ncol = colors.get_len_red_to_yellow()
            self.brush_list = [pg.mkColor(colors.red_to_yellow(i, as255=True)) for i in range(ncol)]

            y_max = np.max(self.node_c)
            y_min = np.min(self.node_c)
            inc = (y_max - y_min) * 0.001
            bis = (self.node_c - y_min) / (y_max + inc - y_min) * ncol
            self.bis = np.array(bis, dtype=int)

        self.timer.timeout.connect(self.updater)

    def updater(self):
        self.i = self.i + 1
        if self.i == len(self.time) - 1:
            self.timer.stop()

        if self.node_c is not None:
            blist = np.array(self.brush_list)[self.bis[self.i]]
            # TODO: try using ScatterPlotWidget and colorMap
            self.node_points_plot.setData(self.x[self.i], self.y[self.i], brush='g', symbol='o', symbolBrush=blist)
        else:
            self.node_points_plot.setData(self.x[self.i], self.y[self.i], brush='g', symbol='o')
        for nl in self.ele_node_tags:
            if nl == 2:
                pen = 'b'
            else:
                pen = 'w'
            ele_x_coords = (self.x[self.i])[self.ele_node_tags[nl] - 1]
            ele_y_coords = (self.y[self.i])[self.ele_node_tags[nl] - 1]
            ele_x_coords = np.insert(ele_x_coords, len(ele_x_coords[0]), ele_x_coords[:, 0], axis=1).flatten()
            ele_y_coords = np.insert(ele_y_coords, len(ele_y_coords[0]), ele_y_coords[:, 0], axis=1).flatten()
            self.ele_lines_plot[nl].setData(ele_x_coords, ele_y_coords, pen=pen, connect=self.ele_connects[nl])
        self.plotItem.setTitle(f"Nodes time: {self.time[self.i]:.4g}s")

    def stop(self):
        print('Exit')
        self.status = False
        self.app.close()
        pg.close()
        # sys.exit()


def get_app_and_window():
    app = QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=False)  # True seems to work as well
    return app, Window()


def plot_two_d_system(win, tds):
    import sfsimodels as sm
    assert isinstance(tds, sm.TwoDSystem)
    y_sps_surf = np.interp(tds.x_sps, tds.x_surf, tds.y_surf)

    for i in range(len(tds.sps)):
        x0 = tds.x_sps[i]
        if i == len(tds.sps) - 1:
            x1 = tds.width
        else:
            x1 = tds.x_sps[i + 1]
        xs = np.array([x0, x1])
        win.plot(tds.x_surf, tds.y_surf, pen='w')
        x_angles = [10] + list(tds.sps[i].x_angles)
        sp = tds.sps[i]
        for ll in range(2, sp.n_layers + 1):
            ys = y_sps_surf[i] - sp.layer_depth(ll) + x_angles[ll - 1] * xs
            win.plot(xs, ys, pen='w')
    win.plot([0, 0], [-tds.height, tds.y_surf[0]], pen='w')
    win.plot([tds.width, tds.width], [-tds.height, tds.y_surf[-1]], pen='w')
    win.plot([0, tds.width], [-tds.height, -tds.height], pen='w')
    for i, bd in enumerate(tds.bds):
        fd = bd.fd
        fcx = tds.x_bds[i] + bd.x_fd
        fcy = np.interp(fcx, tds.x_surf, tds.y_surf)
        print(fcx, fcy)
        x = [fcx - fd.width / 2, fcx + fd.width / 2, fcx + fd.width / 2, fcx - fd.width / 2, fcx - fd.width / 2]
        y = [fcy - fd.depth, fcy - fd.depth, fcy - fd.depth + fd.height, fcy - fd.depth + fd.height, fcy - fd.depth]
        win.plot(x, y, pen='r')



def plot_finite_element_mesh(win, femesh):
    for i in range(len(femesh.y_nodes)):
        win.addItem(pg.InfiniteLine(femesh.y_nodes[i], angle=0, pen=(0, 255, 255, 30)))
    for i in range(len(femesh.x_nodes)):
        win.addItem(pg.InfiniteLine(femesh.x_nodes[i], angle=90, pen=(0, 255, 255, 30)))

    for xx in range(len(femesh.soil_grid)):
        pid = femesh.profile_indys[xx]
        for yy in range(len(femesh.soil_grid[0])):
            sl_ind = femesh.soil_grid[xx][yy]
            if sl_ind > 1000:
                continue

            r = pg.QtGui.QGraphicsRectItem(femesh.x_nodes[xx], femesh.y_nodes[yy],
                                           femesh.x_nodes[xx + 1] - femesh.x_nodes[xx],
                                           femesh.y_nodes[yy + 1] - femesh.y_nodes[yy])
            r.setPen(pg.mkPen(None))
            ci = sl_ind
            r.setBrush(pg.mkBrush(cbox(ci, as255=True, alpha=80)))
            win.addItem(r)



def replot(out_folder='', dynamic=0, dt=0.01, xmag=1, ymag=1, t_scale=1):
    o3res = Results2D()
    o3res.dynamic = dynamic
    o3res.cache_path = out_folder
    o3res.load_from_cache()

    win = Window()
    win.resize(800, 600)
    win.init_model(o3res.coords, o3res.ele_node_tags)
    if dynamic:
        win.plot(o3res.x_disp, o3res.y_disp, node_c=o3res.node_c, dt=dt, xmag=xmag, ymag=ymag, t_scale=t_scale)
    win.start()


# if __name__ == '__main__':
#
#     app = QtWidgets.QApplication([])
#     pg.setConfigOptions(antialias=False)  # True seems to work as well
#     x = np.arange(0, 100)[np.newaxis, :] * np.ones((4, 100)) * 0.01 * np.arange(1, 5)[:, np.newaxis]
#     x = x.T
#     y = np.sin(x)
#     coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
#     win = Window()
#     win.init_model(coords)
#     win.plot(x, y, dt=0.01)
#     win.show()
#     win.resize(800, 600)
#     win.raise_()
#     app.exec_()
