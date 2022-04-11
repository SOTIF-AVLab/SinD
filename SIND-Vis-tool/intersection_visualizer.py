import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from utils.DataReader import read_tracks_all, read_tracks_meta, read_light

try:
    import lanelet2

    use_lanelet2_lib = True
    from utils import map_vis_lanelet2
except ImportError:
    import warnings

    string = "Could not import lanelet2. It must be built and sourced, " + \
             "see https://github.com/fzi-forschungszentrum-informatik/Lanelet2 for details."
    warnings.warn(string)
    print("Using visualization without lanelet2.")
    use_lanelet2_lib = False
    from utils import map_vis_without_lanelet

from loguru import logger
import glob
import os


class Visualizer(object):
    def __init__(self, config):

        self.config = dict(config)
        self.__read_data()
        self.__para_init()
        self.__interface_init()

    def __read_data(self):

        datapath = os.path.join(self.config['path'], self.config['record_name'])
        self.VehTracks, self.PedTracks = read_tracks_all(datapath)
        self.veh_tracks_meta = read_tracks_meta(os.path.join(datapath, 'Veh_tracks_meta.csv'))
        self.ped_tracks_meta = read_tracks_meta(os.path.join(datapath, 'Ped_tracks_meta.csv'))


        self.maximum_frames = np.max([track["finalFrame"] for track in self.veh_tracks_meta.values()])
        self.maximum_frames = max(np.max([track["finalFrame"] for track in self.ped_tracks_meta.values()]),
                                  self.maximum_frames)


        traffic_light_path = glob.glob(datapath+'/*.xlsx')[0]
        self.map_path = glob.glob(self.config['path']+'/*.osm')[0]

        if self.config['plot_traffic_light']:
            self.light_state = read_light(traffic_light_path, self.maximum_frames * 3)


    def __para_init(self):

        self.colors = {'car': 'dodgerblue', 'bicycle': 'green', 'motorcycle': 'orange', 'bus': 'turquoise',
                       'truck': 'yellow', 'tricycle'
                       : 'hotpink', 'pedestrain': 'red'}
        # Save ids for each frame
        self.ids_for_frame = {}
        for i_frame in range(self.maximum_frames):  # 清点每一帧各有哪些TP
            veh_indices = [i_track for i_track in self.veh_tracks_meta.keys()
                           if
                           self.veh_tracks_meta[i_track]["initialFrame"] <= i_frame <=
                           self.veh_tracks_meta[i_track][
                               "finalFrame"]]
            ped_indices = [i_track for i_track in self.ped_tracks_meta.keys()
                           if
                           self.ped_tracks_meta[i_track]["initialFrame"] <= i_frame <=
                           self.ped_tracks_meta[i_track][
                               "finalFrame"]]
            self.ids_for_frame[i_frame] = {'veh': veh_indices, 'ped': ped_indices}

        # Define axes for the widgets
        self.current_frame = 0
        self.play_stop = True
        self.ColorsMap = {'ax_slider': 'lightgoldenrodyellow'}
        self.skip_n_frames = self.config["skip_n_frames"]
        self.delta_time = 1 / 29.97 * 3

        self.changed_button = False
        self.plot_objs = {"veh": {}, "ped": {}, "light": {}, "graph_line": {}}
        self.track_info_figures = {}

    def __interface_init(self):

        self.fig, self.ax = plt.subplots(1, 1)
        self.title = self.fig.suptitle(" ")
        if use_lanelet2_lib:

            projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
            laneletmap = lanelet2.io.load(self.map_path, projector)
            map_vis_lanelet2.draw_lanelet_map(laneletmap, self.ax)

        else:
            map_vis_without_lanelet.draw_map_without_lanelet(self.map_path, self.ax, 0, 0)

        self.fig.canvas.set_window_title("Crossroads Dataset Visualization")

        self.ax_slider = self.fig.add_axes([0.2, 0.035, 0.2, 0.04], facecolor=self.ColorsMap['ax_slider'])  # Slider
        self.ax_button_previous2 = self.fig.add_axes([0.02, 0.035, 0.06, 0.04])
        self.ax_button_previous = self.fig.add_axes([0.09, 0.035, 0.06, 0.04])
        self.ax_button_next = self.fig.add_axes([0.44, 0.035, 0.06, 0.04])
        self.ax_button_next2 = self.fig.add_axes([0.51, 0.035, 0.06, 0.04])
        self.ax_button_play = self.fig.add_axes([0.58, 0.035, 0.06, 0.04])
        self.ax_button_stop = self.fig.add_axes([0.65, 0.035, 0.06, 0.04])

        self.centroid_style = dict(fill=True, edgecolor="black", lw=0.15, alpha=1,
                                   radius=0.2, zorder=30)
        self.track_style = dict(linewidth=1, zorder=10)
        self.track_style_future = dict(color="linen", linewidth=1, alpha=0.7, zorder=10)

        # Define the callbacks for the widgets' actions
        self.frame_slider = FrameControlSlider(self.ax_slider, 'Frame', 0, self.maximum_frames - 1,
                                               valinit=self.current_frame,
                                               valfmt='%s')
        # self.frame_slider = FrameControlSlider(self.ax_slider, 'Progress bar')

        self.button_previous2 = Button(self.ax_button_previous2, 'Previous x%d' % self.skip_n_frames)
        self.button_previous = Button(self.ax_button_previous, 'Previous')
        self.button_next = Button(self.ax_button_next, 'Next')
        self.button_next2 = Button(self.ax_button_next2, 'Next x%d' % self.skip_n_frames)
        self.button_play = Button(self.ax_button_play, 'Play')
        self.button_stop = Button(self.ax_button_stop, 'Stop')

        self.frame_slider.on_changed(self.update_slider)
        self.button_previous.on_clicked(self.update_button_previous)
        self.button_next.on_clicked(self.update_button_next)
        self.button_previous2.on_clicked(self.update_button_previous2)
        self.button_next2.on_clicked(self.update_button_next2)
        self.button_play.on_clicked(self.start_play)
        self.button_stop.on_clicked(self.stop_play)

        self.timer = self.fig.canvas.new_timer(interval=1000 * self.delta_time)
        self.timer.add_callback(self.update_time_next, self.ax)

        # Define the callbacks for the widgets' actions
        self.fig.canvas.mpl_connect('key_press_event', self.update_keypress)

        self.ax.set_autoscale_on(True)
        self.update_figure()

    def trigger_update(self):

        self.remove_patches()
        self.update_figure()
        self.update_pop_up_windows()
        self.frame_slider.update_val_external(self.current_frame)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        artist = event.artist  # return the polygon

        if isinstance(artist, NumPolygon):
            track_id = artist.track_id
            print(str(track_id) + " be picked !")
            self.plot_objs['graph_line'][track_id] = {}

            if not str(track_id).startswith("P"):
                track = self.VehTracks[track_id]
                track_meta = self.veh_tracks_meta[track_id]
            else:
                track = self.PedTracks[track_id]
                track_meta = self.ped_tracks_meta[track_id]

            # Get information of the selected track
            centroids = track["center"]
            centroids = np.transpose(centroids)
            initial_frame = track_meta["initialFrame"]
            final_frame = track_meta["finalFrame"]
            x_limits = [initial_frame, final_frame]
            track_frames = np.linspace(initial_frame, final_frame, centroids.shape[1], dtype=np.int64)

            # Create a new figure that pops up
            fig = plt.figure(np.random.randint(0, 5000, 1))
            fig.canvas.mpl_connect('close_event',
                                   lambda evt: self.close_track_info_figure(evt, track_id))  # Clear cache
            fig.canvas.mpl_connect('resize_event', lambda evt: fig.tight_layout())
            fig.set_size_inches(12, 7)
            fig.canvas.set_window_title("Track {} ({})".format(track_id, track_meta["class"]))

            borders_list = []
            subplot_list = []

            key_check_list = ["vx", "vx", "ax", "ay", "v_lon", "a_lon"]
            counter = 3
            for check_key in key_check_list:
                if check_key in track and track[check_key] is not None:
                    counter = counter + 1

            if 3 < counter <= 6:
                subplot_index = 321
            elif 6 < counter <= 9:
                subplot_index = 331
            else:
                subplot_index = 311

            # ---------- X POSITION ----------
            sub_plot = plt.subplot(subplot_index, title="X-Position [m]")
            subplot_list.append(sub_plot)
            x_positions = centroids[0, :]
            borders = [np.amin(x_positions), np.amax(x_positions)]
            plt.plot(track_frames, x_positions)
            red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")[0]
            self.plot_objs['graph_line'][track_id] = {subplot_index: red_line}

            borders_list.append(borders)
            sub_plot.grid(True)
            plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- Y POSITION ----------
            sub_plot = plt.subplot(subplot_index, title="Y-Position [m]")
            subplot_list.append(sub_plot)
            y_positions = centroids[1, :]
            borders = [np.amin(y_positions), np.amax(y_positions)]
            plt.plot(track_frames, y_positions)
            red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")[0]
            self.plot_objs['graph_line'][track_id][subplot_index] = red_line

            borders_list.append(borders)
            plt.xlim(x_limits)
            plt.ylim(borders)
            sub_plot.grid(True)
            plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- HEADING ----------
            if "heading_rad" in track and track["heading_rad"] is not None:
                rotations = np.rad2deg(track["heading_rad"])
                yaw = np.rad2deg(track["yaw_rad"])
                sub_plot = plt.subplot(subplot_index, title="Yaw angle & Heading angle[deg]")
                subplot_list.append(sub_plot)
                borders = [np.amin(yaw) - 10, np.amax(yaw) + 10]
                plt.plot(track_frames, np.unwrap(yaw, discont=180))
                plt.plot(track_frames, np.unwrap(rotations, discont=180))
                plt.legend(['Yaw angle', 'Heading angle'])

            else:
                # 初始化向量
                rotations = np.rad2deg(np.arctan2(track['vy'], track['vx']))
                sub_plot = plt.subplot(subplot_index, title="People's heading angle[deg]")
                subplot_list.append(sub_plot)
                borders = [np.amin(rotations) - 10, np.amax(rotations) + 10]
                plt.plot(track_frames, np.unwrap(rotations, discont=180))

            red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")[0]
            self.plot_objs['graph_line'][track_id][subplot_index] = red_line
            borders_list.append(borders)
            plt.xlim(x_limits)
            plt.ylim(borders)
            sub_plot.grid(True)
            plt.xlabel('Frame')
            subplot_index = subplot_index + 1
            # ---------- "xVelocity" ----------
            if "vx" in track and track["vx"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="X-Velocity [m/s]")
                subplot_list.append(sub_plot)
                x_velocity = track["vx"]
                borders = [np.amin(x_velocity), np.amax(x_velocity)]
                plt.plot(track_frames, x_velocity)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")[0]
                self.plot_objs['graph_line'][track_id][subplot_index] = red_line
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "yVelocity" ----------
            if "vy" in track and track["vy"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="Y-Velocity [m/s]")
                subplot_list.append(sub_plot)
                y_velocity = track["vy"]
                borders = [np.amin(y_velocity), np.amax(y_velocity)]
                plt.plot(track_frames, y_velocity)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")[0]
                self.plot_objs['graph_line'][track_id][subplot_index] = red_line
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "lonVelocity" ----------
            if "v_lon" in track and track["v_lon"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="Longitudinal-Velocity [m/s]")
                subplot_list.append(sub_plot)
                lon_velocity = track["v_lon"]
                borders = [np.amin(lon_velocity), np.amax(lon_velocity)]
                plt.plot(track_frames, lon_velocity)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")[0]
                self.plot_objs['graph_line'][track_id][subplot_index] = red_line
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "XAcceleration" ----------
            if "ax" in track and track["ax"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="X-Acceleration [m/s^2]")
                subplot_list.append(sub_plot)
                x_acc = track["ax"]
                borders = [np.amin(x_acc), np.amax(x_acc)]
                plt.plot(track_frames, x_acc)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")[0]
                self.plot_objs['graph_line'][track_id][subplot_index] = red_line
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "yAcceleration" ----------
            if "ay" in track and track["ay"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="Y-Acceleration [m/s^2]")
                subplot_list.append(sub_plot)
                y_acc = track["ay"]
                borders = [np.amin(y_acc), np.amax(y_acc)]
                plt.plot(track_frames, y_acc)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")[0]
                self.plot_objs['graph_line'][track_id][subplot_index] = red_line
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "yAcceleration" ----------
            if "a_lon" in track and track["a_lon"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="Longitudinal-Acceleration [m/s^2]")
                subplot_list.append(sub_plot)
                lon_acc = track["a_lon"]
                borders = [np.amin(lon_acc), np.amax(lon_acc)]  # the higest point and lowest point
                plt.plot(track_frames, lon_acc)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")[0]
                self.plot_objs['graph_line'][track_id][subplot_index] = red_line
                borders_list.append(borders)

                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')

            self.track_info_figures[track_id] = {"main_figure": fig,
                                                 "borders": borders_list,
                                                 "subplots": subplot_list}
            plt.show()

    def update_keypress(self, evt):

        if evt.key == "right" and self.current_frame + self.skip_n_frames < self.maximum_frames:
            self.current_frame = self.current_frame + self.skip_n_frames
            self.trigger_update()
            self.stop_play(None)
        elif evt.key == "left" and self.current_frame - self.skip_n_frames >= 0:
            self.current_frame = self.current_frame - self.skip_n_frames
            self.trigger_update()
            self.stop_play(None)
        elif evt.key == "up":
            self.start_play(None)
        elif evt.key == "down":
            self.stop_play(None)

    def update_slider(self, value):
        if not self.changed_button:
            self.current_frame = value
            self.trigger_update()
        self.changed_button = False

    def update_time_next(self, _):
        if self.current_frame + 1 < self.maximum_frames:
            self.current_frame = self.current_frame + 1
            self.changed_button = True
            self.trigger_update()
        else:
            logger.warning(
                "There are no frames available with an index higher than {}.".format(self.maximum_frames))

    def update_button_next(self, _):
        if self.current_frame + 1 < self.maximum_frames:
            self.current_frame = self.current_frame + 1
            self.changed_button = True
            self.trigger_update()
            self.stop_play(None)
        else:
            logger.warning(
                "There are no frames available with an index higher than {}.".format(self.maximum_frames))

    def update_button_next2(self, _):
        if self.current_frame + self.skip_n_frames < self.maximum_frames:
            self.current_frame = self.current_frame + self.skip_n_frames
            self.changed_button = True
            self.trigger_update()
            self.stop_play(None)
        else:
            logger.warning("There are no frames available with an index higher than {}.".format(self.maximum_frames))

    def update_button_previous(self, _):
        if self.current_frame - 1 >= 0:
            self.current_frame = self.current_frame - 1
            self.changed_button = True
            self.trigger_update()
            self.stop_play(None)
        else:
            logger.warning("There are no frames available with an index lower than 1.")

    def update_button_previous2(self, _):
        if self.current_frame - self.skip_n_frames >= 0:
            self.current_frame = self.current_frame - self.skip_n_frames
            self.changed_button = True
            self.trigger_update()
            self.stop_play(None)
        else:
            logger.warning("There are no frames available with an index lower than 1.")

    def start_play(self, _):
        self.timer.start()

    def stop_play(self, _):
        self.timer.stop()

    def update_figure(self):

        self.title.set_text(
            "Frame(s) = \n{} / {} ({:.2f}/{:.2f})".format(self.current_frame, self.maximum_frames,
                                                           self.current_frame * self.delta_time,
                                                           self.maximum_frames * self.delta_time))

        for track_id in self.ids_for_frame[self.current_frame]["veh"]:
            # object is visible
            frame_id = self.current_frame - self.veh_tracks_meta[track_id]['initialFrame']
            value = self.VehTracks[track_id]
            bbox = value['bbox'][frame_id, :, :]
            tri = value['triangle'][frame_id, :, :]
            if track_id not in self.plot_objs['veh']:
                color = self.colors[value["agent_type"]]

                # rect = Num_Polygon(bbox, closed=True,
                #                    zorder=20, facecolor=color, picker=True, track_id=track_id,
                #                    fill=True, edgecolor="k", alpha=0.8)
                rect = NumPolygon(bbox, closed=True,
                                   zorder=20, color=color, picker=True, track_id=track_id,
                                   fill=True, alpha=0.6)
                triangle_style = dict(facecolor="k", fill=True, edgecolor="k", lw=0.1, alpha=0.6, zorder=21)

                triangle = plt.Polygon(tri, True, **triangle_style)
                self.ax.add_patch(rect)
                self.ax.add_patch(triangle)

                if self.config['behaviour_type']:
                    if 'yellow-light running' in self.veh_tracks_meta[track_id]['Signal_Violation_Behavior']:
                        text_color = 'yellow'
                    elif 'red-light running' in self.veh_tracks_meta[track_id]['Signal_Violation_Behavior']:
                        text_color = 'red'
                    else:
                        text_color = 'black'
                else:
                    text_color = 'black'

                show_text = str(track_id)
                text = self.ax.text(value['x'][frame_id], value['y'][frame_id] + 1.5, show_text,
                                    horizontalalignment='center', zorder=30, color=text_color)

                self.plot_objs["veh"][track_id] = {"rect": rect, "tri": triangle, "text": text}

                if self.config["plotTrackingLines"]:

                    plotted_centroid = plt.Circle((value['x'][frame_id],
                                                   value['y'][frame_id]),
                                                  facecolor=color,
                                                  **self.centroid_style)

                    self.ax.add_patch(plotted_centroid)
                    self.plot_objs["veh"][track_id]['point'] = plotted_centroid

                    if value["center"].shape[0] > 0:
                        # Calculate the centroid of the vehicles by using the bounding box information
                        # Check track direction
                        plotted_centroids = self.ax.plot(
                            value["center"][0:frame_id + 1][:, 0],
                            value["center"][0:frame_id + 1][:, 1],
                            color=color, **self.track_style)[0]  # plot return a Handle

                        self.plot_objs['veh'][track_id]['past_tracking_line'] = plotted_centroids
                        if self.config["plotFutureTrackingLines"]:
                            # Check track direction
                            plotted_centroids_future = self.ax.plot(
                                value["center"][frame_id:][:, 0],
                                value["center"][frame_id:][:, 1],
                                **self.track_style_future)[0]
                            self.plot_objs["veh"][track_id]['future_tracking_line'] = plotted_centroids_future

            else:
                self.plot_objs["veh"][track_id]["rect"].set_xy(bbox)
                self.plot_objs["veh"][track_id]["tri"].set_xy(tri)
                self.plot_objs["veh"][track_id]["text"].set_position((value['x'][frame_id], value['y'][frame_id] + 1.5))

                if self.config["plotTrackingLines"]:
                    self.plot_objs["veh"][track_id]['past_tracking_line'].set_data(
                        value["center"][0:frame_id + 1][:, 0],
                        value["center"][0:frame_id + 1][:, 1])
                    self.plot_objs["veh"][track_id]['point'].set_center((value['x'][frame_id], value['y'][frame_id]))
                    if self.config["plotFutureTrackingLines"]:
                        self.plot_objs["veh"][track_id]['future_tracking_line'].set_data(
                            value["center"][frame_id:][:, 0],
                            value["center"][frame_id:][:, 1])

        for track_id in self.ids_for_frame[self.current_frame]["ped"]:

            frame_id = self.current_frame - self.ped_tracks_meta[track_id]['initialFrame']
            value = self.PedTracks[track_id]
            bbox = value['bbox'][frame_id, :, :]
            if track_id not in self.plot_objs['ped']:
                color = self.colors[value["agent_type"]]

                rect = NumPolygon(bbox, closed=True,
                                   zorder=20, color=color, picker=True, track_id=track_id)
                self.ax.add_patch(rect)
                text = self.ax.text(value['x'][frame_id], value['y'][frame_id] + 2, str(track_id),
                                    horizontalalignment='center', zorder=30)
                self.plot_objs["ped"][track_id] = {"rect": rect, "text": text}

                if self.config["plotTrackingLines"]:

                    plotted_centroids = self.ax.plot(
                        value["center"][0:frame_id + 1][:, 0],
                        value["center"][0:frame_id + 1][:, 1],
                        color=color, **self.track_style)[0]  # plot return a Handle

                    self.plot_objs["ped"][track_id]['past_tracking_line'] = plotted_centroids
                    if self.config["plotFutureTrackingLines"]:
                        # Check track direction
                        plotted_centroids_future = self.ax.plot(
                            value["center"][frame_id:][:, 0],
                            value["center"][frame_id:][:, 1],
                            **self.track_style_future)[0]
                        self.plot_objs["ped"][track_id]['future_tracking_line'] = plotted_centroids_future

            else:
                self.plot_objs["ped"][track_id]["rect"].set_xy(bbox)
                self.plot_objs["ped"][track_id]["text"].set_position((value['x'][frame_id], value['y'][frame_id] + 2))

                if self.config["plotTrackingLines"]:
                    self.plot_objs["ped"][track_id]['past_tracking_line'].set_data(
                        value["center"][0:frame_id + 1][:, 0],
                        value["center"][0:frame_id + 1][:, 1])
                    if self.config["plotFutureTrackingLines"]:
                        self.plot_objs["ped"][track_id]['future_tracking_line'].set_data(
                            value["center"][frame_id:][:, 0],
                            value["center"][frame_id:][:, 1])

        if self.config['plot_traffic_light']:
            cur_state = self.light_state[self.current_frame * 3]

            point_colors = {0: 'red', 1: 'green', 3: 'yellow'}
            traffic = ['traffic_light_1', 'traffic_light_2', 'traffic_light_3', 'traffic_light_4',
                       'traffic_light_5', 'traffic_light_6', 'traffic_light_7', 'traffic_light_8']
            # light_colors = {'red': {'on': (252, 52, 62), 'off': (68, 14, 17)},
            #                 'yellow': {'on': (245, 224, 73), 'off': (66, 61, 20)},
            #                 'green': {'on': (58, 181, 73), 'off': (16, 49, 20)}}
            light_colors = {'red': {'on': "#FC343E", 'off': "#440E11"},
                            'yellow': {'on': "#F5E049", 'off': "#423D14"},
                            'green': {'on': "#3AB549", 'off': "#103114"}}

            traffic_pos = [(22, 38), (32, 30), (32, 5), (25, -3), (5, -3), (-5, 5), (-5, 28), (3, 38)]

            for i, key in enumerate(traffic):
                if key not in self.plot_objs["light"]:
                    self.plot_objs["light"][key] = {}
                    pos = traffic_pos[i]
                    self.ax.add_patch(
                        plt.Rectangle(xy=(pos[0] - 1, pos[1] - 2.25), width=2, height=4.5, color='black', zorder=15))
                    pos = {'red': (pos[0], pos[1] + 1.5), 'yellow': pos, 'green': (pos[0], pos[1] - 1.5)}
                    for value in point_colors.values():
                        if point_colors[cur_state[i]] == value:
                            circle = plt.Circle(pos[value], radius=0.5, zorder=19,
                                                color=light_colors[value]['on'])
                        else:
                            circle = plt.Circle(pos[value], radius=0.5, zorder=19,
                                                color=light_colors[value]['off'])

                        # circle = plt.Circle(traffic_pos[i], radius=0.5, zorder=19,
                        #                     color=point_colors[cur_state[i]])
                        self.ax.add_patch(circle)
                        self.plot_objs["light"][key][value] = circle

                    # self.ax.add_patch(circle)
                    # self.plot_objs["light"][key] = circle
                    # text_dict[key] = axes.text(traffic_pos[i][0], traffic_pos[i][1]-2, str(key), horizontalalignment='center', zorder=30)
                    # circle.set_color(point_colors[cur_state])
                else:

                    for value in point_colors.values():
                        if point_colors[cur_state[i]] == value:
                            self.plot_objs["light"][key][value].set_color(light_colors[value]['on'])
                        else:
                            self.plot_objs["light"][key][value].set_color(light_colors[value]['off'])

        self.fig.canvas.mpl_connect('pick_event', self.on_click)

    def remove_patches(self):

        self.fig.canvas.mpl_disconnect('pick_event')  # why cancel ?

        for ObjType in self.plot_objs.keys():

            if ObjType in ["light"]:
                continue

            last_cache = list(self.plot_objs[ObjType].keys())
            istrack = ObjType in ['veh', 'ped']

            for track_id in last_cache:
                if istrack and track_id in self.ids_for_frame[self.current_frame][ObjType]:
                    continue
                for _, plot_obj in self.plot_objs[ObjType][track_id].items():
                    plot_obj.remove()
                self.plot_objs[ObjType].pop(track_id)

    def close_track_info_figure(self, evt, track_id):
        if track_id in self.track_info_figures:
            self.track_info_figures[track_id]["main_figure"].canvas.mpl_disconnect('close_event')
            self.track_info_figures.pop(track_id)

    def update_pop_up_windows(self):

        for track_id, track_map in self.track_info_figures.items():
            borders = track_map["borders"]
            subplots = track_map["subplots"]
            for subplot_index, subplot_figure in enumerate(subplots):

                if track_id not in self.plot_objs["graph_line"]:
                    new_line = subplot_figure.plot([self.current_frame, self.current_frame], borders[subplot_index],
                                                   "--r")[0]  # it is a list
                    self.plot_objs["graph_line"][track_id] = {subplot_index: new_line}
                else:
                    if subplot_index in self.plot_objs["graph_line"][track_id]:
                        self.plot_objs["graph_line"][track_id][subplot_index].set_data(
                            [self.current_frame, self.current_frame], borders[subplot_index])
                    else:
                        new_line = subplot_figure.plot([self.current_frame, self.current_frame], borders[subplot_index],
                                                       "--r")[0]
                        self.plot_objs["graph_line"][track_id][subplot_index] = new_line

            track_map["main_figure"].canvas.draw_idle()

    @staticmethod
    @logger.catch(reraise=True)
    def show():
        plt.show()


class FrameControlSlider(Slider):
    def __init__(self, *args, **kwargs):
        self.inc = kwargs.pop('increment', 1)
        self.valfmt = '%s'
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        if self.val != val:
            discrete_val = int(int(val / self.inc) * self.inc)
            xy = self.poly.xy
            xy[2] = discrete_val, 1
            xy[3] = discrete_val, 0
            self.poly.xy = xy
            self.valtext.set_text(self.valfmt % discrete_val)
            if self.drawon:
                self.ax.figure.canvas.draw()
            self.val = val
            if not self.eventson:
                return
            for cid, func in self.observers.items():
                func(discrete_val)

    def update_val_external(self, val):
        self.set_val(val)


class NumPolygon(plt.Polygon):

    def __init__(self, *args, **kwargs):
        self.track_id = kwargs.pop('track_id', 0)
        super(NumPolygon, self).__init__(*args, **kwargs)

