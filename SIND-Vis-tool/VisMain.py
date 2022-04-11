import argparse
from intersection_visualizer import Visualizer


def args_parser():
    configs = argparse.ArgumentParser(description='')

    configs.add_argument('--path', default="../data/",
                         help="Dir with track files", type=str)
    configs.add_argument('--record_name', default="8_02_1",
                         help="Dir with track files", type=str)
    configs.add_argument('--plot_traffic_light', default=True,
                         help="Optional: decide whether to plot the traffic light state or not.",
                         type=bool)

    configs.add_argument('--behaviour_type', default=True,
                         help="Optional: decide whether to show the vehicle's violation behavior by color of text or not.",
                         type=bool)

    configs.add_argument('--skip_n_frames', default=3,
                         help="Skip n frames when using the second skip button.",
                         type=int)
    configs.add_argument('--plotTrackingLines', default=True,
                                      help="Optional: decide whether to plot the direction lane intersection points or not.",
                                      type=bool)
    configs.add_argument('--plotFutureTrackingLines', default=True,
                                      help="Optional: decide whether to plot the tracking lines or not.",
                                      type=bool)
    
    configs = vars(configs.parse_args())

    return configs


if __name__ == "__main__":
    config = args_parser()
    visualization_plot = Visualizer(config)
    visualization_plot.show()
