# File Directory

For every scenario with name `<scenario_name>`, the scripts expect the map with name `<scenario_name>.osm` to be in folder `maps` and the track files to be in a subfolder with name `<scenario_name>` within folder `recorded_trackfiles`.

```bash
.
├── README.md
├── Format.md
├── doc
│   ├── File-Directory.md
│   └── ...
├── SIND-Vis-tool
│   ├──utils
│   │   ├── DataReader.py
│   │   ├── dict_utils.py
│   │   ├── map_vis_lanelet2.py
│   │   └── map_vis_without_lanelet.py 
│   ├── VisMain.py
│   ├── intersection_visualizer.py
│   ├── requirements.txt
│   └── README.md
└── Data
       ├── [recording_day_n] 
       │   ├── Ped_smoothed_tracks.csv
       │   ├── Ped_tracks_meta.csv
       │   ├── TraficLight_[recording_day_n].csv
       │   ├── Veh_smoothed_tracks.csv
       │   ├── Veh_tracks_meta.csv
       │   └──recoding_metas.csv
       └──mapfile-Tianjin.osm

```
