# SinD Data Layouts

SinD 2.0 supports two local representations:

- raw 1.0-style CSV record folders;
- processed 2.0 PKL folders consumed by `trajdata`.

Only public CSV samples are committed in this repository.

## Raw CSV Samples

The included samples follow the original SinD 1.0 layout:

```text
Data/
├── Changchun/
│   ├── Changchun_Pudong.osm
│   ├── Changchun_Pudong.png
│   └── changchun_pudong_507_009/
│       ├── Ped_smoothed_tracks.csv
│       └── Traffic_Lights.csv
├── Chongqing/
│   └── 6_22_NR_1/
├── Tianjin/
│   └── 8_2_1/
└── Xi'an/
    └── Xi'an_412_m1/
```

CSV trajectory files contain per-frame state columns such as `track_id`,
`frame_id`, `timestamp_ms`, `agent_type`, `x`, `y`, `vx`, `vy`, optional
`heading_rad`, optional `length`, `width`, and optional acceleration columns.

## Processed PKL Layout

`trajdata` expects processed SinD data under a dataset root:

```text
datasets/SinD_dataset/
├── cc/
│   ├── tp_info_cc.pkl
│   ├── frame_data_cc.pkl
│   └── cc_map.json
├── xa/
│   ├── tp_info_xa.pkl
│   ├── frame_data_xa.pkl
│   └── xa_map.json
├── cqNR/
├── tj/
├── cqIR/
├── xasl/
├── cqR/
└── output_json/
    ├── cc_map.json
    ├── xa_map.json
    └── ...
```

Each location folder uses:

- `tp_info_<location>.pkl`: `scene_id -> track_id -> track_data`;
- `frame_data_<location>.pkl`: `scene_id -> frame_id -> list[agent_state]`;
- `<location>_map.json`: lightweight JSON map, or an equivalent file under
  `output_json/`.

Optional traffic-light PKLs can be generated as:

```text
datasets/SinD_dataset/<location>/traffic_lights_<location>.pkl
```

## Conversion

Convert a raw CSV record to processed PKL:

```bash
python tools/sind_csv_to_pkl.py \
  --record-dir Data/Tianjin/8_2_1 \
  --location tj \
  --scene-id 8_2_1 \
  --output-dir datasets/SinD_dataset
```

Regenerate `frame_data` from `tp_info`:

```bash
python tools/tp_info_to_frame_data.py \
  --tp-info datasets/SinD_dataset/tj/tp_info_tj.pkl \
  --output datasets/SinD_dataset/tj/frame_data_tj.pkl
```

Build optional local traffic-light PKLs from original traffic-light CSV folders:

```bash
python scripts/sind_build_traffic_light_pkls.py \
  --sind-data-dir datasets/SinD_dataset \
  --traffic-light-dir /path/to/raw/csv/root
```
