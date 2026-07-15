# Semantic Scenario Video Preview

Generate a self-contained Bokeh HTML viewer from SinD semantic labels and
processed PKL trajectories:

```bash
python semantic_preview/build_semantic_video_preview.py \
  --labels datasets/SinD_dataset/Semantic_labels/scenarios.json \
  --data-dir datasets/SinD_dataset \
  --output semantic_preview/semantic_video_preview.html \
  --max-scenarios 24
```

The generated page supports filtering by semantic tag, scene, and scenario id,
then plays the selected traffic clip with agent boxes, headings, history
trails, ego highlighting, and the static map background.

For quick testing with the committed sample labels:

```bash
python semantic_preview/build_semantic_video_preview.py \
  --labels semantic_preview/sample_data/scenarios_sample.json \
  --data-dir /path/to/SinD_dataset \
  --max-scenarios 12
```

The committed sample label file contains a broader set of examples across
semantic categories and locations. Increase `--max-scenarios` when you want a
larger self-contained HTML; larger values produce larger HTML files.
