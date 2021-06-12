# MOT Utilities

## MOT Challenge formatted dataset creation steps:

- import required classes:

```python
from sahi.utils.mot import MotAnnotation, MotFrame, MotVideo
```

- init video:

```python
mot_video = MotVideo(export_dir="mot_video")
```

- init first frame:

```python
mot_frame = MotFrame()
```

- add annotations to frame:

```python
mot_frame.add_annotation(
  MotAnnotation(bbox=[x_min, y_min, width, height])
)

mot_frame.add_annotation(
  MotAnnotation(bbox=[x_min, y_min, width, height])
)
```

- add frame to video:

```python
mot_video.add_frame(mot_frame)
```

- after adding all frames, your MOT formatted files are ready at `mot_video/` folder.

## Advanced MOT Challenge formatted dataset creation:

- you can customize tracker while initializing mot video object:

```python
tracker_params = {
  'max_distance_between_points': 30,
  'min_detection_threshold': 0,
  'hit_inertia_min': 10,
  'hit_inertia_max': 12,
  'point_transience': 4,
}
# for details: https://github.com/tryolabs/norfair/tree/master/docs#arguments

mot_video = MotVideo(export_dir="mot_video", tracker_kwargs=tracker_params)
```