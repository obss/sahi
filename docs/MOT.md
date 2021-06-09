# MOT Utilities

## MOT dataset creation steps:

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

- after adding all frames, your MOT formatted files are ready at 'mot_video/' folder.
