# MOT Utilities

<details closed>
<summary>
<big><b>MOT Challenge formatted ground truth dataset creation:</b></big>
</summary>

- import required classes:

```python
from sahi.utils.mot import MotAnnotation, MotFrame, MotVideo
```

- init video:

```python
mot_video = MotVideo(name="sequence_name")
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

- export in MOT challenge format:

```python
mot_video.export(export_dir="mot_data", type="gt")
```

- your MOT challenge formatted ground truth files are ready under `mot_data/sequence_name/` folder.
</details>

<details closed>
<summary>
<big><b>Advanced MOT Challenge formatted ground truth dataset creation:</b></big>
</summary>

- you can customize tracker while initializing mot video object:

```python
tracker_params = {
  'distance_threshold': 30,
  'detection_threshold': 0,
  'hit_inertia_min': 10,
  'hit_inertia_max': 12,
  'point_transience': 4,
}
# for details: https://github.com/tryolabs/norfair/tree/master/docs#arguments

mot_video = MotVideo(tracker_kwargs=tracker_params)
```

- you can omit automatic track id generation and directly provide track ids of annotations:


```python
# create annotations with track ids:
mot_frame.add_annotation(
  MotAnnotation(bbox=[x_min, y_min, width, height], track_id=1)
)

mot_frame.add_annotation(
  MotAnnotation(bbox=[x_min, y_min, width, height], track_id=2)
)

# add frame to video:
mot_video.add_frame(mot_frame)

# export in MOT challenge format without automatic track id generation:
mot_video.export(export_dir="mot_data", type="gt", use_tracker=False)
```

- you can overwrite the results into already present directory by adding `exist_ok=True`:

```python
mot_video.export(export_dir="mot_data", type="gt", exist_ok=True)
```

- your MOT challenge formatted ground truth files are ready at `mot_data/sequence_name/gt/gt.txt`.
</details>

<details closed>
<summary>
<big><b>MOT Challenge formatted tracker output creation:</b></big>
</summary>

- import required classes:

```python
from sahi.utils.mot import MotAnnotation, MotFrame, MotVideo
```

- init video by providing video name:

```python
mot_video = MotVideo(name="sequence_name")
```

- init first frame:

```python
mot_frame = MotFrame()
```

- add tracker outputs to frame:

```python
mot_frame.add_annotation(
  MotAnnotation(bbox=[x_min, y_min, width, height], track_id=1)
)

mot_frame.add_annotation(
  MotAnnotation(bbox=[x_min, y_min, width, height], track_id=2)
)
```

- add frame to video:

```python
mot_video.add_frame(mot_frame)
```

- export in MOT challenge format:

```python
mot_video.export(export_dir="mot_data", type="det")
```

- your MOT challenge formatted detection output file is ready at `mot_data/sequence_name/det/det.txt`.
</details>

<details closed>
<summary>
<big><b>Advanced MOT Challenge formatted detection output creation:</b></big>
</summary>

- you can overwrite the results into already present directory by adding `exist_ok=True`:

```python
mot_video.export(export_dir="mot_data", type="det", exist_ok=True)
```
</details>
