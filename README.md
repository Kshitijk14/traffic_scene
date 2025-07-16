# traffic_scene

```
python inference_zone.py --roboflow_api_key <ROBOFLOW API KEY> --source_video_path data/vehicle-counting.mp4 --confidence_threshold 0.3 --iou_threshold 0.5 --target_video_path data/result_vehicle-counting.mp4
```

```
python ultralytics_zone.py --source_weights_path yolov8n.pt --source_video_path data/vehicle-counting.mp4 --confidence_threshold 0.3 --iou_threshold 0.5 --target_video_path data/result_vehicle-counting.mp4
```