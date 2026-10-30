# RoadGuard Streamlit starter

## Files
- app.py: Streamlit prototype
- requirements.txt: app dependencies
- best.pt: copy your trained YOLO weights into this folder before running/deploying

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important prototype limitations
- Class labels currently display as class IDs because the exported dataset YAML uses names 0-4. Confirm the true mapping before describing detections as potholes or cracks.
- Severity is operator-confirmed. The suggested severity uses bounding-box area only as a demo heuristic; it is not a validated engineering risk assessment.
- GPS is entered manually. Automatic GPS capture from a vehicle/mobile device is not implemented.
- Repeat observations are matched when GPS coordinates are within 25 metres. This is a prototype rule and needs field validation.
- Priority score is a proposed formula, not a validated maintenance standard.
- Government/municipal reporting is not integrated.
- SQLite database is local to the app environment. For hosted deployment, use persistent storage or a managed database if observation history must survive app restarts.
