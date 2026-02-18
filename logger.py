# logger.py

violations = []   # stores all violations in memory before writing Excel

def log_violation(vtype, details, conf, img_path, timestamp):
    violations.append({
        "type": vtype,
        "details": details,
        "confidence": conf,
        "image_path": img_path,
        "timestamp": timestamp
    })
