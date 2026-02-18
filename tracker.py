class CentroidTracker:
    def __init__(self, max_disappeared=20, max_distance=120):
        self.next_id = 1
        self.objects = {}
        self.history = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections, frame_idx):
        updated = {}
        for box in detections:
            updated[self.next_id] = box
            x1, y1, x2, y2 = box
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)
            self.history.setdefault(self.next_id, []).append((frame_idx, (cx, cy)))
            self.next_id += 1
        return updated
