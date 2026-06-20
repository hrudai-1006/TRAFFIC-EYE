"""Evidence generation and management for traffic violations."""

import json
import uuid
import os
from datetime import datetime, timezone

import cv2
import numpy as np

__all__ = ['EvidenceGenerator']


class EvidenceGenerator:
    """Generates annotated evidence images and structured JSON records for violations."""

    def __init__(self, output_dir='evidence_store',
                 config_path='config/camera_locations.json',
                 weights_config_path='config/violation_weights.json',
                 confidence_threshold=None):
        """
        Args:
            output_dir: Directory to save evidence images and JSON records.
            config_path: Path to camera locations config for location metadata.
            weights_config_path: Path to violation weights config (used to read
                confidence_threshold — must be the same file used by
                RiskScoreCalculator so there is a single source of truth).
            confidence_threshold: Explicit override. If provided, this value is
                used directly instead of reading from the weights config file.
                Useful for testing. Last-resort default is 0.85.
        """
        self.output_dir = os.path.abspath(output_dir)
        self.output_root = os.path.dirname(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'records'), exist_ok=True)

        # Load camera config
        self.camera_config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.camera_config = json.load(f).get('cameras', {})

        # Load confidence threshold from the shared violation-weights config,
        # mirroring the same load logic used by RiskScoreCalculator.
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        else:
            self.confidence_threshold = self._load_confidence_threshold(
                weights_config_path
            )

    @staticmethod
    def _load_confidence_threshold(weights_config_path):
        """Read confidence_threshold from the violation-weights JSON config.

        Falls back to 0.85 if the file is missing, malformed, or lacks the key.
        """
        default_threshold = 0.85
        if os.path.exists(weights_config_path):
            try:
                with open(weights_config_path, 'r') as f:
                    data = json.load(f)
                return data.get('confidence_threshold', default_threshold)
            except (json.JSONDecodeError, IOError):
                pass
        return default_threshold

    def generate_evidence(self, frame, violation_info, camera_id='CAM_001',
                          plate_info=None, risk_score=0.0, repeat_count=0,
                          risk_category='Low'):
        """
        Generate a complete evidence record with annotated image.

        Args:
            frame: Original video frame (numpy array).
            violation_info: Dict with keys: violation_type, confidence, vehicle_bbox,
                          vehicle_id, detail.
            camera_id: Camera identifier string.
            plate_info: Dict with plate_text and plate_confidence, or None.
            risk_score: Calculated risk score float.
            repeat_count: Number of prior violations for this plate.
            risk_category: 'Low', 'Medium', or 'High'.

        Returns:
            Dict: The complete evidence record.
        """
        evidence_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Determine status based on confidence threshold (read from config)
        confidence = violation_info.get('confidence', 0.0)
        status = 'formal_record' if confidence >= self.confidence_threshold else 'pending_review'

        # Get camera location from config
        cam_config = self.camera_config.get(camera_id, {})
        camera_location = cam_config.get('location_name', f'Unknown ({camera_id})')

        # Build evidence record
        record = {
            'evidence_id': evidence_id,
            'timestamp': timestamp,
            'camera_id': camera_id,
            'camera_location': camera_location,
            'violation_type': violation_info.get('violation_type', 'unknown'),
            'confidence': round(confidence, 4),
            'vehicle_plate': plate_info.get('plate_text') if plate_info else None,
            'plate_confidence': round(plate_info.get('plate_confidence', 0.0), 4) if plate_info else None,
            'risk_score': round(risk_score, 4),
            'repeat_offender_count': repeat_count,
            'risk_category': risk_category,
            'evidence_image_path': '',
            'status': status,
            'detail': violation_info.get('detail', {}),
            'frame_number': violation_info.get('frame_number'),
            'vehicle_id': violation_info.get('vehicle_id'),
        }

        # Generate annotated evidence image
        evidence_image = self._annotate_evidence_image(
            frame, violation_info, record, plate_info
        )

        # Save image
        image_filename = f"{evidence_id}.jpg"
        image_dir = os.path.join(self.output_dir, 'images')
        image_path = os.path.join(image_dir, image_filename)
        relative_image_path = os.path.relpath(image_path, start=self.output_root)

        write_success = False
        if not os.path.isdir(image_dir):
            print(f"   ⚠️  Evidence image directory missing: {image_dir}")
        elif evidence_image is None or getattr(evidence_image, 'size', 0) == 0:
            print(
                f"   ⚠️  Evidence image is empty for {evidence_id} - "
                "record will have no image reference."
            )
        else:
            write_success = cv2.imwrite(
                image_path,
                evidence_image,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

        if write_success:
            record['evidence_image_path'] = relative_image_path
        else:
            record['evidence_image_path'] = None
            print(
                f"   ⚠️  Failed to write evidence image to {image_path} - "
                "record will have no image reference."
            )

        # Save JSON record
        record_path = os.path.join(self.output_dir, 'records', f"{evidence_id}.json")
        with open(record_path, 'w') as f:
            json.dump(record, f, indent=2, default=str)

        return record

    def _annotate_evidence_image(self, frame, violation_info, record, plate_info):
        """Create annotated evidence image with overlays."""
        img = frame.copy()
        h, w = img.shape[:2]

        # Draw vehicle bounding box
        bbox = violation_info.get('vehicle_bbox')
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            # Red box for violation
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Violation type label above box
            label = f"VIOLATION: {record['violation_type'].upper()}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 10, y1), (0, 0, 255), -1)
            cv2.putText(img, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw plate text if available
        if plate_info and plate_info.get('plate_text'):
            plate_label = f"PLATE: {plate_info['plate_text']} ({plate_info.get('plate_confidence', 0):.0%})"
            cv2.putText(img, plate_label, (10, h - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Info overlay bar at bottom
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Overlay text
        info_line1 = (f"Camera: {record['camera_id']} | "
                      f"Location: {record['camera_location']} | "
                      f"Time: {record['timestamp'][:19]}")
        info_line2 = (f"Confidence: {record['confidence']:.0%} | "
                      f"Risk: {record['risk_category']} ({record['risk_score']:.2f}) | "
                      f"Status: {record['status'].upper()}")

        cv2.putText(img, info_line1, (10, h - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, info_line2, (10, h - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # TRAFFIC-EYE AI watermark
        cv2.putText(img, "TRAFFIC-EYE AI", (w - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        return img

    def load_all_records(self):
        """Load all evidence records from the store."""
        records = []
        records_dir = os.path.join(self.output_dir, 'records')
        if not os.path.exists(records_dir):
            return records

        for filename in os.listdir(records_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(records_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        records.append(json.load(f))
                except (json.JSONDecodeError, IOError):
                    continue

        return sorted(records, key=lambda r: r.get('timestamp', ''), reverse=True)

    def get_records_by_type(self, violation_type):
        """Get all records of a specific violation type."""
        return [r for r in self.load_all_records()
                if r.get('violation_type') == violation_type]

    def get_formal_records(self):
        """Get only formal (high-confidence) records."""
        return [r for r in self.load_all_records()
                if r.get('status') == 'formal_record']

    def get_pending_reviews(self):
        """Get records pending human review."""
        return [r for r in self.load_all_records()
                if r.get('status') == 'pending_review']
