"""Agentic alert routing with ReAct-style decision loop."""

import json
import os
from datetime import datetime, timezone

__all__ = ['AlertRouter']


class AlertRouter:
    """
    ReAct-style alert routing agent.
    
    Given a violation evidence record, the agent:
    1. THINKS: Analyzes the violation severity based on risk score
    2. ACTS: Looks up the appropriate officer/zone for dispatch  
    3. OBSERVES: Formats and dispatches (simulates) the alert
    
    Note: Alert dispatch is SIMULATED (logged to console/file).
    No real WhatsApp/SMS API integration.
    """

    def __init__(self, officer_config_path='config/officer_zones.json',
                 alert_log_path='evidence_store/alerts.json'):
        """
        Args:
            officer_config_path: Path to officer-zone mapping config.
            alert_log_path: Path to save dispatched alerts log.
        """
        self.officers = self._load_officer_config(officer_config_path)
        self.alert_log_path = alert_log_path
        self.alerts = []

        # Load existing alerts if any
        if os.path.exists(alert_log_path):
            try:
                with open(alert_log_path, 'r') as f:
                    self.alerts = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.alerts = []

    def _load_officer_config(self, config_path):
        """Load officer-zone mapping from config."""
        default = {
            'officers': {
                'DEFAULT': {
                    'officer_name': 'Control Room',
                    'officer_id': 'TPS_CTRL',
                    'badge_number': 'BTP-0000',
                    'contact_placeholder': '+91-80-2294-3400',
                    'jurisdiction': 'Bengaluru City'
                }
            },
            'default_officer': {
                'officer_name': 'Control Room',
                'officer_id': 'TPS_CTRL',
                'badge_number': 'BTP-0000',
                'contact_placeholder': '+91-80-2294-3400',
                'jurisdiction': 'Bengaluru City'
            }
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return default

    def route_alert(self, evidence_record):
        """
        Process an evidence record through the ReAct decision loop.
        
        Args:
            evidence_record: Complete evidence record dict.
            
        Returns:
            Dict with alert details and routing decision.
        """
        # === THINK: Analyze severity ===
        risk_score = evidence_record.get('risk_score', 0.0)
        severity = self._classify_severity(risk_score)
        violation_type = evidence_record.get('violation_type', 'unknown')

        thought = (
            f"[THINK] Violation: {violation_type.upper()} | "
            f"Risk Score: {risk_score:.2f} | "
            f"Confidence: {evidence_record.get('confidence', 0):.0%} | "
            f"Repeat Offender Count: {evidence_record.get('repeat_offender_count', 0)} | "
            f"Severity Classification: {severity}"
        )
        print(thought)

        # === ACT: Look up officer/zone ===
        camera_id = evidence_record.get('camera_id', '')
        zone_id = self._get_zone_from_camera(camera_id)
        officer = self._lookup_officer(zone_id)

        action = (
            f"[ACT] Camera: {camera_id} -> Zone: {zone_id} -> "
            f"Officer: {officer['officer_name']} ({officer['badge_number']})"
        )
        print(action)

        # === OBSERVE: Format and dispatch alert ===
        alert_message = self._format_alert(evidence_record, severity, officer)
        
        alert_record = {
            'alert_id': f"ALERT-{len(self.alerts) + 1:04d}",
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'evidence_id': evidence_record.get('evidence_id'),
            'severity': severity,
            'zone_id': zone_id,
            'officer': officer,
            'message': alert_message,
            'dispatch_status': 'SIMULATED',
            'violation_type': violation_type,
            'risk_score': risk_score,
        }

        observe = (
            f"[OBSERVE] Alert {alert_record['alert_id']} dispatched "
            f"(SIMULATED) to {officer['officer_name']}"
        )
        print(observe)
        print(f"\n{'='*60}")
        print(alert_message)
        print(f"{'='*60}\n")

        # Log the alert
        self.alerts.append(alert_record)
        self._save_alerts()

        return alert_record

    def _classify_severity(self, risk_score):
        """Classify severity tier from risk score."""
        if risk_score >= 0.8:
            return 'CRITICAL'
        elif risk_score >= 0.5:
            return 'URGENT'
        else:
            return 'ROUTINE'

    def _get_zone_from_camera(self, camera_id):
        """Look up zone ID from camera config (simplified mapping)."""
        # Simple mapping based on camera ID
        zone_map = {
            'CAM_001': 'ZONE_A',
            'CAM_002': 'ZONE_B', 
            'CAM_003': 'ZONE_C',
        }
        return zone_map.get(camera_id, 'DEFAULT')

    def _lookup_officer(self, zone_id):
        """Look up the assigned officer for a zone."""
        officers = self.officers.get('officers', {})
        if zone_id in officers:
            return officers[zone_id]
        return self.officers.get('default_officer', {
            'officer_name': 'Control Room',
            'officer_id': 'TPS_CTRL',
            'badge_number': 'BTP-0000',
            'contact_placeholder': '+91-80-2294-3400',
            'jurisdiction': 'Bengaluru City'
        })

    def _format_alert(self, record, severity, officer):
        """Format a human-readable alert message."""
        severity_emoji = {'CRITICAL': '🚨', 'URGENT': '⚠️', 'ROUTINE': 'ℹ️'}
        emoji = severity_emoji.get(severity, 'ℹ️')

        plate = record.get('vehicle_plate', 'UNREADABLE')
        if not plate:
            plate = 'UNREADABLE'

        message = f"""
{emoji} TRAFFIC-EYE AI — {severity} ALERT {emoji}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Violation: {record.get('violation_type', 'Unknown').upper().replace('_', ' ')}
📍 Location: {record.get('camera_location', 'Unknown')}
📷 Camera: {record.get('camera_id', 'Unknown')}
🕐 Time: {record.get('timestamp', 'Unknown')[:19]}

🚗 Vehicle Plate: {plate}
📊 Confidence: {record.get('confidence', 0):.0%}
⚡ Risk Score: {record.get('risk_score', 0):.2f} ({record.get('risk_category', 'Unknown')})
🔄 Prior Violations: {record.get('repeat_offender_count', 0)}

👮 Assigned To: {officer['officer_name']}
🎖️ Badge: {officer['badge_number']}
📞 Contact: {officer['contact_placeholder']}
📌 Jurisdiction: {officer['jurisdiction']}

🔗 Evidence ID: {record.get('evidence_id', 'N/A')}
📸 Evidence Image: {record.get('evidence_image_path', 'N/A')}

[⚠️ THIS IS A SIMULATED ALERT — NO ACTUAL DISPATCH]
"""
        return message.strip()

    def _save_alerts(self):
        """Persist alerts to JSON file."""
        os.makedirs(os.path.dirname(self.alert_log_path), exist_ok=True)
        with open(self.alert_log_path, 'w') as f:
            json.dump(self.alerts, f, indent=2, default=str)

    def get_alert_stats(self):
        """Get alert statistics."""
        stats = {
            'total_alerts': len(self.alerts),
            'by_severity': {},
            'by_type': {},
        }
        for alert in self.alerts:
            sev = alert.get('severity', 'UNKNOWN')
            stats['by_severity'][sev] = stats['by_severity'].get(sev, 0) + 1

            vtype = alert.get('violation_type', 'unknown')
            stats['by_type'][vtype] = stats['by_type'].get(vtype, 0) + 1

        return stats
