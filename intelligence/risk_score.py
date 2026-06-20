"""Risk score calculation for traffic violations."""

import json
import os

__all__ = ['RiskScoreCalculator']


class RiskScoreCalculator:
    """
    Calculates risk scores for violations.
    
    Formula: risk_score = type_weight * confidence * repeat_offender_factor
    repeat_offender_factor = min(base + increment * prior_count, max_factor)
    """

    def __init__(self, config_path='config/violation_weights.json'):
        """
        Args:
            config_path: Path to violation weights config.
        """
        self.config = self._load_config(config_path)
        self.type_weights = self.config.get('type_weights', {})
        self.repeat_config = self.config.get('repeat_offender', {})
        self.risk_thresholds = self.config.get('risk_categories', {})
        self.confidence_threshold = self.config.get('confidence_threshold', 0.85)

    def _load_config(self, config_path):
        """Load risk configuration from JSON file."""
        default_config = {
            'type_weights': {
                'red_light': 1.0, 'wrong_side': 0.9, 'triple_riding': 0.7,
                'helmet': 0.6, 'seatbelt': 0.6, 'anpr_flag': 0.5,
                'illegal_parking': 0.4
            },
            'repeat_offender': {
                'base_factor': 1.0, 'increment_per_prior': 0.1, 'max_factor': 2.0
            },
            'risk_categories': {'high': 0.7, 'medium': 0.4, 'low': 0.0},
            'confidence_threshold': 0.85
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return default_config

    def calculate(self, violation_type, confidence, prior_violation_count=0):
        """
        Calculate risk score for a violation.

        Args:
            violation_type: Type of violation (e.g., 'helmet', 'red_light').
            confidence: Detection confidence (0-1).
            prior_violation_count: Number of prior violations for this offender.

        Returns:
            Dict with risk_score, risk_category, repeat_offender_factor, status.
        """
        # Get type weight (default 0.5 for unknown types)
        type_weight = self.type_weights.get(violation_type, 0.5)

        # Calculate repeat offender factor
        base = self.repeat_config.get('base_factor', 1.0)
        increment = self.repeat_config.get('increment_per_prior', 0.1)
        max_factor = self.repeat_config.get('max_factor', 2.0)
        repeat_factor = min(base + increment * prior_violation_count, max_factor)

        # Calculate risk score
        risk_score = type_weight * confidence * repeat_factor

        # Determine risk category
        risk_category = self._categorize_risk(risk_score)

        # Determine status based on confidence
        status = 'formal_record' if confidence >= self.confidence_threshold else 'pending_review'

        return {
            'risk_score': round(risk_score, 4),
            'risk_category': risk_category,
            'repeat_offender_factor': round(repeat_factor, 2),
            'type_weight': type_weight,
            'status': status
        }

    def _categorize_risk(self, risk_score):
        """Categorize risk score into Low/Medium/High."""
        high_threshold = self.risk_thresholds.get('high', 0.7)
        medium_threshold = self.risk_thresholds.get('medium', 0.4)

        if risk_score >= high_threshold:
            return 'High'
        elif risk_score >= medium_threshold:
            return 'Medium'
        else:
            return 'Low'

    def get_severity_tier(self, risk_score):
        """
        Map risk score to alert severity tier for the alert router.
        
        Returns: 'CRITICAL', 'URGENT', or 'ROUTINE'
        """
        if risk_score >= 0.8:
            return 'CRITICAL'
        elif risk_score >= 0.5:
            return 'URGENT'
        else:
            return 'ROUTINE'
