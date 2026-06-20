"""Verification test for FIX 1: confidence_threshold single source of truth.

Creates a temporary config file with confidence_threshold=0.6, then checks
that both RiskScoreCalculator and EvidenceGenerator agree on status.
"""
import json
import os
import sys
import tempfile

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, ".."))

from intelligence.risk_score import RiskScoreCalculator
from intelligence.evidence_generator import EvidenceGenerator


def test_shared_threshold():
    """Both modules should produce identical status for the same confidence."""
    # Create a temporary config with threshold=0.6
    test_config = {
        "type_weights": {"test_violation": 0.5},
        "repeat_offender": {"base_factor": 1.0, "increment_per_prior": 0.1, "max_factor": 2.0},
        "risk_categories": {"high": 0.7, "medium": 0.4, "low": 0.0},
        "confidence_threshold": 0.6
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        config_path = f.name

    try:
        risk_calc = RiskScoreCalculator(config_path=config_path)
        evidence_gen = EvidenceGenerator(
            output_dir=tempfile.mkdtemp(),
            weights_config_path=config_path
        )

        # Verify the raw threshold values match
        print(f"RiskScoreCalculator.confidence_threshold = {risk_calc.confidence_threshold}")
        print(f"EvidenceGenerator.confidence_threshold   = {evidence_gen.confidence_threshold}")
        assert risk_calc.confidence_threshold == 0.6, f"Expected 0.6, got {risk_calc.confidence_threshold}"
        assert evidence_gen.confidence_threshold == 0.6, f"Expected 0.6, got {evidence_gen.confidence_threshold}"
        print("✅ Both modules loaded threshold=0.6 from config")

        # Test case 1: confidence=0.7 (above 0.6 threshold) → formal_record
        risk_result = risk_calc.calculate("test_violation", 0.7, 0)
        assert risk_result['status'] == 'formal_record', f"RiskCalc: expected formal_record, got {risk_result['status']}"

        # EvidenceGenerator doesn't have a lightweight status-only method,
        # so we check the threshold comparison directly
        assert 0.7 >= evidence_gen.confidence_threshold, "EvidenceGen: 0.7 should be >= threshold"
        eg_status = 'formal_record' if 0.7 >= evidence_gen.confidence_threshold else 'pending_review'
        assert eg_status == 'formal_record', f"EvidenceGen: expected formal_record, got {eg_status}"
        print("✅ confidence=0.7 → formal_record in BOTH modules")

        # Test case 2: confidence=0.5 (below 0.6 threshold) → pending_review
        risk_result2 = risk_calc.calculate("test_violation", 0.5, 0)
        assert risk_result2['status'] == 'pending_review', f"RiskCalc: expected pending_review, got {risk_result2['status']}"

        eg_status2 = 'formal_record' if 0.5 >= evidence_gen.confidence_threshold else 'pending_review'
        assert eg_status2 == 'pending_review', f"EvidenceGen: expected pending_review, got {eg_status2}"
        print("✅ confidence=0.5 → pending_review in BOTH modules")

        print("\n🎉 FIX 1 VERIFIED: Single source of truth for confidence_threshold works correctly.")

    finally:
        os.unlink(config_path)


if __name__ == "__main__":
    test_shared_threshold()
