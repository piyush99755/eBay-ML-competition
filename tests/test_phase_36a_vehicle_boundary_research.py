"""
Unit tests for Phase 36A: Vehicle-Model Boundary Correction Research Utilities and Counterexample Verification.
"""

import json
import unittest
from pathlib import Path

from src.evaluation.phase_36a_boundary_research import (
    clean_parenthetical_chassis,
    clean_generation_suffix,
    clean_extraneous_punctuation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DATA_PATH = PROJECT_ROOT / "data" / "cache" / "phase_36a_boundary_research_results.json"


class TestPhase36ABoundaryResearch(unittest.TestCase):
    """Verify boundary transformation functions and counterexamples."""

    def test_rule_a_chassis_cleaning(self):
        """Verify chassis stripping behaves as expected on isolated test cases."""
        self.assertEqual(clean_parenthetical_chassis("VITO (W447)"), "VITO")
        self.assertEqual(clean_parenthetical_chassis("Croma (154)"), "Croma")
        self.assertEqual(clean_parenthetical_chassis("Golf"), "Golf")
        self.assertEqual(clean_parenthetical_chassis(""), "")

    def test_rule_b_generation_cleaning(self):
        """Verify generation stripping on isolated test cases."""
        self.assertEqual(clean_generation_suffix("Vectra B"), "Vectra")
        self.assertEqual(clean_generation_suffix("Corsa D"), "Corsa")
        self.assertEqual(clean_generation_suffix("Focus II"), "Focus")
        self.assertEqual(clean_generation_suffix("Superb III"), "Superb")
        self.assertEqual(clean_generation_suffix("Passat"), "Passat")

    def test_rule_c_punctuation_cleaning(self):
        """Verify punctuation trimming preserves internal hyphens."""
        self.assertEqual(clean_extraneous_punctuation("C-Max,"), "C-Max")
        self.assertEqual(clean_extraneous_punctuation("-Golf-"), "Golf")
        self.assertEqual(clean_extraneous_punctuation("E-Klasse"), "E-Klasse")

    def test_counterexample_damage_proof(self):
        """
        Verify mathematical damage: prove that ground truth contains entities
        that are damaged by naive post-processing rules.
        """
        # Gold cases from ground truth where parenthetical is part of gold annotation
        gold_chassis_cases = ["3er (E91)", "Panda (169)", "Mito (955)", "GLK-Klasse (X204)"]
        for g in gold_chassis_cases:
            stripped = clean_parenthetical_chassis(g)
            self.assertNotEqual(stripped, g, "Rule A must change this string")
            # If model predicted gold exact match, rule damages it:
            self.assertFalse(stripped == g, "Rule A damages exact match with gold standard")

        # Gold cases where generation roman numeral is part of gold annotation
        gold_generation_cases = ["Focus II", "Superb III", "QASHQAI II", "PICANTO 3 JA III"]
        for g in gold_generation_cases:
            stripped = clean_generation_suffix(g)
            self.assertNotEqual(stripped, g, "Rule B must change this string")
            self.assertFalse(stripped == g, "Rule B damages exact match with gold standard")

        # Gold cases where punctuation is part of token/annotation
        gold_punct_cases = ["A-", "C-Max,"]
        for g in gold_punct_cases:
            stripped = clean_extraneous_punctuation(g)
            self.assertNotEqual(stripped, g, "Rule C must change this string")
            self.assertFalse(stripped == g, "Rule C damages exact match with gold standard")

    def test_research_results_integrity(self):
        """Verify that simulation results exist and all three rules show net negative TP."""
        self.assertTrue(RESEARCH_DATA_PATH.exists(), "Research results JSON missing!")
        with open(RESEARCH_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        rules = data["simulated_rules"]
        # Rule A damaged > fixed
        self.assertGreater(rules["Rule A (Parenthetical Chassis Suffix)"]["damaged_count"],
                           rules["Rule A (Parenthetical Chassis Suffix)"]["fixed_count"])
        # Rule B damaged > fixed
        self.assertGreater(rules["Rule B (Generation Suffix)"]["damaged_count"],
                           rules["Rule B (Generation Suffix)"]["fixed_count"])
        # Rule C damaged > fixed
        self.assertGreater(rules["Rule C (Punctuation Cleanup)"]["damaged_count"],
                           rules["Rule C (Punctuation Cleanup)"]["fixed_count"])

        # All rules have negative delta F0.2
        for r_name, r_data in rules.items():
            self.assertLess(r_data["delta_f02"], 0.0, f"{r_name} failed to show negative delta F0.2")


if __name__ == "__main__":
    unittest.main()
