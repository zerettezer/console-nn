"""Tests for domain holdout, calibration, baselines and contrast pairs."""
from __future__ import annotations

import unittest

import numpy as np

from console_nn.baselines import regex_status
from console_nn.calibration import (
    expected_calibration_error,
    fit_temperature,
    risk_coverage,
    softmax,
)
from console_nn.config import SynthConfig
from console_nn.contrast import build_contrast_pairs
from console_nn.domain_holdout import DOMAIN_GROUPS, group_of


class DomainGroupTest(unittest.TestCase):
    def test_groups_cover_every_generator_domain(self):
        configured = set(SynthConfig().domains)
        covered = {domain for values in DOMAIN_GROUPS.values() for domain in values}
        self.assertEqual(configured, covered)
        self.assertEqual(len(covered), len(configured))

    def test_unknown_domain_falls_back(self):
        self.assertEqual(group_of({"domain": "python"}), "python")
        self.assertEqual(group_of({"domain": "not-a-domain"}), "other")


class CalibrationTest(unittest.TestCase):
    def test_temperature_recovers_synthetic_scale(self):
        rng = np.random.default_rng(4)
        logits = rng.normal(size=(4000, 4)) * 2.0
        true_temperature = 2.5
        probs = softmax(logits / true_temperature)
        labels = np.asarray(
            [rng.choice(4, p=row) for row in probs], dtype=int
        )
        fitted = fit_temperature(logits, labels)
        self.assertAlmostEqual(fitted["temperature"], true_temperature, delta=0.35)
        self.assertLessEqual(fitted["nll_after"], fitted["nll_before"])

    def test_ece_and_risk_coverage_shapes(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(200, 4))
        probs = softmax(logits)
        labels = np.argmax(logits, axis=1)
        ece = expected_calibration_error(probs, labels)
        self.assertGreaterEqual(ece["ece"], 0.0)
        self.assertLessEqual(ece["ece"], 1.0)
        curve = risk_coverage(probs, labels, ["success", "fail", "block", "unknown"])
        coverage = [point["coverage"] for point in curve["curve"]]
        self.assertEqual(coverage, sorted(coverage))


class BaselineTest(unittest.TestCase):
    def test_regex_priority(self):
        self.assertEqual(regex_status("Permission denied"), "block")
        self.assertEqual(regex_status("Traceback: ValueError: bad"), "fail")
        self.assertEqual(regex_status("build finished"), "success")


class ContrastPairTest(unittest.TestCase):
    def test_same_command_different_labels(self):
        documents = [
            {"platform": "linux", "command": "pytest", "status": "success"},
            {"platform": "linux", "command": "pytest", "status": "fail"},
            {"platform": "linux", "command": "make", "status": "success"},
        ]
        pairs, ordered = build_contrast_pairs(documents)
        self.assertEqual(len(ordered), 3)
        self.assertEqual(len(pairs), 1)
        left, right = pairs[0]
        self.assertEqual(documents[left]["command"], documents[right]["command"])
        self.assertNotEqual(documents[left]["status"], documents[right]["status"])


if __name__ == "__main__":
    unittest.main()
