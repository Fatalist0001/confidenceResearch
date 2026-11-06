"""Static guards for safety properties of scripts/run.ps1.

PowerShell variable names are case-insensitive. A local ``$profile`` therefore
shadows the top-level string parameter ``$Profile``; this once broke profile IDs in
the full workflow. Keep a small test so that refactors cannot reintroduce it.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers


RUNNER = os.path.join(helpers.ROOT, "scripts", "run.ps1")


class RunnerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(RUNNER, encoding="utf-8") as f:
            cls.text = f.read()

    def test_profile_id_is_not_shadowed_by_a_local_profile_variable(self):
        assignments = re.findall(r"(?im)^\s*\$profile\s*=", self.text)
        parameters = re.findall(r"(?im)^function[^\n]*\(\s*\$profile(?:\s*[,\)])", self.text)
        self.assertEqual(assignments, [], "use $profileConfig; $profile shadows -Profile")
        self.assertEqual(parameters, [], "use $profileConfig; $profile shadows -Profile")

    def test_only_pilot_is_labeled_inside_generation(self):
        start = self.text.index("function Invoke-Generation")
        end = self.text.index("function Get-RunDir", start)
        block = self.text[start:end]
        self.assertEqual(block.count("Join-Path $Pipeline 'label.py'"), 1)
        self.assertRegex(block, r"if \(\$kind -eq 'pilot'\)[\s\S]+class-balance gate")
        self.assertIn("'--questions', $questionSource", block)
        self.assertIn("Next: -Stage label, then verify, baseline and probe.", block)

    def test_required_transition_guards_are_present(self):
        for phrase in (
            "binary version mismatch:",
            "the model requires BOS id",
            "prompt check is missing:",
            "no pilot gate:",
            "pilot gate has no passed class-balance check:",
            "the pilot used a different profile:",
            "the pilot and main flags differ:",
            "verify.json missing: run -Stage verify first",
            "This runner never overwrites artifacts",
        ):
            self.assertIn(phrase, self.text)

    def test_batch_flags_are_not_built(self):
        start = self.text.index("function Build-BinaryArgs")
        end = self.text.index("function Assert-NotBlocked", start)
        block = self.text[start:end]
        self.assertNotRegex(block, r"['\"]-(?:b|ub)['\"]")

    def test_native_processes_are_not_piped_through_powershell(self):
        self.assertNotIn("Tee-Object", self.text)
        self.assertNotIn("| Out-Host", self.text)

    def test_gate_returns_only_the_process_exit_code(self):
        start = self.text.index("function Invoke-PilotGate")
        end = self.text.index("function Invoke-MetaPostprocessor", start)
        block = self.text[start:end]
        self.assertIn("Start-Process", block)
        self.assertIn("return $proc.ExitCode", block)


if __name__ == "__main__":
    unittest.main()
