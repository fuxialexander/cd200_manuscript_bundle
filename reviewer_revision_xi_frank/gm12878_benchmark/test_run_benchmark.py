"""Verify recovery preserves experiments and refuses a duplicate GPU queue."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("benchmark_supervisor", Path(__file__).with_name("run_benchmark.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class RecoveryChecks(unittest.TestCase):
    def test_k562_waits_for_existing_queue_and_uses_its_own_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            cache = root / "cache"
            cache.mkdir()
            (cache / "cache_manifest.json").write_text('{"status":"complete"}')
            other = root / "other_status.json"
            other.write_text('{"state":"running"}')
            commands, events = [], []

            def wait(seconds):
                events.append("wait")
                other.write_text('{"state":"complete"}')

            def launch(command, **kwargs):
                commands.append(command)
                events.append("gpu-stage")
                if "released" in command:
                    d = Path(command[command.index("--output-dir") + 1]) / "test_predictions"
                    d.mkdir(parents=True)
                    (d / "prediction_metadata.json").write_text("{}")
                return mock.Mock(pid=123456, returncode=0, poll=lambda: 0)

            args = ["run_benchmark.py", "--benchmark-dir", str(root), "--cache-dir", str(cache),
                    "--dataset", "K562", "--experiment", "ENCSR868FGK", "--annotation", "ENCSR467RSV",
                    "--count-weight", "123.4", "--wait-for-benchmark", str(other), "--run-published-inference"]
            with mock.patch.object(M, "BASE", root), mock.patch.object(M.sys, "argv", args), \
                 mock.patch.object(M.time, "sleep", side_effect=wait), \
                 mock.patch.object(M.subprocess, "Popen", side_effect=launch), \
                 mock.patch.object(M, "summarize") as summarize:
                M.main()
            self.assertEqual(events[0], "wait")
            for command in commands:
                if "--bias-model" in command:
                    self.assertTrue(command[command.index("--bias-model") + 1].endswith("ENCSR868FGK.h5"))
                if "--count-weight" in command:
                    self.assertEqual(command[command.index("--count-weight") + 1], "123.4")
            self.assertEqual(summarize.call_args.args[-2:], ("K562", "ENCSR868FGK"))
            status = json.loads((root / "benchmark_status.json").read_text())
            self.assertEqual(status["configuration"]["annotation"], "ENCSR467RSV")

    def test_recovery_reuses_complete_explainn_and_preserves_partial_modl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            cache = root / "cache"
            cache.mkdir()
            (cache / "cache_manifest.json").write_text('{"status":"complete"}')
            prior = {"state": "running", "supervisor_pid": 999999, "stages": []}
            (root / "benchmark_status.json").write_text(json.dumps(prior))
            for name in ("chrombpnet_published_fold0", "explainn_seed1234"):
                p = root / "runs" / name / "test_predictions"
                p.mkdir(parents=True)
                (p / "prediction_metadata.json").write_text("{}")
            (root / "runs/explainn_seed1234/training_result.json").write_text('{"pilot":false}')
            partial = root / "runs/modl_seed1234"
            partial.mkdir()
            (partial / "best.pt").write_bytes(b"preserve-checkpoint")
            commands = []

            def launch(command, **kwargs):
                commands.append(command)
                return mock.Mock(pid=123456, returncode=0, poll=lambda: 0)

            args = ["run_benchmark.py", "--cache-dir", str(cache), "--restart-interrupted"]
            with mock.patch.object(M, "BASE", root), mock.patch.object(M.sys, "argv", args), \
                 mock.patch.object(M.os, "kill", side_effect=ProcessLookupError), \
                 mock.patch.object(M.subprocess, "Popen", side_effect=launch), \
                 mock.patch.object(M, "summarize"):
                M.main()
            self.assertEqual((partial / "best.pt").read_bytes(), b"preserve-checkpoint")
            torch_commands = [c for c in commands if "--model" in c]
            self.assertEqual(len(torch_commands), 2)
            for command in torch_commands:
                self.assertEqual(command[command.index("--model") + 1], "modl")
                self.assertIn("modl_seed1234_restart_", command[command.index("--output-dir") + 1])
            status = json.loads((root / "benchmark_status.json").read_text())
            self.assertEqual(status["state"], "complete")
            self.assertTrue(list(root.glob("benchmark_status.restart_*.json")))

    def test_recovery_refuses_an_existing_supervisor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "benchmark_status.json"
            original = '{"state":"running","supervisor_pid":12345,"stages":[]}'
            path.write_text(original)
            with mock.patch.object(M, "BASE", root), \
                 mock.patch.object(M.sys, "argv", ["run_benchmark.py", "--restart-interrupted"]), \
                 mock.patch.object(M.os, "kill", return_value=None), self.assertRaises(SystemExit):
                M.main()
            self.assertEqual(path.read_text(), original)


if __name__ == "__main__":
    unittest.main()
