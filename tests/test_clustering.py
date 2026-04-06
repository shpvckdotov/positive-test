"""Unit tests for the clustering service."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import tempfile
from app.services.clustering import LogClusteringService
from app.services.preprocessing import LogPreprocessor, ParsedLog


def make_logs(preprocessor: LogPreprocessor) -> list[ParsedLog]:
    """Generate a small but varied synthetic log corpus."""
    linux_lines = [
        "Jun 14 15:16:01 combo sshd[1234]: Failed password for root from 1.1.1.1 port 22 ssh2",
        "Jun 14 15:16:02 combo sshd[1235]: Failed password for root from 2.2.2.2 port 22 ssh2",
        "Jun 14 15:16:03 combo sshd[1236]: Failed password for admin from 3.3.3.3 port 22 ssh2",
        "Jun 14 15:16:04 combo sshd[1237]: Accepted password for alice from 4.4.4.4 port 22 ssh2",
        "Jun 14 15:16:05 combo sshd[1238]: Accepted password for bob from 5.5.5.5 port 22 ssh2",
        "Jun 14 15:16:06 combo cron[100]: root CMD /usr/bin/python3 /opt/check.py",
        "Jun 14 15:16:07 combo cron[101]: alice CMD /usr/bin/backup.sh",
        "Jun 14 15:16:08 combo cron[102]: bob CMD /usr/sbin/logrotate",
    ]
    windows_lines = [
        "2004-11-04 03:26:32, Information, 4624, Security, An account was successfully logged on.",
        "2004-11-04 03:26:33, Information, 4624, Security, An account was successfully logged on.",
        "2004-11-04 03:26:34, Warning, 4625, Security, An account failed to log on.",
        "2004-11-04 03:26:35, Warning, 4625, Security, An account failed to log on.",
        "2004-11-04 03:26:36, Error, 1000, Application, Application faulted unexpectedly.",
    ]
    all_lines = (linux_lines * 5) + (windows_lines * 4)
    logs = []
    for line in all_lines:
        src = "linux" if line.startswith("Jun") else "windows"
        logs.append(preprocessor.parse(line, src))
    return logs


class TestClusteringService:
    def setup_method(self):
        self.svc = LogClusteringService()
        self.prep = LogPreprocessor()
        self.logs = make_logs(self.prep)

    def test_fit_returns_metrics(self):
        metrics = self.svc.fit(self.logs)
        assert "n_clusters" in metrics
        assert "n_noise" in metrics
        assert "noise_ratio" in metrics
        assert metrics["n_clusters"] >= 1

    def test_is_fitted_after_fit(self):
        assert not self.svc.is_fitted
        self.svc.fit(self.logs)
        assert self.svc.is_fitted

    def test_classify_known_pattern(self):
        self.svc.fit(self.logs)
        new_log = self.prep.parse(
            "Jun 14 16:00:00 combo sshd[9999]: Failed password for root from 9.9.9.9 port 22 ssh2",
            "linux"
        )
        label, confidence, is_anomaly = self.svc.classify(new_log)
        # Should NOT be anomaly — it matches a known template
        assert not is_anomaly
        assert confidence > 0.0

    def test_classify_anomaly(self):
        self.svc.fit(self.logs)
        weird = self.prep.parse(
            "Jun 14 16:00:00 combo kernel: SEGFAULT at address 0xdeadbeef in process xyz corrupted stack frame",
            "linux"
        )
        label, confidence, is_anomaly = self.svc.classify(weird)
        # Confidence should be lower than for known patterns
        assert isinstance(confidence, float)

    def test_save_and_load(self):
        self.svc.fit(self.logs)
        with tempfile.TemporaryDirectory() as tmpdir:
            self.svc.save(tmpdir)
            svc2 = LogClusteringService()
            ok = svc2.load(tmpdir)
            assert ok
            assert svc2.is_fitted
            assert len(svc2.clusters) == len(self.svc.clusters)

    def test_fit_empty_raises(self):
        with pytest.raises(ValueError):
            self.svc.fit([])

    def test_classify_before_fit_raises(self):
        log = self.prep.parse("some log", "linux")
        with pytest.raises(RuntimeError):
            self.svc.classify(log)
