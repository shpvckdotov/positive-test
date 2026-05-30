"""Unit tests for log preprocessing."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from app.services.preprocessing import LogPreprocessor, _normalize


class TestNormalize:
    # _normalize() lowercases everything, so tokens are <ip>, <num>, <path>, etc.
    def test_ip_replaced(self):
        assert "<ip>" in _normalize("connection from 192.168.1.1")

    def test_numbers_replaced(self):
        assert "<num>" in _normalize("failed 3 times")

    def test_path_replaced(self):
        assert "<path>" in _normalize("reading /etc/passwd now")

    def test_uuid_replaced(self):
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        assert "<uuid>" in _normalize(f"uuid={uuid}")


class TestLinuxParser:
    def setup_method(self):
        self.p = LogPreprocessor()

    def test_sshd_auth_failure(self):
        line = "Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; logname= uid=0 rhost=218.188.2.4"
        result = self.p.parse(line, "linux")
        assert result.source_type == "linux"
        assert result.hostname == "combo"
        assert result.process_name == "sshd(pam_unix)"
        assert "authentication failure" in result.message

    def test_fallback_on_unknown_format(self):
        line = "this is not a standard log"
        result = self.p.parse(line, "linux")
        assert result.source_type == "linux"
        assert result.raw == line

    def test_normalized_message_created(self):
        line = "Jun 14 15:16:01 combo sshd[1234]: Failed password for root from 10.0.0.1 port 22 ssh2"
        result = self.p.parse(line, "linux")
        assert result.normalized_message != ""
        assert "<ip>" in result.normalized_message


class TestWindowsParser:
    def setup_method(self):
        self.p = LogPreprocessor()

    def test_windows_event_log(self):
        line = "2004-11-04 03:26:32, Information, 4624, Security, An account was successfully logged on."
        result = self.p.parse(line, "windows")
        assert result.source_type == "windows"
        assert result.log_level == "Information"
        assert result.event_id == "4624"

    def test_windows_normalized(self):
        line = "2004-11-04 03:26:33, Warning, 4625, Security, An account failed to log on from 192.168.1.1"
        result = self.p.parse(line, "windows")
        assert "<ip>" in result.normalized_message


class TestHDFSParser:
    def setup_method(self):
        self.p = LogPreprocessor()

    def test_hdfs_info(self):
        line = "081109 204005 1 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1234 src: /10.0.0.1:54106 dest: /10.0.0.2:50010"
        result = self.p.parse(line, "hdfs")
        assert result.source_type == "hdfs"
        assert result.log_level == "INFO"
        assert result.process_name == "dfs.DataNode$DataXceiver"
