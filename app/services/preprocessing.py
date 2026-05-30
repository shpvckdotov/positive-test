"""
Log preprocessing: parsing, normalization, and feature extraction.
Supports Linux syslog, Windows Event Logs, and HDFS logs.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Replacement tokens (Drain-style normalization) ──────────────────────────
_PATTERNS = [
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
    (re.compile(r"\b[0-9a-fA-F]{32,}\b"), "<HEX>"),
    (re.compile(r"(?<![/\w])/(?:[\w./-]+)"), "<PATH>"),           # unix paths
    (re.compile(r"[A-Za-z]:\\(?:[\w\\. ]+)+"), "<PATH>"),         # windows paths
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"), "<DATETIME>"),
    (re.compile(r"\b\d{2}:\d{2}:\d{2}\b"), "<TIME>"),
    (re.compile(r"\b\d+\b"), "<NUM>"),
    (re.compile(r"[^\w\s<>]"), " "),                              # strip punctuation (keep tokens)
]

# ── Linux syslog ─────────────────────────────────────────────────────────────
_LINUX_RE = re.compile(
    r"^(?P<month>\w+)\s+(?P<day>\d+)\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<hostname>\S+)\s+(?P<process>[^\[:\s]+)(?:\[(?P<pid>\d+)\])?:\s*(?P<message>.+)$"
)

# ── Windows Event Log (LogHub CSV-like format) ────────────────────────────────
# LogHub Windows dataset has columns: Date, Time, Level, EventID, Component, Content
_WINDOWS_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2}),\s*"
    r"(?P<level>\w+),\s*(?P<event_id>\d+),\s*(?P<component>[^,]+),\s*(?P<message>.+)$"
)
# Alternative Windows format (simpler)
_WINDOWS_RE2 = re.compile(
    r"^(?P<date>\S+)\s+(?P<time>\S+)\s+(?P<level>Information|Warning|Error|Critical|Verbose)\s+"
    r"(?P<event_id>\d+)\s+(?P<component>\S+)\s+(?P<message>.+)$"
)

# ── HDFS ──────────────────────────────────────────────────────────────────────
_HDFS_RE = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+(?P<level>\w+)\s+"
    r"(?P<component>\S+):\s*(?P<message>.+)$"
)


@dataclass
class ParsedLog:
    raw: str
    source_type: str
    message: str
    normalized_message: str = ""
    log_level: Optional[str] = None
    event_id: Optional[str] = None
    hostname: Optional[str] = None
    process_name: Optional[str] = None
    log_timestamp: Optional[datetime] = None


def _normalize(text: str) -> str:
    """Apply token replacement patterns for clustering-friendly normalization."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    # collapse multiple spaces
    return " ".join(text.split()).lower()


def _try_parse_linux(line: str) -> Optional[ParsedLog]:
    m = _LINUX_RE.match(line.strip())
    if not m:
        return None
    msg = m.group("message")
    return ParsedLog(
        raw=line,
        source_type="linux",
        message=msg,
        normalized_message=_normalize(msg),
        hostname=m.group("hostname"),
        process_name=m.group("process"),
        log_level=None,
    )


def _try_parse_windows(line: str) -> Optional[ParsedLog]:
    m = _WINDOWS_RE.match(line.strip()) or _WINDOWS_RE2.match(line.strip())
    if not m:
        return None
    msg = m.group("message")
    try:
        ts = datetime.strptime(
            f"{m.group('date')} {m.group('time')}", "%Y-%m-%d %H:%M:%S"
        )
    except (ValueError, KeyError):
        ts = None
    return ParsedLog(
        raw=line,
        source_type="windows",
        message=msg,
        normalized_message=_normalize(msg),
        log_level=m.groupdict().get("level"),
        event_id=m.groupdict().get("event_id"),
        log_timestamp=ts,
    )


def _try_parse_hdfs(line: str) -> Optional[ParsedLog]:
    m = _HDFS_RE.match(line.strip())
    if not m:
        return None
    msg = m.group("message")
    return ParsedLog(
        raw=line,
        source_type="hdfs",
        message=msg,
        normalized_message=_normalize(msg),
        log_level=m.group("level"),
        process_name=m.group("component"),
    )


def _fallback_parse(line: str, source_type: str) -> ParsedLog:
    """Last-resort parser: use the whole line as the message."""
    msg = line.strip()
    return ParsedLog(
        raw=line,
        source_type=source_type,
        message=msg,
        normalized_message=_normalize(msg),
    )


class LogPreprocessor:
    """
    Stateless log line parser and normalizer.
    Routes each line to the appropriate format-specific parser.
    """

    _PARSERS = {
        "linux": [_try_parse_linux],
        "windows": [_try_parse_windows],
        "hdfs": [_try_parse_hdfs],
    }

    def parse(self, raw_log: str, source_type: str) -> ParsedLog:
        source_type = source_type.lower().strip()
        parsers = self._PARSERS.get(source_type, [])
        for parser in parsers:
            result = parser(raw_log)
            if result:
                return result
        return _fallback_parse(raw_log, source_type)

    def normalize(self, text: str) -> str:
        return _normalize(text)

    def parse_file(self, path: str, source_type: str) -> list[ParsedLog]:
        logs = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(self.parse(line, source_type))
        return logs
