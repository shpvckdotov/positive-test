"""
Download sample log datasets from LogHub (Zenodo).

Usage:
    python data/download_data.py [--output-dir data/sample_logs]

The script downloads small representative samples from the LogHub datasets:
  - Linux (auth.log / syslog)
  - Windows Event Logs
  - HDFS

If download fails, generates synthetic sample logs for quick testing.
"""
import argparse
import logging
import os
import random
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ── Synthetic log generators (fallback) ──────────────────────────────────────

LINUX_TEMPLATES = [
    "Jun 14 15:16:01 combo sshd(pam_unix)[{pid}]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost={ip}",
    "Jun 14 15:16:02 combo sshd[{pid}]: Failed password for root from {ip} port {port} ssh2",
    "Jun 14 15:16:03 combo sshd[{pid}]: Accepted password for {user} from {ip} port {port} ssh2",
    "Jun 14 15:16:04 combo sudo[{pid}]: {user} : TTY=pts/1 ; PWD=/home/{user} ; USER=root ; COMMAND=/bin/cat /etc/shadow",
    "Jun 14 15:16:05 combo kernel: ACPI: IRQ{num} used by override.",
    "Jun 14 15:16:06 combo cron[{pid}]: ({user}) CMD (/usr/bin/python3 /opt/monitor.py)",
    "Jun 14 15:16:07 combo sshd[{pid}]: pam_unix(sshd:auth): check pass; user unknown",
    "Jun 14 15:16:08 combo sshd[{pid}]: Invalid user admin from {ip}",
    "Jun 14 15:16:09 combo su[{pid}]: Successful su for root by {user}",
    "Jun 14 15:16:10 combo ntpd[{pid}]: time reset +{num}.{num} s",
    "Jun 14 15:16:11 combo postfix/smtpd[{pid}]: connect from unknown[{ip}]",
    "Jun 14 15:16:12 combo postfix/smtp[{pid}]: {hex}: to=<user@example.com>, relay={ip}[{ip}]:{port}, status=sent",
]

WINDOWS_TEMPLATES = [
    "2004-11-04 03:26:32, Information, 4624, Security, An account was successfully logged on. Subject: Security ID: S-1-5-18 Account Name: {host}$ Logon Type: 3",
    "2004-11-04 03:26:33, Warning, 4625, Security, An account failed to log on. Account Name: {user} Failure Reason: Unknown user name or bad password",
    "2004-11-04 03:26:34, Information, 7036, System, The {service} service entered the running state.",
    "2004-11-04 03:26:35, Error, 1000, Application, Faulting application name: {app}.exe, version {num}.{num}.{num}.{num}",
    "2004-11-04 03:26:36, Information, 4688, Security, A new process has been created. Creator Subject: Security ID: {sid} Process Name: C:\\Windows\\System32\\{app}.exe",
    "2004-11-04 03:26:37, Warning, 4776, Security, The computer attempted to validate the credentials for an account. Authentication Package: MICROSOFT_AUTHENTICATION_PACKAGE",
    "2004-11-04 03:26:38, Information, 4672, Security, Special privileges assigned to new logon. Subject: Security ID: S-1-5-18",
    "2004-11-04 03:26:39, Error, 7034, System, The {service} service terminated unexpectedly. It has done this {num} time(s).",
    "2004-11-04 03:26:40, Information, 1102, Security, The audit log was cleared. Subject: Security ID: S-1-5-21-{num}-{num}-{num}-{num}",
    "2004-11-04 03:26:41, Warning, 4698, Security, A scheduled task was created. Task Name: \\Microsoft\\Windows\\{app}",
]

HDFS_TEMPLATES = [
    "081109 204005 1 INFO dfs.DataNode$DataXceiver: Receiving block blk_{num} src: /{ip}:{port} dest: /{ip}:{port}",
    "081109 204005 2 INFO dfs.DataNode$PacketResponder: PacketResponder {num} for block blk_{num} terminating",
    "081109 204005 3 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/task_{num}/part-{num}. blk_{num}",
    "081109 204005 4 INFO dfs.DataNode$DataXceiver: writeBlock blk_{num} received exception java.io.IOException: Connection reset by peer",
    "081109 204005 5 WARN dfs.DataNode: Slow BlockReceiver write data to disk cost:{num}ms (threshold=300ms)",
    "081109 204005 6 ERROR dfs.DataNode: {num}:Exception writing block blk_{num} to mirror {ip}:{port}",
    "081109 204005 7 INFO dfs.DataNode$DataXceiver: Served block blk_{num} to /{ip}",
    "081109 204005 8 INFO namenode.FSNamesystem: BLOCK* NameSystem.delete: blk_{num} is added to invalidSet of {ip}:{port}",
]



class ModelManager:
    def __init__(self):
        current_file_path = Path(__file__).resolve()
        
        project_root = current_file_path.parent.parent
        
        self.models_dir = project_root / "models"
        
        self.vectorizer_path = self.models_dir / "vectorizer.pkl"
        self.clusters_path = self.models_dir / "clusters.json"

    def check_files_exist(self) -> bool:
        if not self.models_dir.exists():
            return False
            
        files_found = []
        if self.vectorizer_path.exists():
            files_found.append("vectorizer.pkl")
        else:
            pass
            
        if self.clusters_path.exists():
            files_found.append("clusters.json")
        else:
            pass
            
        if len(files_found) == 2:
            return True
        else:
            return False
        

def _fill(template: str) -> str:
    rnd = random.Random()
    return (
        template
        .replace("{pid}", str(rnd.randint(1000, 99999)))
        .replace("{port}", str(rnd.randint(1024, 65535)))
        .replace("{num}", str(rnd.randint(1, 9999)))
        .replace("{hex}", "".join(rnd.choices("0123456789ABCDEF", k=10)))
        .replace("{ip}", f"{rnd.randint(1,254)}.{rnd.randint(0,254)}.{rnd.randint(0,254)}.{rnd.randint(1,254)}")
        .replace("{user}", rnd.choice(["alice", "bob", "charlie", "root", "admin", "service"]))
        .replace("{host}", rnd.choice(["WIN-SERVER01", "DC01", "WORKSTATION-42", "LAPTOP-USER"]))
        .replace("{service}", rnd.choice(["Windows Update", "Print Spooler", "Task Scheduler", "WinRM"]))
        .replace("{app}", rnd.choice(["svchost", "explorer", "lsass", "notepad", "chrome"]))
        .replace("{sid}", f"S-1-5-21-{rnd.randint(100000,999999)}")
    )


def generate_synthetic(output_dir: str, n_lines: int = 5000) -> None:
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Generating synthetic Linux logs (%d lines) …", n_lines)
    with open(os.path.join(output_dir, "linux.log"), "w") as f:
        for _ in range(n_lines):
            f.write(_fill(random.choice(LINUX_TEMPLATES)) + "\n")

    logger.info("Generating synthetic Windows logs (%d lines) …", n_lines)
    with open(os.path.join(output_dir, "windows.log"), "w") as f:
        for _ in range(n_lines):
            f.write(_fill(random.choice(WINDOWS_TEMPLATES)) + "\n")

    logger.info("Generating synthetic HDFS logs (%d lines) …", n_lines)
    with open(os.path.join(output_dir, "hdfs.log"), "w") as f:
        for _ in range(n_lines):
            f.write(_fill(random.choice(HDFS_TEMPLATES)) + "\n")

    logger.info("Synthetic data written to %s", output_dir)


def try_download(url: str, dest: str) -> bool:
    try:
        logger.info("Downloading %s …", url)
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        logger.warning("Download failed: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/sample_logs")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Skip download attempt, use synthetic data")
    parser.add_argument("--n-lines", type=int, default=10000)
    args = parser.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    if not args.synthetic_only:
        model_manager = ModelManager()
        if model_manager.check_files_exist():
            return 
        else:
            pass
        logger.info("LogHub datasets require manual download from:")
        logger.info("  https://github.com/logpai/loghub")
        logger.info("Place files as: data/sample_logs/linux.log, windows.log, hdfs.log")
        logger.info("Falling back to synthetic data generation …")

    generate_synthetic(out, n_lines=args.n_lines)


if __name__ == "__main__":
    main()
