"""System information collector for production debugging."""

import datetime
import locale
import multiprocessing
import os
import platform as pyplatform
import pwd
import resource
import socket
import sys
import sysconfig
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class SystemInfo:
    """Collects basic system information for debugging production issues."""

    # Python info
    python_version: str = field(default_factory=lambda: sys.version)
    python_version_info: dict[str, Any] = field(
        default_factory=lambda: {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
            "releaselevel": sys.version_info.releaselevel,
            "serial": sys.version_info.serial,
        }
    )
    python_implementation: str = field(default_factory=pyplatform.python_implementation)
    python_executable: str = field(default_factory=lambda: sys.executable)

    # Platform info
    platform: str = field(default_factory=pyplatform.platform)
    platform_machine: str = field(default_factory=pyplatform.machine)
    platform_processor: str = field(default_factory=pyplatform.processor)
    platform_system: str = field(default_factory=pyplatform.system)
    platform_release: str = field(default_factory=pyplatform.release)
    platform_version: str = field(default_factory=pyplatform.version)
    architecture: dict[str, str] = field(
        default_factory=lambda: {
            "bits": pyplatform.architecture()[0],
            "linkage": pyplatform.architecture()[1],
        }
    )

    # System info
    hostname: str = field(default_factory=socket.gethostname)
    cpu_count: Optional[int] = field(
        default_factory=lambda: multiprocessing.cpu_count()
    )

    # Time info
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    timezone: str = field(
        default_factory=lambda: str(
            datetime.timezone(
                datetime.timedelta(
                    seconds=-datetime.datetime.now()
                    .astimezone()
                    .utcoffset()
                    .total_seconds()
                )
            )
        )
    )

    # Resource limits
    resource_limits: dict[str, dict[str, int]] = field(default_factory=dict)

    # Locale info
    locale_info: dict[str, Optional[str]] = field(default_factory=dict)

    # Current process info
    pid: int = field(default_factory=os.getpid)
    cwd: str = field(default_factory=os.getcwd)

    # User info (safe, no passwords)
    user_info: dict[str, Any] = field(default_factory=dict)

    # Python paths
    python_paths: dict[str, Optional[str]] = field(default_factory=dict)

    def __post_init__(self):
        """Collect additional system information after initialization."""
        self._collect_resource_limits()
        self._collect_locale_info()
        self._collect_user_info()
        self._collect_python_paths()

    def _collect_resource_limits(self) -> None:
        """Collect resource limits information."""
        try:
            limits = {
                "RLIMIT_CPU": resource.RLIMIT_CPU,
                "RLIMIT_FSIZE": resource.RLIMIT_FSIZE,
                "RLIMIT_DATA": resource.RLIMIT_DATA,
                "RLIMIT_STACK": resource.RLIMIT_STACK,
                "RLIMIT_CORE": resource.RLIMIT_CORE,
                "RLIMIT_RSS": resource.RLIMIT_RSS,
                "RLIMIT_NPROC": resource.RLIMIT_NPROC,
                "RLIMIT_NOFILE": resource.RLIMIT_NOFILE,
                "RLIMIT_MEMLOCK": resource.RLIMIT_MEMLOCK,
                "RLIMIT_AS": resource.RLIMIT_AS,
            }

            for name, res_id in limits.items():
                try:
                    soft, hard = resource.getrlimit(res_id)
                    self.resource_limits[name] = {
                        "soft": soft if soft != resource.RLIM_INFINITY else "unlimited",
                        "hard": hard if hard != resource.RLIM_INFINITY else "unlimited",
                    }
                except (ValueError, OSError):
                    pass
        except Exception:
            pass

    def _collect_locale_info(self) -> None:
        """Collect locale information."""
        try:
            self.locale_info = {
                "default": locale.getdefaultlocale()[0],
                "preferred_encoding": locale.getpreferredencoding(False),
            }
        except Exception:
            pass

    def _collect_user_info(self) -> None:
        """Collect current user information (safe data only)."""
        try:
            uid = os.getuid()
            pwd_info = pwd.getpwuid(uid)
            self.user_info = {
                "uid": uid,
                "gid": os.getgid(),
                "username": pwd_info.pw_name,
                "home": pwd_info.pw_dir,
                "shell": pwd_info.pw_shell,
            }
        except Exception:
            # Fallback for systems without pwd module
            try:
                self.user_info = {
                    "uid": os.getuid() if hasattr(os, "getuid") else None,
                    "gid": os.getgid() if hasattr(os, "getgid") else None,
                    "username": os.getlogin() if hasattr(os, "getlogin") else None,
                }
            except Exception:
                pass

    def _collect_python_paths(self) -> None:
        """Collect Python-related paths."""
        try:
            self.python_paths = {
                "prefix": sys.prefix,
                "base_prefix": sys.base_prefix,
                "exec_prefix": sys.exec_prefix,
                "base_exec_prefix": sys.base_exec_prefix,
                "stdlib": sysconfig.get_path("stdlib"),
                "platstdlib": sysconfig.get_path("platstdlib"),
                "purelib": sysconfig.get_path("purelib"),
                "platlib": sysconfig.get_path("platlib"),
                "include": sysconfig.get_path("include"),
                "scripts": sysconfig.get_path("scripts"),
                "data": sysconfig.get_path("data"),
            }
        except Exception:
            pass


def get_system_info() -> dict[str, Any]:
    """Get system information as a JSON-serializable dictionary."""
    return asdict(SystemInfo())
