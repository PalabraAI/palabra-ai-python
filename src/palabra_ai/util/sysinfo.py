"""System information collector for production debugging."""

import datetime
import gc
import locale
import multiprocessing
import os
import platform as pyplatform
import pwd
import resource
import shutil
import socket
import subprocess
import sys
import sysconfig
from dataclasses import asdict, dataclass, field
from typing import Any

try:
    from importlib import metadata
except ImportError:
    # Fallback for older Python versions, though we target 3.11+
    import importlib_metadata as metadata  # type: ignore

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def _run_command(cmd: list[str], timeout: int = 5) -> str | None:
    """Run a command and return its output, or None if it fails."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


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
    cpu_count: int | None = field(default_factory=lambda: multiprocessing.cpu_count())

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
    locale_info: dict[str, str | None] = field(default_factory=dict)

    # Current process info
    pid: int = field(default_factory=os.getpid)
    cwd: str = field(default_factory=os.getcwd)

    # User info (safe, no passwords)
    user_info: dict[str, Any] = field(default_factory=dict)

    # Python paths
    python_paths: dict[str, str | None] = field(default_factory=dict)

    # Installed packages
    installed_packages: dict[str, str] = field(default_factory=dict)

    # Disk space information
    disk_space: dict[str, Any] = field(default_factory=dict)

    # Memory information
    memory_info: dict[str, Any] = field(default_factory=dict)

    # Python process memory
    process_memory: dict[str, Any] = field(default_factory=dict)

    # Garbage collector info
    gc_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Collect additional system information after initialization."""
        self._collect_resource_limits()
        self._collect_locale_info()
        self._collect_user_info()
        self._collect_python_paths()
        self._collect_installed_packages()
        self._collect_disk_space()
        self._collect_memory_info()
        self._collect_process_memory()
        self._collect_gc_info()

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
            self.locale_info = locale.getlocale()
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

    def _collect_installed_packages(self) -> None:
        """Collect installed packages and their versions using importlib.metadata."""
        try:
            # Get all installed distributions
            for dist in metadata.distributions():
                try:
                    # Get package name and version
                    name = dist.metadata.get("Name", "unknown")
                    version = dist.metadata.get("Version", "unknown")
                    if name and version:
                        self.installed_packages[name] = version
                except Exception:
                    # Skip packages that can't be read
                    pass
        except Exception:
            # If metadata.distributions() fails, try alternative approach
            try:
                # Alternative: use pkg_resources if available (though deprecated)
                import pkg_resources

                for dist in pkg_resources.working_set:
                    self.installed_packages[dist.key] = dist.version
            except Exception:
                # If all methods fail, leave empty
                pass

    def _collect_disk_space(self) -> None:
        """Collect disk space information for all mounted partitions."""
        try:
            if HAS_PSUTIL:
                # Use psutil for comprehensive disk info
                partitions = psutil.disk_partitions(all=False)
                for partition in partitions:
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        self.disk_space[partition.mountpoint] = {
                            "device": partition.device,
                            "fstype": partition.fstype,
                            "total": usage.total,
                            "used": usage.used,
                            "free": usage.free,
                            "percent": usage.percent,
                        }
                    except PermissionError:
                        # Some partitions may not be accessible
                        continue
            else:
                # Fallback to shutil for current directory's disk
                stat = shutil.disk_usage(os.getcwd())
                self.disk_space["current_directory"] = {
                    "total": stat.total,
                    "used": stat.used,
                    "free": stat.free,
                    "percent": (stat.used / stat.total * 100) if stat.total > 0 else 0,
                }

            # Add df -h output
            df_output = _run_command(["df", "-h"])
            if df_output:
                self.disk_space["df_output"] = df_output

        except Exception:
            pass

    def _collect_memory_info(self) -> None:
        """Collect system memory information."""
        try:
            if HAS_PSUTIL:
                # Virtual memory (RAM)
                vm = psutil.virtual_memory()
                self.memory_info["virtual"] = {
                    "total": vm.total,
                    "available": vm.available,
                    "percent": vm.percent,
                    "used": vm.used,
                    "free": vm.free,
                }

                # Swap memory
                swap = psutil.swap_memory()
                self.memory_info["swap"] = {
                    "total": swap.total,
                    "used": swap.used,
                    "free": swap.free,
                    "percent": swap.percent,
                }
            else:
                # Fallback to resource module for basic info
                import resource

                rusage = resource.getrusage(resource.RUSAGE_SELF)
                self.memory_info["rusage"] = {
                    "maxrss": rusage.ru_maxrss,  # Maximum resident set size
                    "ixrss": rusage.ru_ixrss,  # Integral shared memory size
                    "idrss": rusage.ru_idrss,  # Integral unshared data size
                    "isrss": rusage.ru_isrss,  # Integral unshared stack size
                }

            # Add system command outputs
            # Try free command (Linux)
            free_output = _run_command(["free", "-h"])
            if free_output:
                self.memory_info["free_output"] = free_output

            # Try vm_stat command (macOS)
            vm_stat_output = _run_command(["vm_stat"])
            if vm_stat_output:
                self.memory_info["vm_stat_output"] = vm_stat_output

            # Try top command for memory summary
            if sys.platform == "darwin":
                # macOS version
                top_output = _run_command(["top", "-l", "1", "-n", "0"])
            else:
                # Linux version
                top_output = _run_command(["top", "-b", "-n", "1"])

            if top_output:
                # Extract just the memory lines
                lines = top_output.split("\n")
                memory_lines = [
                    line
                    for line in lines
                    if any(
                        keyword in line.lower()
                        for keyword in ["mem", "swap", "physicalmem"]
                    )
                ]
                if memory_lines:
                    self.memory_info["top_memory_summary"] = "\n".join(
                        memory_lines[:5]
                    )  # Limit to first 5 relevant lines

        except Exception:
            pass

    def _collect_process_memory(self) -> None:
        """Collect current Python process memory usage."""
        try:
            if HAS_PSUTIL:
                process = psutil.Process(os.getpid())

                # Memory info
                mem_info = process.memory_info()
                self.process_memory["memory"] = {
                    "rss": mem_info.rss,  # Resident Set Size
                    "vms": mem_info.vms,  # Virtual Memory Size
                }

                # Memory percent
                self.process_memory["memory_percent"] = process.memory_percent()

                # Full memory info if available
                if hasattr(process, "memory_full_info"):
                    full_info = process.memory_full_info()
                    self.process_memory["memory_full"] = {
                        "uss": full_info.uss,  # Unique Set Size
                        "pss": full_info.pss,  # Proportional Set Size
                        "swap": full_info.swap,
                    }
            else:
                # Fallback to resource module
                import resource

                rusage = resource.getrusage(resource.RUSAGE_SELF)
                # Convert to bytes (on Darwin/macOS it's in KB)
                multiplier = 1024 if sys.platform == "darwin" else 1
                self.process_memory["maxrss"] = rusage.ru_maxrss * multiplier

            # Add ps output for current process
            pid = os.getpid()

            # ps with memory info
            ps_output = _run_command(
                ["ps", "-p", str(pid), "-o", "pid,vsz,rss,pmem,comm"]
            )
            if ps_output:
                self.process_memory["ps_output"] = ps_output

            # More detailed ps output (platform-specific)
            if sys.platform == "darwin":
                # macOS extended format
                ps_extended = _run_command(["ps", "-p", str(pid), "-v"])
            else:
                # Linux extended format
                ps_extended = _run_command(["ps", "-p", str(pid), "-u"])

            if ps_extended:
                self.process_memory["ps_extended"] = ps_extended

        except Exception:
            pass

    def _collect_gc_info(self) -> None:
        """Collect garbage collector statistics."""
        try:
            # GC stats
            self.gc_info["stats"] = gc.get_stats()

            # GC counts for each generation
            self.gc_info["count"] = gc.get_count()

            # GC thresholds
            self.gc_info["threshold"] = gc.get_threshold()

            # Is GC enabled?
            self.gc_info["enabled"] = gc.isenabled()

            # Number of objects
            self.gc_info["objects"] = len(gc.get_objects())

            # Referrers count for common types
            type_counts = {}
            for obj in gc.get_objects():
                type_name = type(obj).__name__
                type_counts[type_name] = type_counts.get(type_name, 0) + 1

            # Get top 10 most common types
            sorted_types = sorted(
                type_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]
            self.gc_info["top_object_types"] = dict(sorted_types)

        except Exception:
            pass


def get_system_info() -> dict[str, Any]:
    """Get system information as a JSON-serializable dictionary."""
    return asdict(SystemInfo())
