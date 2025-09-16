import os
import sys
import subprocess
import gc
import platform
import pwd
from unittest.mock import MagicMock, patch, call
import pytest
from palabra_ai.util.sysinfo import SystemInfo, get_system_info, _run_command, HAS_PSUTIL

class TestRunCommand:
    """Test _run_command function"""

    def test_run_command_success(self):
        """Test successful command execution"""
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "  output  \n"
            mock_run.return_value = mock_result

            result = _run_command(["echo", "test"])

            assert result == "output"
            mock_run.assert_called_once_with(
                ["echo", "test"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )

    def test_run_command_failure(self):
        """Test command with non-zero return code"""
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = "error output"
            mock_run.return_value = mock_result

            result = _run_command(["false"])

            assert result is None

    def test_run_command_timeout(self):
        """Test command timeout"""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 5)

            result = _run_command(["sleep", "10"])

            assert result is None

    def test_run_command_file_not_found(self):
        """Test command not found"""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()

            result = _run_command(["nonexistent_command"])

            assert result is None

    def test_run_command_os_error(self):
        """Test OS error during command execution"""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = OSError("Permission denied")

            result = _run_command(["restricted_command"])

            assert result is None

    def test_run_command_custom_timeout(self):
        """Test command with custom timeout"""
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "output"
            mock_run.return_value = mock_result

            result = _run_command(["echo", "test"], timeout=10)

            assert result == "output"
            mock_run.assert_called_once_with(
                ["echo", "test"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False
            )

class TestSystemInfo:
    """Test SystemInfo class"""

    def test_init_basic_fields(self):
        """Test initialization of basic fields"""
        with patch.multiple(
            'palabra_ai.util.sysinfo.SystemInfo',
            _collect_resource_limits=MagicMock(),
            _collect_locale_info=MagicMock(),
            _collect_user_info=MagicMock(),
            _collect_python_paths=MagicMock(),
            _collect_installed_packages=MagicMock(),
            _collect_disk_space=MagicMock(),
            _collect_memory_info=MagicMock(),
            _collect_process_memory=MagicMock(),
            _collect_gc_info=MagicMock()
        ):
            info = SystemInfo()

            # Check Python info
            assert info.python_version == sys.version
            assert info.python_version_info["major"] == sys.version_info.major
            assert info.python_version_info["minor"] == sys.version_info.minor
            assert info.python_executable == sys.executable

            # Check platform info
            assert info.platform == platform.platform()
            assert info.platform_machine == platform.machine()
            assert info.platform_system == platform.system()

            # Check system info
            assert info.hostname == platform.node()
            assert info.pid == os.getpid()
            assert info.cwd == os.getcwd()

    def test_collect_resource_limits(self):
        """Test _collect_resource_limits method"""
        info = SystemInfo.__new__(SystemInfo)
        info.resource_limits = {}

        with patch('resource.getrlimit') as mock_getrlimit:
            mock_getrlimit.side_effect = [
                (1024, 2048),  # CPU
                (1024, -1),    # FSIZE with RLIM_INFINITY
                ValueError("Not supported"),  # DATA - error case
            ] + [(0, 0)] * 10  # Rest of the limits

            with patch('resource.RLIM_INFINITY', -1):
                info._collect_resource_limits()

            assert info.resource_limits["RLIMIT_CPU"] == {"soft": 1024, "hard": 2048}
            assert info.resource_limits["RLIMIT_FSIZE"] == {"soft": 1024, "hard": "unlimited"}
            assert "RLIMIT_DATA" not in info.resource_limits  # Failed to get

    def test_collect_resource_limits_exception(self):
        """Test _collect_resource_limits with general exception"""
        info = SystemInfo.__new__(SystemInfo)
        info.resource_limits = {}

        with patch('resource.getrlimit') as mock_getrlimit:
            mock_getrlimit.side_effect = Exception("General error")

            info._collect_resource_limits()

            assert info.resource_limits == {}

    def test_collect_locale_info(self):
        """Test _collect_locale_info method"""
        info = SystemInfo.__new__(SystemInfo)
        info.locale_info = {}

        with patch('locale.getlocale') as mock_getlocale:
            mock_getlocale.return_value = ('en_US', 'UTF-8')

            info._collect_locale_info()

            assert info.locale_info == ('en_US', 'UTF-8')

    def test_collect_locale_info_exception(self):
        """Test _collect_locale_info with exception"""
        info = SystemInfo.__new__(SystemInfo)
        info.locale_info = {}

        with patch('locale.getlocale') as mock_getlocale:
            mock_getlocale.side_effect = Exception("Locale error")

            info._collect_locale_info()

            assert info.locale_info == {}

    def test_collect_user_info(self):
        """Test _collect_user_info method"""
        info = SystemInfo.__new__(SystemInfo)
        info.user_info = {}

        mock_pwd_info = MagicMock()
        mock_pwd_info.pw_name = "testuser"
        mock_pwd_info.pw_dir = "/home/testuser"
        mock_pwd_info.pw_shell = "/bin/bash"

        with patch('os.getuid', return_value=1000):
            with patch('os.getgid', return_value=1000):
                with patch('pwd.getpwuid', return_value=mock_pwd_info):
                    info._collect_user_info()

        assert info.user_info == {
            "uid": 1000,
            "gid": 1000,
            "username": "testuser",
            "home": "/home/testuser",
            "shell": "/bin/bash"
        }

    def test_collect_user_info_fallback(self):
        """Test _collect_user_info fallback when pwd fails"""
        info = SystemInfo.__new__(SystemInfo)
        info.user_info = {}

        with patch('pwd.getpwuid', side_effect=Exception("No pwd")):
            with patch('os.getuid', return_value=1000):
                with patch('os.getgid', return_value=1000):
                    with patch('os.getlogin', return_value="testuser"):
                        info._collect_user_info()

        assert info.user_info == {
            "uid": 1000,
            "gid": 1000,
            "username": "testuser"
        }

    def test_collect_user_info_no_methods(self):
        """Test _collect_user_info when os methods don't exist"""
        info = SystemInfo.__new__(SystemInfo)
        info.user_info = {}

        with patch('pwd.getpwuid', side_effect=Exception("No pwd")):
            with patch('os.getuid', side_effect=AttributeError()):
                info._collect_user_info()

        assert info.user_info == {}

    def test_collect_python_paths(self):
        """Test _collect_python_paths method"""
        info = SystemInfo.__new__(SystemInfo)
        info.python_paths = {}

        with patch('sysconfig.get_path') as mock_get_path:
            mock_get_path.side_effect = lambda name: f"/path/to/{name}"

            info._collect_python_paths()

        assert info.python_paths["prefix"] == sys.prefix
        assert info.python_paths["stdlib"] == "/path/to/stdlib"
        assert info.python_paths["purelib"] == "/path/to/purelib"
        assert len(info.python_paths) == 11

    def test_collect_python_paths_exception(self):
        """Test _collect_python_paths with exception"""
        info = SystemInfo.__new__(SystemInfo)
        info.python_paths = {}

        with patch('sysconfig.get_path', side_effect=Exception("Path error")):
            info._collect_python_paths()

        assert info.python_paths == {}

    def test_collect_installed_packages(self):
        """Test _collect_installed_packages method"""
        info = SystemInfo.__new__(SystemInfo)
        info.installed_packages = {}

        # Mock distribution objects
        dist1 = MagicMock()
        dist1.metadata.get.side_effect = lambda key, default=None: {
            "Name": "package1",
            "Version": "1.0.0"
        }.get(key, default)

        dist2 = MagicMock()
        dist2.metadata.get.side_effect = lambda key, default=None: {
            "Name": "package2",
            "Version": "2.0.0"
        }.get(key, default)

        with patch('palabra_ai.util.sysinfo.metadata.distributions', return_value=[dist1, dist2]):
            info._collect_installed_packages()

        assert info.installed_packages == {
            "package1": "1.0.0",
            "package2": "2.0.0"
        }

    def test_collect_installed_packages_fallback(self):
        """Test _collect_installed_packages with pkg_resources fallback"""
        info = SystemInfo.__new__(SystemInfo)
        info.installed_packages = {}

        with patch('palabra_ai.util.sysinfo.metadata.distributions', side_effect=Exception("No metadata")):
            # Mock pkg_resources
            mock_dist1 = MagicMock()
            mock_dist1.key = "package1"
            mock_dist1.version = "1.0.0"

            mock_dist2 = MagicMock()
            mock_dist2.key = "package2"
            mock_dist2.version = "2.0.0"

            with patch('sys.modules', {'pkg_resources': MagicMock(working_set=[mock_dist1, mock_dist2])}):
                with patch('builtins.__import__', return_value=MagicMock(working_set=[mock_dist1, mock_dist2])):
                    info._collect_installed_packages()

        assert info.installed_packages == {
            "package1": "1.0.0",
            "package2": "2.0.0"
        }

    def test_collect_installed_packages_all_fail(self):
        """Test _collect_installed_packages when all methods fail"""
        info = SystemInfo.__new__(SystemInfo)
        info.installed_packages = {}

        with patch('palabra_ai.util.sysinfo.metadata.distributions', side_effect=Exception("No metadata")):
            with patch('builtins.__import__', side_effect=ImportError("No pkg_resources")):
                info._collect_installed_packages()

        assert info.installed_packages == {}
        """Test _collect_disk_space without psutil"""
        info = SystemInfo.__new__(SystemInfo)
        info.disk_space = {}

        mock_stat = MagicMock()
        mock_stat.total = 1000000
        mock_stat.used = 600000
        mock_stat.free = 400000

        with patch('palabra_ai.util.sysinfo.HAS_PSUTIL', False):
            with patch('shutil.disk_usage', return_value=mock_stat):
                with patch('palabra_ai.util.sysinfo._run_command', return_value=None):
                    info._collect_disk_space()

        assert "current_directory" in info.disk_space
        assert info.disk_space["current_directory"]["total"] == 1000000
        assert info.disk_space["current_directory"]["percent"] == 60.0

    def test_collect_disk_space_exception(self):
        """Test _collect_disk_space with exception"""
        info = SystemInfo.__new__(SystemInfo)
        info.disk_space = {}

        with patch('shutil.disk_usage', side_effect=Exception("Disk error")):
            info._collect_disk_space()

        assert info.disk_space == {}
        """Test _collect_memory_info without psutil"""
        info = SystemInfo.__new__(SystemInfo)
        info.memory_info = {}

        mock_rusage = MagicMock()
        mock_rusage.ru_maxrss = 1024
        mock_rusage.ru_ixrss = 512
        mock_rusage.ru_idrss = 256
        mock_rusage.ru_isrss = 128

        with patch('palabra_ai.util.sysinfo.HAS_PSUTIL', False):
            with patch('resource.getrusage', return_value=mock_rusage):
                with patch('palabra_ai.util.sysinfo._run_command', return_value=None):
                    info._collect_memory_info()

        assert "rusage" in info.memory_info
        assert info.memory_info["rusage"]["maxrss"] == 1024

    def test_collect_memory_info_with_commands(self):
        """Test _collect_memory_info with system commands"""
        info = SystemInfo.__new__(SystemInfo)
        info.memory_info = {}

        def mock_run_command(cmd):
            if cmd[0] == "free":
                return "free output"
            elif cmd[0] == "vm_stat":
                return "vm_stat output"
            elif cmd[0] == "top":
                return "PhysicalMem: 16GB\nMemory: 8GB used\nSwap: 2GB"
            return None

        with patch('palabra_ai.util.sysinfo.HAS_PSUTIL', False):
            with patch('resource.getrusage', return_value=MagicMock()):
                with patch('palabra_ai.util.sysinfo._run_command', side_effect=mock_run_command):
                    info._collect_memory_info()

        assert info.memory_info.get("free_output") == "free output"
        assert info.memory_info.get("vm_stat_output") == "vm_stat output"
        assert "PhysicalMem" in info.memory_info.get("top_memory_summary", "")

    def test_collect_memory_info_exception(self):
        """Test _collect_memory_info with exception"""
        info = SystemInfo.__new__(SystemInfo)
        info.memory_info = {}

        with patch('resource.getrusage', side_effect=Exception("Memory error")):
            info._collect_memory_info()

        assert info.memory_info == {}
        """Test _collect_process_memory without psutil"""
        info = SystemInfo.__new__(SystemInfo)
        info.process_memory = {}

        mock_rusage = MagicMock()
        mock_rusage.ru_maxrss = 1024

        with patch('palabra_ai.util.sysinfo.HAS_PSUTIL', False):
            with patch('resource.getrusage', return_value=mock_rusage):
                with patch('sys.platform', 'darwin'):  # macOS multiplies by 1024
                    with patch('palabra_ai.util.sysinfo._run_command', return_value=None):
                        info._collect_process_memory()

        assert info.process_memory["maxrss"] == 1024 * 1024  # KB to bytes on macOS

    def test_collect_process_memory_with_ps(self):
        """Test _collect_process_memory with ps commands"""
        info = SystemInfo.__new__(SystemInfo)
        info.process_memory = {}

        def mock_run_command(cmd):
            if "-o" in cmd:
                return "PID VSZ RSS %MEM COMMAND\n1234 100000 50000 5.0 python"
            elif "-v" in cmd or "-u" in cmd:
                return "Extended ps output"
            return None

        with patch('palabra_ai.util.sysinfo.HAS_PSUTIL', False):
            with patch('resource.getrusage', return_value=MagicMock(ru_maxrss=1024)):
                with patch('palabra_ai.util.sysinfo._run_command', side_effect=mock_run_command):
                    with patch('os.getpid', return_value=1234):
                        info._collect_process_memory()

        assert "ps_output" in info.process_memory
        assert "1234" in info.process_memory["ps_output"]
        assert info.process_memory["ps_extended"] == "Extended ps output"

    def test_collect_process_memory_exception(self):
        """Test _collect_process_memory with exception"""
        info = SystemInfo.__new__(SystemInfo)
        info.process_memory = {}

        with patch('resource.getrusage', side_effect=Exception("Process error")):
            info._collect_process_memory()

        assert info.process_memory == {}

    def test_collect_gc_info(self):
        """Test _collect_gc_info method"""
        info = SystemInfo.__new__(SystemInfo)
        info.gc_info = {}

        # Mock gc functions
        mock_stats = [{"collections": 10, "collected": 100}]
        mock_count = (5, 2, 1)
        mock_threshold = (700, 10, 10)

        # Mock objects for type counting
        mock_objects = [
            "string1", "string2", "string3",
            123, 456,
            [], [], [], [],
            {}, {}, {},
        ]

        with patch('gc.get_stats', return_value=mock_stats):
            with patch('gc.get_count', return_value=mock_count):
                with patch('gc.get_threshold', return_value=mock_threshold):
                    with patch('gc.isenabled', return_value=True):
                        with patch('gc.get_objects', return_value=mock_objects):
                            info._collect_gc_info()

        assert info.gc_info["stats"] == mock_stats
        assert info.gc_info["count"] == mock_count
        assert info.gc_info["threshold"] == mock_threshold
        assert info.gc_info["enabled"] is True
        assert info.gc_info["objects"] == len(mock_objects)
        assert "top_object_types" in info.gc_info
        assert info.gc_info["top_object_types"]["list"] == 4  # 4 lists in mock_objects
        assert info.gc_info["top_object_types"]["dict"] == 3  # 3 dicts in mock_objects

    def test_collect_gc_info_exception(self):
        """Test _collect_gc_info with exception"""
        info = SystemInfo.__new__(SystemInfo)
        info.gc_info = {}

        with patch('gc.get_stats', side_effect=Exception("GC error")):
            info._collect_gc_info()

        assert info.gc_info == {}

    def test_post_init_calls_all_collectors(self):
        """Test that __post_init__ calls all collector methods"""
        with patch.object(SystemInfo, '_collect_resource_limits') as mock_res:
            with patch.object(SystemInfo, '_collect_locale_info') as mock_loc:
                with patch.object(SystemInfo, '_collect_user_info') as mock_user:
                    with patch.object(SystemInfo, '_collect_python_paths') as mock_paths:
                        with patch.object(SystemInfo, '_collect_installed_packages') as mock_pkg:
                            with patch.object(SystemInfo, '_collect_disk_space') as mock_disk:
                                with patch.object(SystemInfo, '_collect_memory_info') as mock_mem:
                                    with patch.object(SystemInfo, '_collect_process_memory') as mock_proc:
                                        with patch.object(SystemInfo, '_collect_gc_info') as mock_gc:
                                            info = SystemInfo()

                                            mock_res.assert_called_once()
                                            mock_loc.assert_called_once()
                                            mock_user.assert_called_once()
                                            mock_paths.assert_called_once()
                                            mock_pkg.assert_called_once()
                                            mock_disk.assert_called_once()
                                            mock_mem.assert_called_once()
                                            mock_proc.assert_called_once()
                                            mock_gc.assert_called_once()

class TestGetSystemInfo:
    """Test get_system_info function"""

    def test_get_system_info_returns_dict(self):
        """Test that get_system_info returns a dictionary"""
        with patch.multiple(
            'palabra_ai.util.sysinfo.SystemInfo',
            _collect_resource_limits=MagicMock(),
            _collect_locale_info=MagicMock(),
            _collect_user_info=MagicMock(),
            _collect_python_paths=MagicMock(),
            _collect_installed_packages=MagicMock(),
            _collect_disk_space=MagicMock(),
            _collect_memory_info=MagicMock(),
            _collect_process_memory=MagicMock(),
            _collect_gc_info=MagicMock()
        ):
            result = get_system_info()

            assert isinstance(result, dict)
            assert "python_version" in result
            assert "platform" in result
            assert "hostname" in result
            assert "pid" in result

    def test_get_system_info_serializable(self):
        """Test that get_system_info returns JSON-serializable data"""
        import json

        with patch.multiple(
            'palabra_ai.util.sysinfo.SystemInfo',
            _collect_resource_limits=MagicMock(),
            _collect_locale_info=MagicMock(),
            _collect_user_info=MagicMock(),
            _collect_python_paths=MagicMock(),
            _collect_installed_packages=MagicMock(),
            _collect_disk_space=MagicMock(),
            _collect_memory_info=MagicMock(),
            _collect_process_memory=MagicMock(),
            _collect_gc_info=MagicMock()
        ):
            result = get_system_info()

            # Should not raise an exception
            json_str = json.dumps(result)
            assert isinstance(json_str, str)
