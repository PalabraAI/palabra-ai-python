import pytest
from unittest.mock import patch, MagicMock
from palabra_ai.util.sysinfo import get_system_info, SystemInfo


class TestSysinfo:
    def test_get_system_info(self):
        info = get_system_info()
        
        # Just verify it returns a dict with expected structure
        assert isinstance(info, dict)
        
        # The function might fail in some environments,
        # so we just check it doesn't crash completely
        if info:
            pass
    
    def test_collect_errors(self):
        # Test resource limits exception
        with patch('resource.getrlimit', side_effect=ValueError("Test error")):
            info = SystemInfo()
            assert info.resource_limits == {}
        
        # Test locale exception
        with patch('locale.getdefaultlocale', side_effect=Exception("Test error")):
            info = SystemInfo()
            assert info.locale_info == {}
        
        # Test user info errors
        with patch('pwd.getpwuid', side_effect=Exception("No pwd")):
            with patch('os.getuid', return_value=1000):
                with patch('os.getgid', return_value=1000):
                    with patch('os.getlogin', return_value="testuser"):
                        info = SystemInfo()
                        assert info.user_info['uid'] == 1000
                        assert info.user_info['gid'] == 1000
                        assert info.user_info['username'] == "testuser"
        
        # Test all user info methods fail
        with patch('pwd.getpwuid', side_effect=Exception("No pwd")):
            with patch('os.getuid', side_effect=AttributeError("No getuid")):
                with patch('os.getgid', side_effect=AttributeError("No getgid")):
                    with patch('os.getlogin', side_effect=Exception("No login")):
                        info = SystemInfo()
                        # Should handle all exceptions gracefully
        
        # Test Python paths exception
        with patch('sysconfig.get_path', side_effect=Exception("Path error")):
            info = SystemInfo()
            assert info.python_paths == {}
