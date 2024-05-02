import unittest
from unittest.mock import patch

from logstat.cli import cli, console_main


class TestMainFunction(unittest.TestCase):
    @patch('sys.argv', ['logstat.cli.py', '--help'])
    def test_main_runs_without_errors(self):
        # Test that main runs without raising any exceptions
        with self.assertRaises(SystemExit) as cm:
            cli()
        self.assertEqual(cm.exception.code, 0)

    def test_main_runs_help(self):
        # Test that main runs without raising any exceptions
        with self.assertRaises(SystemExit) as cm:
            cli(['--help'])
        self.assertEqual(cm.exception.code, 0)

    @patch('sys.argv', ['logstat.cli.py', '--help'])
    def test_main_runs_console_main_help(self):
        # Test that main runs without raising any exceptions
        with self.assertRaises(SystemExit) as cm:
            console_main()
        self.assertEqual(cm.exception.code, 0)


if __name__ == '__main__':
    unittest.main()
