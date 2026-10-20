import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from package_release import package_release


class ReleaseTests(unittest.TestCase):
    def test_windows_bob_output_becomes_a_portable_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'bob'
            output.mkdir()
            for version in ('v1.0', 'v1.1'):
                (output / f'ripple-{version}.bot.toml').write_text(
                    "[settings]\nrun_command = 'windows\\bot.exe'\n"
                    "run_command_linux = 'linux\\bot'\n"
                    f"logo_file = 'ripple-{version}.png'\n"
                    f"loadout_file = 'ripple-{version}.loadout.toml'\n", encoding='utf-8')
                for asset in (f'ripple-{version}.png', f'ripple-{version}.loadout.toml'):
                    (output / asset).touch()
            for name in ('windows/bot.exe', 'linux/bot'):
                path = output / name
                path.parent.mkdir()
                path.write_bytes(b'fake binary')
            archive_path = package_release(output, Path(directory) / 'Ripple.zip')
            with ZipFile(archive_path) as archive:
                for version in ('v1.0', 'v1.1'):
                    text = archive.read(f'ripple-{version}.bot.toml').decode('utf-8')
                    settings = tomllib.loads(text)['settings']
                    self.assertEqual(settings['run_command'], 'windows\\bot.exe')
                    self.assertEqual(settings['run_command_linux'], 'linux/bot')
                self.assertEqual(archive.getinfo('linux/bot').external_attr >> 16, 0o100755)
                for name in ('LICENSE', 'NOTICE'):
                    self.assertEqual(archive.read(name), (ROOT / name).read_bytes())
            # Generated Bob output remains available for comparison/debugging.
            self.assertIn('linux\\bot', (output / 'ripple-v1.1.bot.toml').read_text())
