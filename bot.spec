# One executable serves both agent IDs. ONNX Runtime is bundled; torch
# and the original training exports are only needed for development/parity.
from pathlib import Path
import os
import sys

root = Path(SPECPATH)
binaries = []
if sys.platform == 'win32':
    runtime = Path(os.environ['SystemRoot']) / 'System32/MSVCP140.dll'
    if runtime.is_file():
        # Do not pick an obsolete C++ runtime from Java or another PATH entry.
        # RocketSim's filesystem/mutex code requires the installed VC runtime.
        binaries.append((str(runtime), '.'))
a = Analysis(
    [str(root / 'src/bot.py')], pathex=[str(root / 'src')], binaries=binaries,
    datas=[(str(path), str(path.parent.relative_to(root))) for path in (root / 'bots').rglob('*.onnx')] + [
           (str(root / 'src/collision_meshes'), 'collision_meshes'),
           (str(root / 'LICENSE'), '.'), (str(root / 'NOTICE'), '.')],
    hiddenimports=['timeit', 'pickletools', 'uuid', 'unittest.mock', 'ctypes.wintypes'],
    hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=['torch'],
    noarchive=False, optimize=0,
)
if sys.platform == 'win32':
    # Windows supplies UCRT and API-set shims. An old JDK may expose private
    # copies on PATH; bundling those causes native crashes on modern Windows.
    a.binaries = [entry for entry in a.binaries
                  if entry[0].lower() != 'ucrtbase.dll'
                  and not entry[0].lower().startswith('api-ms-win-')]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='bot',
          debug=False, bootloader_ignore_signals=False, strip=False,
          upx=False, console=True)
