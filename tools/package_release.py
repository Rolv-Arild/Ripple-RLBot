"""Package Bob's project output with licenses and portable Linux launch paths."""
import argparse
from pathlib import Path
import re
import shutil
import tomllib
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def package_release(project_dir, output):
    project_dir, output = Path(project_dir).resolve(), Path(output).resolve()
    if output.is_relative_to(project_dir):
        raise ValueError('Place the release archive outside the Bob project output')
    files = {path.relative_to(project_dir).as_posix(): path
             for path in project_dir.rglob('*') if path.is_file()}
    if any(name.endswith('.pt') for name in files):
        raise ValueError('Original .pt weights do not belong in the release package')
    config = tomllib.loads((ROOT / 'bob.toml').read_text(encoding='utf-8'))
    replacements = {}
    linux_binaries = set()
    for source in config['config'][0]['bot_configs']:
        name = Path(source).name
        text = files[name].read_text(encoding='utf-8')
        settings = tomllib.loads(text)['settings']
        linux_command = settings['run_command_linux'].replace('\\', '/')
        # Bob uses host path separators when generating configs on Windows.
        replacements[name] = re.sub(
            r'^run_command_linux\s*=.*$',
            lambda match: f"run_command_linux = '{linux_command}'",
            text, flags=re.MULTILINE).encode('utf-8')
        linux_binaries.add(linux_command)
        for asset in (settings['run_command'], linux_command,
                      settings['logo_file'], settings['loadout_file']):
            if asset.replace('\\', '/') not in files:
                raise ValueError(f'{name} references a missing release file: {asset}')
    for name in ('LICENSE', 'NOTICE'):
        replacements[name] = (ROOT / name).read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(files.keys() | replacements.keys()):
            info = ZipInfo(name)
            info.create_system = 3
            info.external_attr = (0o100755 if name in linux_binaries else 0o100644) << 16
            info.compress_type = ZIP_DEFLATED
            if name in replacements:
                archive.writestr(info, replacements[name])
            else:
                with files[name].open('rb') as source, archive.open(info, 'w') as target:
                    shutil.copyfileobj(source, target)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project_dir', type=Path, help='Bob output directory, e.g. bob_build/Ripple')
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/Ripple.zip')
    args = parser.parse_args()
    print(package_release(args.project_dir, args.output))
