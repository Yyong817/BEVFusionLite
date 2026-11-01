"""Check Python syntax, local imports, registries and CLI help without CUDA."""
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'pcdet.ops.iou3d_nms.iou3d_nms_cuda',
              'pcdet.ops.roiaware_pool3d.roiaware_pool3d_cuda',
              'pcdet.ops.bev_pool.bev_pool_ext'}


def exists(module):
    path = ROOT.joinpath(*module.split('.'))
    return path.with_suffix('.py').is_file() or (path / '__init__.py').is_file() or module in EXTENSIONS


def exports(module):
    path = ROOT.joinpath(*module.split('.'))
    path = path / '__init__.py' if path.is_dir() else path.with_suffix('.py')
    if not path.exists():
        return set()
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    names.update(node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
    return names


def main():
    errors = []
    files = list(ROOT.rglob('*.py'))
    for path in files:
        if '__pycache__' in path.parts:
            continue
        source = path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), 'exec')
        relative = path.relative_to(ROOT)
        package = list(relative.parent.parts)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                prefix = package[:len(package) - node.level + 1]
                module = '.'.join(prefix + (node.module.split('.') if node.module else []))
            else:
                module = node.module or ''
            if not module.startswith(('pcdet', 'tools')):
                continue
            if not exists(module):
                errors.append(f'{relative}:{node.lineno}: missing module {module}')
                continue
            available = exports(module)
            for alias in node.names:
                if alias.name != '*' and alias.name not in available and not exists(f'{module}.{alias.name}'):
                    errors.append(f'{relative}:{node.lineno}: missing {module}.{alias.name}')
    for entry in ['train.py', 'test.py', 'prepare_data.py']:
        result = subprocess.run([sys.executable, str(ROOT / entry), '--help'],
                                cwd=ROOT.parent, capture_output=True, text=True)
        if result.returncode:
            errors.append(f'{entry} --help: {result.stderr}')
    report = dict(python_files=len(files), local_imports='passed' if not errors else 'failed',
                  cli_help='checked', errors=errors)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
