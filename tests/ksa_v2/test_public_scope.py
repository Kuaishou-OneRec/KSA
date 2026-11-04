"""The public V2 workflow exposes KSA, C-KSA and summary-based S-KSA only."""
import ast
import tomllib
from pathlib import Path

import pytest

from muse.config.model_config import Qwen3SummaryAttentionConfig

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('mode', ['attention_reuse', 'token_indexer'])
def test_unsupported_routing_modes_are_rejected(mode):
    with pytest.raises(ValueError):
        Qwen3SummaryAttentionConfig(sksa_routing_mode=mode)


def test_inference_exports_only_defined_public_symbols():
    for path in (REPO / 'examples/ksa_v2/hf_template').glob('*.py'):
        tree = ast.parse(path.read_text())
        defined = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                defined.update(item.asname or item.name for item in node.names)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets):
                assert set(ast.literal_eval(node.value)) <= defined, path


def test_training_kernel_exports_have_implementations():
    import ksa_attn
    assert all(hasattr(ksa_attn, name) for name in ksa_attn.__all__)
    assert not any('token_sparse' in name or 'attention_reuse' in name for name in ksa_attn.__all__)


def test_one_shared_kernel_package_and_dependency_version():
    kernel_root = REPO / 'summary_attention_kernel'
    assert not list(kernel_root.glob('*.whl'))
    package_root = kernel_root / 'ksa_v2/v0.1'
    project = tomllib.loads((package_root / 'pyproject.toml').read_text())
    assert project['project']['name'] == 'ksa_attn'
    assert project['tool']['setuptools']['packages'] == ['ksa_attn']
    for directory in (package_root, kernel_root / 'ksa_v2/inference'):
        assert [p.name for p in directory.glob('*_attn')] == ['ksa_attn']
    for path in (REPO / 'dockerfile/requirements.txt', REPO / 'examples/ksa_v2/requirements.txt'):
        requirements = path.read_text()
        for pin in ('nvidia-cutlass-dsl==4.3.5', 'apache-tvm-ffi==0.1.13.post3', 'torch-c-dlpack-ext==0.1.4'):
            assert pin in requirements


def test_public_method_names_and_config_entrypoints():
    obsolete = 'kb' + 'sa'
    ignored = {'.git', 'output', '.tmp', '.cache', '.pytest_cache', 'assets', 'data', 'licenses', 'deps', 'wheels'}
    for path in REPO.rglob('*'):
        if not path.is_file() or ignored.intersection(path.relative_to(REPO).parts):
            continue
        if path.suffix not in {'.py', '.md', '.json', '.sh'}:
            continue
        assert obsolete not in path.name.lower(), path
        assert obsolete not in path.read_text().lower(), path
    config_dir = REPO / 'examples/ksa_v2/configs'
    assert (config_dir / 'cksa_g128_sksa_block4_1b9_128k.json').is_file()
    assert not (REPO / 'examples/ksa_v2/tools/train.py').exists()
    assert not (REPO / 'examples/ksa_v2/tools/prepare_data.py').exists()


def test_internal_names_are_sksa_except_legacy_export_inputs():
    old = 'sc' + 'sa'
    compatibility = {
        REPO / 'examples/ksa_v2/tools/export_hf.py',
        REPO / 'examples/ksa_v1/pretrain/convert/convert.py',
    }
    for directory in ('muse', 'recipes', 'examples', 'summary_attention_kernel/ksa_v2'):
        for path in (REPO / directory).rglob('*'):
            if not path.is_file() or path in compatibility or 'output' in path.relative_to(REPO).parts:
                continue
            if path.suffix not in {'.py', '.json', '.sh', '.toml', '.txt', '.md'}:
                continue
            assert old not in path.name.lower(), path
            assert old not in path.read_text().lower(), path
