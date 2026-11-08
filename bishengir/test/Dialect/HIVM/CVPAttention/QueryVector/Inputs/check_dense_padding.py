"""Probe derived dense-P padding through scope, VF and copy views."""
import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--opt', required=True)
parser.add_argument('--fixtures', type=Path, required=True)
parser.add_argument('--artifacts', type=Path)
args = parser.parse_args()
if args.artifacts:
    args.artifacts.mkdir(parents=True, exist_ok=False)
counter = 0


def run(source, options):
    global counter
    command = [args.opt, *options, '-verify-each']
    result = subprocess.run(command, input=source, text=True, capture_output=True)
    if args.artifacts:
        stem = f'{counter:03d}'
        for suffix, value in [('input.mlir', source), ('output.mlir', result.stdout), ('stderr.txt', result.stderr)]:
            (args.artifacts / (stem + '.' + suffix)).write_text(value)
        (args.artifacts / (stem + '.json')).write_text(json.dumps({
            'argv': command, 'exit': result.returncode, 'expected_error': None
        }, indent=2) + '\n')
    counter += 1
    assert result.returncode == 0, result.stderr
    return result.stdout


manifest = json.loads((args.fixtures / 'MANIFEST.json').read_text())
reference = next(row for row in manifest if row['arm'] == 'key-major')
path = args.fixtures / reference['file']
assert hashlib.sha256(path.read_bytes()).hexdigest() == reference['sha256']
module = next(line for line in path.read_text().splitlines() if line.startswith('module attributes '))
module = module.replace(', hivm.module_core_type = #hivm.module_core_type<MIX>', '')

for keys, dynamic in itertools.product((128, 256), (False, True)):
    half = keys // 2
    row = half * 16
    stride = (half + 1) * 16
    ub = '#hivm.address_space<ub>'
    gm = '#hivm.address_space<gm>'
    full = f'memref<{"?" if dynamic else "8"}x{half}x16xf16, {ub}>'
    piece = f'memref<8x1x16xf16, strided<[{row}, 16, 1], offset: ?>, {ub}>'
    low = f'memref<4x{half}x16xf16, strided<[{row}, 16, 1]>, {ub}>'
    high = f'memref<4x{half}x16xf16, strided<[{row}, 16, 1], offset: {4 * row}>, {ub}>'
    flat_low = f'memref<4x{row}xf16, strided<[{row}, 1]>, {ub}>'
    flat_high = f'memref<4x{row}xf16, strided<[{row}, 1], offset: {4 * row}>, {ub}>'
    output = f'memref<4x{row}xf16, {gm}>'
    source = module + f'''
  func.func private @write_p(%p: {full}) attributes {{hivm.vector_function, no_inline, hivm.func_core_type = #hivm.func_core_type<AIV>}} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %end = arith.constant {half} : index
    %value = arith.constant dense<0.000000e+00> : vector<8x1x16xf16>
    scf.for %k = %c0 to %end step %c1 {{
      %view = memref.subview %p[0, %k, 0] [8, 1, 16] [1, 1, 1] : {full} to {piece}
      vector.transfer_write %value, %view[%c0, %c0, %c0] {{in_bounds = [true, true, true]}} : vector<8x1x16xf16>, {piece}
    }}
    return
  }}
  func.func @entry(%low_out: {output}, %high_out: {output}{", %extent: index" if dynamic else ""}) attributes {{hivm.func_core_type = #hivm.func_core_type<AIV>}} {{
    %p = scope.scope : () -> {full} {{
      %allocation = memref.alloc({"%extent" if dynamic else ""}) {{alignment = 64 : i64}} : {full}
      annotation.mark %allocation {{hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: {2 * (half + 1)}>}} : {full}
      annotation.mark %allocation {{buffer_size_in_byte = {8 * half * 16 * 2} : i64}} : {full}
      scope.return %allocation : {full}
    }} {{no_inline}}
    func.call @write_p(%p) {{hivm.vector_function, no_inline}} : ({full}) -> ()
    %low = memref.subview %p[0, 0, 0] [4, {half}, 16] [1, 1, 1] : {full} to {low}
    %high = memref.subview %p[4, 0, 0] [4, {half}, 16] [1, 1, 1] : {full} to {high}
    %low_flat = memref.collapse_shape %low [[0], [1, 2]] : {low} into {flat_low}
    %high_flat = memref.collapse_shape %high [[0], [1, 2]] : {high} into {flat_high}
    hivm.hir.store ins(%low_flat : {flat_low}) outs(%low_out : {output})
    hivm.hir.store ins(%high_flat : {flat_high}) outs(%high_out : {output})
    return
  }}
}}
'''
    before = run(source, [])
    # Inspect the pre-canonicalization size mark on dynamic storage. Static
    # marks are deliberately removed by FoldUselessBufferSizeMarkOp; their
    # actual allocation shape is the byte-size authority.
    aligned = run(before, ['--hivm-enable-stride-align'])
    if dynamic:
        assert f'buffer_size_in_byte={8 * (half + 1) * 16 * 2}:i64' in ''.join(aligned.split())
    after = run(aligned, ['-canonicalize'])
    compact = ''.join(after.split())
    allocations = re.findall(r'memref.alloc\([^\n]*? : memref<([0-9x?]+)xf16', after)
    assert len(allocations) == 1, allocations
    dimensions = allocations[0].split('x')
    assert dimensions.count('?') == int(dynamic)
    # The dynamic case's explicit allocation-size bound is eight planes.
    shape = [8 if dim == '?' else int(dim) for dim in dimensions]
    assert math.prod(shape) == 8 * (half + 1) * 16 and half + 1 in shape, shape
    if dynamic:
        assert f'buffer_size_in_byte={8 * (half + 1) * 16 * 2}:i64' in compact
    else:
        assert 'buffer_size_in_byte' not in after
    vf_headers = [line for line in after.splitlines() if 'func.func ' in line and 'hivm.vector_function' in line]
    assert any(f'strided<[{stride},16,1]' in ''.join(line.split()) for line in vf_headers)
    assert f'strided<[{stride},1],offset:{4 * stride}>' in compact
    assert compact.count('hivm.hir.store') == 2
    assert 'unrealized_conversion_cast' not in after
    assert 'memref.copy' not in after and 'hivm.hir.copy' not in after
    assert run(after, ['--hivm-enable-stride-align', '-canonicalize']) == after
print('Dense-P padding probe passed BN128/256 with static/dynamic allocation inputs: derived H+1 storage, byte accounting, '
      'scope/VF/view propagation and two copy spans; no packing/native/NPU qualification')
