"""CPU checks of metadata policy; automatic default still needs real JIT evidence."""
import unittest
from runtime_common import validate_native_route_metadata

class NativeQueryPolicyTest(unittest.TestCase):
    def test_ordinary_default_remains_valid(self):
        validate_native_route_metadata(dict(ir_override=None,bitcodes=None))
        validate_native_route_metadata(dict(ir_override=None,bitcodes=[],enable_cvp_query_pipeline=False,
                                           disable_tightly_coupled_buffer_reuse=False))

    def test_query_requires_exact_effective_true(self):
        for value in (None,False,0,1,'true'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_native_route_metadata(dict(ir_override=None,bitcodes=[],enable_cvp_query_pipeline=True,
                                                   disable_tightly_coupled_buffer_reuse=value))

    def test_protected_query_allows_normal_empty_bitcode_defaults(self):
        for value in (None,[],()):
            validate_native_route_metadata(dict(ir_override=None,bitcodes=value,enable_cvp_query_pipeline=True,
                                               disable_tightly_coupled_buffer_reuse=True))

    def test_protected_policy_does_not_admit_injected_ir_or_bitcodes(self):
        for change in (dict(ir_override='file.mlir'),dict(bitcodes=['private.bc'])):
            metadata=dict(ir_override=None,bitcodes=[],enable_cvp_query_pipeline=True,
                          disable_tightly_coupled_buffer_reuse=True)
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_native_route_metadata(dict(metadata,**change))

if __name__=='__main__':unittest.main(verbosity=2)
