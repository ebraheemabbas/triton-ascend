"""Exercise the real frontend option functions without loading an NPU runtime."""
from pathlib import Path
import ast
import os
import re
import subprocess
import types
import unittest
from unittest.mock import patch

REPO_SOURCE=Path(__file__).resolve().parents[2]/'backend/compiler.py'
DEFAULT=(REPO_SOURCE if REPO_SOURCE.is_file() else
    Path(__file__).resolve().parents[1]/'stages7-9-source-v1/src/triton-ascend/third_party/ascend/backend/compiler.py')
SOURCE=Path(os.environ.get('CVP_FRONTEND_SOURCE',str(DEFAULT)))
tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
NAMES={'_cvp_query_buffer_policy','_validate_attention_vector_recipe_options','_validate_attention_layout_options',
       '_native_attention_layout_options','_native_vector_control_options'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in NAMES]
nodes += [n for n in tree.body if isinstance(n,ast.Assign) and
          any(isinstance(t,ast.Name) and t.id=='_ATTENTION_LAYOUT_CONTROLS' for t in n.targets)]
assert len(nodes)==6
namespace={'re':re,'subprocess':subprocess}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),namespace)


def recipe(**changes):
    values=dict(enable_cvp_query_pipeline=True,cvp_query_pipeline_mode='skew',cvp_query_key_unroll=1,
        enable_cvp_attention_key_major=True,allow_cvp_attention_sum_reassociation=True,
        enable_dynamic_cv_pipeline=False,enable_mixed_cv=True,enable_preload=True,
        enable_vf_operand_substitution=True,enable_fused_multiply_add=False,
        compile_on_910_95=True,parallel_mode='simd',mix_mode='mix')
    values.update(changes)
    return values


class FrontendWithoutStage6Test(unittest.TestCase):
    def validate(self,values):
        return namespace['_validate_attention_layout_options'](values)

    def test_query_buffer_policy_defaults_to_protected(self):
        resolve = namespace['_cvp_query_buffer_policy']
        self.assertIs(resolve(recipe()), True)
        self.assertIs(resolve(recipe(disable_tightly_coupled_buffer_reuse=None)), True)
        self.assertIs(resolve(recipe(disable_tightly_coupled_buffer_reuse=True)), True)

    def test_query_buffer_policy_rejects_explicit_unsafe_false(self):
        with self.assertRaisesRegex(ValueError, 'disable_tightly_coupled_buffer_reuse=True'):
            self.validate(recipe(disable_tightly_coupled_buffer_reuse=False))

    def test_non_query_buffer_policy_preserves_existing_choices(self):
        resolve = namespace['_cvp_query_buffer_policy']
        for value, expected in ((None, False), (False, False), (True, True)):
            with self.subTest(value=value):
                self.assertIs(resolve(dict(disable_tightly_coupled_buffer_reuse=value)), expected)

    def test_query_buffer_policy_rejects_truthy_non_boolean(self):
        for value in (0, 1, 'true', 'false', [], {}):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.validate(recipe(disable_tightly_coupled_buffer_reuse=value))

    def test_query_unroll_policy_is_also_protected_without_qualifying_unroll(self):
        values = recipe(cvp_query_pipeline_mode='unroll', enable_preload=False)
        self.assertIs(namespace['_cvp_query_buffer_policy'](values), True)
        with self.assertRaises(ValueError):
            self.validate(dict(values, disable_tightly_coupled_buffer_reuse=False))

    def test_skew_query_needs_no_stage6_flags(self):
        self.assertTrue(self.validate(recipe()))

    def test_neutral_cached_and_auto_cube_combinations_are_available(self):
        for mode in ('neutral','cached','auto'):
            with self.subTest(mode=mode):
                self.assertTrue(self.validate(recipe(cvp_invariant_cache_mode=mode,
                    enable_cvp_cube_banked_ready=True,enable_cvp_cube_overlap=True,
                    enable_cvp_cube_derived_drain=True)))

    def test_query_unroll_u1_u2_u4_remain_available(self):
        for lanes in (1,2,4):
            with self.subTest(lanes=lanes):
                self.assertTrue(self.validate(recipe(cvp_query_pipeline_mode='unroll',
                    cvp_query_key_unroll=lanes,enable_preload=False)))

    def test_cross_core_synchronization_still_required(self):
        with self.assertRaisesRegex(ValueError,'cross-core'):
            self.validate(recipe(disable_auto_inject_block_sync=True))

    def test_layout_still_required(self):
        with self.assertRaisesRegex(ValueError,'key-major'):
            self.validate(recipe(enable_cvp_attention_key_major=False))

    def test_dc_vp_cannot_replace_the_cvpipeline_route(self):
        with self.assertRaisesRegex(ValueError,'DCVP'):
            self.validate(recipe(enable_dynamic_cv_pipeline=True))

    def test_fma_does_not_grant_sum_reassociation(self):
        with self.assertRaisesRegex(ValueError,'reassociation'):
            self.validate(recipe(enable_fused_multiply_add=True,allow_cvp_attention_sum_reassociation=False))

    def test_math_and_vf_choices_stay_explicit(self):
        for key in ('enable_vf_operand_substitution','enable_fused_multiply_add'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.validate(recipe(**{key:None}))

    def test_cube_overlap_and_drain_keep_readiness_prerequisite(self):
        for key in ('enable_cvp_cube_overlap','enable_cvp_cube_derived_drain'):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'readiness'):
                self.validate(recipe(cvp_invariant_cache_mode='cached',**{key:True}))

    def test_banked_readiness_keeps_template_pair_prerequisite(self):
        with self.assertRaisesRegex(ValueError,'template pair'):
            self.validate(recipe(enable_cvp_cube_banked_ready=True))

    def test_unqualified_unroll_cache_is_not_silently_accepted(self):
        for mode in ('cached', 'auto'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError,'Unroll invariant-cache'):
                self.validate(recipe(cvp_query_pipeline_mode='unroll',enable_preload=False,cvp_invariant_cache_mode=mode))

    def test_bad_modes_lanes_and_boolean_types_reject(self):
        for change in ({'cvp_query_pipeline_mode':'bad'},{'cvp_query_key_unroll':3},
                       {'cvp_query_key_unroll':True},{'enable_cvp_query_pipeline':1}):
            with self.subTest(change=change),self.assertRaises((ValueError,TypeError)):
                self.validate(recipe(**change))

    def test_default_off_sends_no_new_flags_or_help_probe(self):
        with patch.object(subprocess,'run') as run:
            self.assertEqual(namespace['_native_attention_layout_options']({},'/compiler',{}),[])
            self.assertEqual(namespace['_native_vector_control_options']({},'/compiler',{}),[])
            run.assert_not_called()

    def test_normal_frontend_forwards_all_retained_options(self):
        values=recipe(cvp_invariant_cache_mode='cached',enable_cvp_cube_banked_ready=True,
            enable_cvp_cube_overlap=True,enable_cvp_cube_derived_drain=True)
        flags={'--'+k.replace('_','-'):v for k,v in values.items() if k in namespace['_ATTENTION_LAYOUT_CONTROLS']}
        help_text='\n'.join('  '+k+'=<value>' for k in flags)
        with patch.object(subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout=help_text)):
            actual=namespace['_native_attention_layout_options'](values,'/compiler',{})
        self.assertEqual(set(actual),{f'{k}={v}' for k,v in flags.items()})
        self.assertFalse(any('async-ownership' in x or 'physical-memory-footprints' in x for x in actual))

    def test_missing_compiler_capability_rejects(self):
        with patch.object(subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='')):
            with self.assertRaisesRegex(RuntimeError,'does not support'):
                namespace['_native_attention_layout_options'](recipe(),'/compiler',{})

    def test_final_native_route_is_rechecked(self):
        with self.assertRaisesRegex(ValueError,'final Ascend'):
            namespace['_native_attention_layout_options'](recipe(mix_mode='aiv'),'/compiler',{})

    def test_stage6_options_and_callbacks_are_absent(self):
        fields=set()
        for node in tree.body:
            if isinstance(node,ast.ClassDef) and node.name=='NPUOptions':
                fields={n.target.id for n in node.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)}
        self.assertIn('enable_cvp_query_pipeline',fields)
        self.assertIn('cvp_invariant_cache_mode',fields)
        self.assertNotIn('enable_cvp_async_ownership',fields)
        self.assertNotIn('enable_physical_memory_footprints',fields)
        self.assertNotIn('_read_cvp_async_launch_callback',{n.name for n in tree.body if isinstance(n,ast.FunctionDef)})


    def test_final_output_forward_default_is_none(self):
        options = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NPUOptions')
        field = next(n for n in options.body if isinstance(n, ast.AnnAssign)
                     and isinstance(n.target, ast.Name)
                     and n.target.id == 'enable_cvp_query_final_output_forward')
        self.assertIsNone(ast.literal_eval(field.value))

    def test_final_output_forward_requires_query_route(self):
        with self.assertRaisesRegex(ValueError, 'final output forwarding requires an explicit query pipeline'):
            self.validate(recipe(enable_cvp_query_pipeline=False,
                                 enable_cvp_query_final_output_forward=True))

    def test_final_output_forward_true_and_false_are_independent(self):
        for value in (False, True):
            with self.subTest(value=value):
                self.assertTrue(self.validate(recipe(enable_cvp_query_final_output_forward=value)))

    def test_final_output_forward_rejects_invalid_types(self):
        for value in (0, 1, 'true', [], {}):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.validate(recipe(enable_cvp_query_final_output_forward=value))

    def test_final_output_and_input_forwarding_are_cumulative(self):
        self.assertTrue(self.validate(recipe(enable_cvp_query_state_copy_forward=True,
                                             enable_cvp_query_final_output_forward=True)))

    def test_final_output_forward_requires_selected_compiler_capability(self):
        values = recipe(enable_cvp_query_final_output_forward=True)
        help_text = '\n'.join('--'+k.replace('_','-')+'=<bool>'
                              for k in values if k in namespace['_ATTENTION_LAYOUT_CONTROLS']
                              and k != 'enable_cvp_query_final_output_forward')
        with patch.object(subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout=help_text)):
            with self.assertRaisesRegex(RuntimeError, 'enable-cvp-query-final-output-forward'):
                namespace['_native_attention_layout_options'](values, '/compiler', {})

    def test_final_output_forward_uses_separate_native_flag(self):
        values = recipe(enable_cvp_query_final_output_forward=True)
        flags = {'--'+k.replace('_','-'):v for k,v in values.items()
                 if k in namespace['_ATTENTION_LAYOUT_CONTROLS']}
        help_text = '\n'.join(k+'=<bool>' for k in flags)
        with patch.object(subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout=help_text)):
            actual = namespace['_native_attention_layout_options'](values, '/compiler', {})
        self.assertEqual(set(actual), {f'{k}={v}' for k,v in flags.items()})
        self.assertIn('--enable-cvp-query-final-output-forward=True', actual)
        self.assertFalse(any(arg.startswith('--enable-cvp-query-state-copy-forward') for arg in actual))



    def test_dense_two_chain_complete_recipe(self):
        self.assertTrue(self.validate(recipe(enable_cvp_attention_dense_p=True,
            cvp_attention_producer_fusion='max-p-state',
            enable_cvp_attention_score_rematerialization=True,
            cvp_attention_reduction_chains=2)))

    def test_dense_two_chain_incomplete_recipe_rejects(self):
        values=recipe(enable_cvp_attention_dense_p=True,
            cvp_attention_producer_fusion='max-p-state',
            enable_cvp_attention_score_rematerialization=True,
            cvp_attention_reduction_chains=2)
        for key,value in (('enable_cvp_attention_dense_p',False),
                          ('cvp_attention_producer_fusion','p-state'),
                          ('enable_cvp_attention_score_rematerialization',False),
                          ('allow_cvp_attention_sum_reassociation',False)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.validate(dict(values,**{key:value}))

    def test_dense_two_chain_unroll_rejects(self):
        with self.assertRaisesRegex(ValueError,'Unroll'):
            self.validate(recipe(enable_cvp_attention_dense_p=True,
                cvp_attention_producer_fusion='max-p-state',
                enable_cvp_attention_score_rematerialization=True,
                cvp_attention_reduction_chains=2,
                cvp_query_pipeline_mode='unroll',enable_preload=False))

    def test_dense_two_chain_uses_existing_native_option(self):
        values=recipe(enable_cvp_attention_dense_p=True,
            cvp_attention_producer_fusion='max-p-state',
            enable_cvp_attention_score_rematerialization=True,
            cvp_attention_reduction_chains=2)
        flags={'--'+key.replace('_','-'):value for key,value in values.items()
               if key in namespace['_ATTENTION_LAYOUT_CONTROLS']}
        help_text='\n'.join(key+'=<value>' for key in flags)
        with patch.object(subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout=help_text)):
            actual=namespace['_native_attention_layout_options'](values,'/compiler',{})
        self.assertIn('--cvp-attention-reduction-chains=2',actual)
        self.assertEqual(set(actual),{f'{key}={value}' for key,value in flags.items()})


    def test_output_copy_forward_complete_recipe(self):
        self.assertTrue(self.validate(recipe(enable_cvp_query_state_copy_forward=True,
            enable_cvp_query_final_output_forward=True, enable_cvp_query_output_copy_forward=True)))

    def test_output_copy_forward_requires_all_preconditions(self):
        base=recipe(enable_cvp_query_state_copy_forward=True,
            enable_cvp_query_final_output_forward=True, enable_cvp_query_output_copy_forward=True)
        for name in ('enable_cvp_query_pipeline', 'enable_cvp_query_state_copy_forward',
                     'enable_cvp_query_final_output_forward', 'enable_vf_operand_substitution'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate(dict(base, **{name:False}))

    def test_output_copy_forward_requires_skew(self):
        with self.assertRaisesRegex(ValueError, 'Unroll'):
            self.validate(recipe(enable_cvp_query_state_copy_forward=True,
                enable_cvp_query_final_output_forward=True, enable_cvp_query_output_copy_forward=True,
                cvp_query_pipeline_mode='unroll'))


    def test_scalar_input_forward_complete_recipe(self):
        self.assertTrue(self.validate(recipe(enable_cvp_query_scalar_input_forward=True)))

    def test_scalar_input_forward_requires_query_and_vf(self):
        for key in ('enable_cvp_query_pipeline', 'enable_vf_operand_substitution'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(recipe(enable_cvp_query_scalar_input_forward=True, **{key:False}))

    def test_scalar_input_forward_type_and_mode(self):
        for value in (0, 1, 'true', [], {}):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.validate(recipe(enable_cvp_query_scalar_input_forward=value))
        with self.assertRaisesRegex(ValueError, 'Unroll'):
            self.validate(recipe(enable_cvp_query_scalar_input_forward=True,
                cvp_query_pipeline_mode='unroll'))

if __name__=='__main__':
    unittest.main(verbosity=2)
