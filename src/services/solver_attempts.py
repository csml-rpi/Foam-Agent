"""Shared input restoration and failed-input stopping for solver runners."""
from logger import print

from .case_inputs import case_input_fingerprint, canonical_input_protections
from .case_workspace import prepare_case_workspace
from .execution_logs import declared_startup_inputs, check_execution_outputs
from .review_history import ReviewHistory
from .stage_plan import stage_outcome
from .run_hpc import hpc_execution_outputs


def prepare_solver_attempt(state, runner):
    protected = canonical_input_protections(
        state.get('case_input_source_dir'), declared_startup_inputs(state['solver_operations']))
    outputs = hpc_execution_outputs('solver') if runner == 'hpc' else ['Allrun.out', 'Allrun.err']
    errors = check_execution_outputs(outputs, protected)
    if errors:
        # A rejected output declaration is not an established canonical input.
        current = {**state, 'case_input_source_dir': state.get('case_input_source_dir'),
                   'last_run_input_fingerprint': None}
        outcome = {'executed': False, 'error_logs': errors}
        return {**current, **record_solver_attempt(current, {
            **stage_outcome(current, 'solver', outcome),
            'error_logs': errors, 'solver_execution': outcome}, runner)}, True
    workspace = prepare_case_workspace(
        state['case_dir'], state.get('case_import_report_dir'), state.get('case_input_source_dir'),
        protected_inputs=declared_startup_inputs(state['solver_operations']))
    current = {**state, 'case_input_source_dir': str(workspace.source_dir) if workspace else None}
    fingerprint = case_input_fingerprint(
        current['case_dir'], protected_inputs=canonical_input_protections(
            current['case_input_source_dir'], declared_startup_inputs(current['solver_operations'])))
    current['last_run_input_fingerprint'] = fingerprint
    history = ReviewHistory(current['case_dir'], current.get('case_import_report_dir'))
    failed = current.get('failed_run_inputs') or {}
    if fingerprint and fingerprint in failed:
        message = 'Repair restored inputs that already failed in this workflow; execution was not repeated.'
        history.append('run_skipped', stage='solver', reason='revisited_failed_inputs',
                       input_fingerprint=fingerprint, runtime_loop=current.get('loop_count', 0))
        print(f'<runner>{message}</runner>')
        return {**current, 'workflow_status': 'failed', 'termination_reason': 'repair_made_no_progress',
                'workflow_message': message, 'error_logs': failed[fingerprint]}, True
    history.append('run_started', stage='solver', runner=runner, input_fingerprint=fingerprint,
                   runtime_loop=current.get('loop_count', 0))
    return current, False


def record_solver_attempt(state, result, runner):
    fingerprint = state['last_run_input_fingerprint']
    failed = dict(state.get('failed_run_inputs') or {})
    errors = result.get('error_logs') or []
    if fingerprint and errors and (result.get('solver_execution') or {}).get('executed'):
        failed[fingerprint] = errors
    ReviewHistory(state['case_dir'], state.get('case_import_report_dir')).append(
        'run_result', stage='solver', runner=runner, input_fingerprint=fingerprint,
        runtime_loop=state.get('loop_count', 0), job_id=result.get('job_id'), error_logs=errors)
    return {**result, 'case_input_source_dir': state['case_input_source_dir'],
            'failed_run_inputs': failed, 'last_run_input_fingerprint': fingerprint}
