import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import {runInNewContext} from 'node:vm';
import test from 'node:test';
import {parse} from 'yaml';
import {allowed, evaluate} from './workflow-expressions.mjs';

const require = createRequire(import.meta.url);
const engine = fileURLToPath(new URL('../../.clouseau-ci/', import.meta.url));
const config = JSON.parse(readFileSync(new URL('../clouseau.json', import.meta.url)));
const workflow = parse(readFileSync(new URL('../workflows/ci.yml', import.meta.url), 'utf8'));
const revision = 'a'.repeat(40);
const assess = workflow.jobs['assess-intent'];
const guard = assess.steps[0];

function context(result = 'success') {
  return {github: {event_name: 'push', ref: 'refs/heads/policy-install/pr-23', run_id: 7, event: {deleted: false}},
    inputs: {},
    needs: {'assess-intent': {result, outputs: {assessment: JSON.stringify({record_digest: 'b'.repeat(64)})}}}};
}

async function runGuard(mutate = () => {}) {
  const state = {context: {actor: 'gaulatti', repo: {owner: 'gaulatti', repo: 'mistify'}, sha: revision,
      ref: 'refs/heads/policy-install/pr-23'}, reads: 0, outputs: {},
    pr: {state: 'open', draft: false, base: {ref: 'main'}, user: {login: 'gaulatti'},
      head: {sha: revision, repo: {full_name: 'gaulatti/mistify'}}},
    comments: [{user: {login: 'gaulatti', type: 'User'}, body: `/approve-intent ${revision}`}]};
  mutate(state);
  const listComments = () => {};
  const github = {rest: {pulls: {get: async ({pull_number}) => {
    state.reads++; assert.equal(pull_number, 23); return {data: state.pr};
  }}, issues: {listComments}}, paginate: async (method, {issue_number}) => {
    state.reads++; assert.equal(method, listComments); assert.equal(issue_number, 23); return state.comments;
  }};
  await runInNewContext(`(async () => {${guard.with.script}\n})()`,
    {github, context: state.context, core: {setOutput: (name, value) => {state.outputs[name] = value;}}});
  return state;
}

test('the actual installer guard accepts only an owner-confirmed, frozen same-repository PR', async () => {
  const accepted = await runGuard();
  assert.equal(accepted.reads, 2);
  assert.equal(accepted.outputs.number, '23');
  for (const mutate of [
    s => {s.context.actor = 'contributor';}, s => {s.context.sha = 'invalid';},
    s => {s.context.ref = 'refs/heads/main';}, s => {s.context.ref = 'refs/heads/policy-install/pr-23;merge';},
    s => {s.context.ref = 'refs/heads/policy-install/pr-' + '9'.repeat(30);},
    s => {s.pr.state = 'closed';}, s => {s.pr.draft = true;}, s => {s.pr.base.ref = 'other';},
    s => {s.pr.user.login = 'contributor';}, s => {s.pr.head.sha = 'c'.repeat(40);},
    s => {s.pr.head.repo.full_name = 'external/mistify';}, s => {s.pr.head.repo = null;},
    s => {s.comments = [];}, s => {s.comments[0].user.type = 'Bot';},
    s => {s.comments[0].user.login = 'contributor';}, s => {s.comments[0].body = '/approve-intent stale';},
  ]) await assert.rejects(runGuard(mutate), /requires|Invalid installation/);
  assert.equal(guard.uses, 'actions/github-script@v7');
  assert.ok(Object.values(assess.permissions).every(value => value === 'read'));
  const provider = assess.steps.findIndex(step => step.with?.operation === 'assess');
  assert.ok(provider > 0);
  assert.ok(!JSON.stringify(assess.steps.slice(0, provider)).includes('secrets.'));
  assert.equal(assess.steps[provider].with['pull-request'], '${{ steps.install.outputs.number }}');
});

test('installation reuses independent verification, fails closed and cannot trigger repair or a duplicate reassessment', () => {
  // GitHub excludes workflow_dispatch job checks from required PR checks.
  // Installation must run on the eligible push event at the exact PR head.
  assert.ok(workflow.on.push.branches.includes('policy-install/**'));
  assert.equal(workflow.on.workflow_dispatch.inputs.install_reviewer, undefined);
  const c = context();
  for (const name of ['unit', 'build', 'request-reassessment', 'apply-repair', 'check-author']) {
    assert.equal(allowed(workflow.jobs[name].if, c, {success: false}), false, name);
  }
  assert.equal(evaluate(workflow.jobs.verify.name, c).coerceString(), 'metadata-only');
  assert.equal(allowed(assess.if, c), true);
  assert.equal(evaluate(workflow.jobs['approve-and-merge'].name, c).coerceString(), 'approve-and-merge');
  assert.equal(allowed(workflow.jobs['finalize-merge'].if, c), true);
  const gate = workflow.jobs['approve-and-merge'];
  for (const result of ['failure', 'cancelled', 'skipped']) {
    assert.equal(allowed(gate.steps[0].if, context(result)), true);
    assert.equal(gate.steps[0].run, 'exit 1');
    assert.equal(allowed(gate.steps.at(-1).if, context(result), {success: false}), false);
  }
  assert.equal(gate.steps.at(-1).with['native-gate'], 'true');
  assert.equal(gate.steps.at(-1).with['pull-request'], '${{ needs.assess-intent.outputs.number }}');
  assert.ok(!JSON.stringify(gate).includes('provider-key'));
  assert.equal(allowed(guard.if, c), true);
  const manual = {...c, github: {...c.github, event_name: 'workflow_dispatch'},
    inputs: {pull_request: '23', check_author: false}};
  assert.equal(allowed(assess.if, manual), false);
  assert.equal(evaluate(gate.name, manual).coerceString(), 'metadata-only-approval');
  c.github.event_name = 'pull_request';
  assert.equal(allowed(assess.if, c), false);
});

test('the actual installation wait checks exact-PR verification and rejects stale or failed execution evidence', async () => {
  const step = assess.steps.find(item => item.name === 'Wait for tests, exact coverage, and build');
  for (const scenario of ['ready', 'stale-head', 'failed-tests']) {
    const context = {sha: revision, repo: {owner: 'gaulatti', repo: 'mistify'}, payload: {}};
    const pr = {number: 23, state: 'open', draft: false,
      head: {sha: scenario === 'stale-head' ? 'c'.repeat(40) : revision, ref: 'ci/docs', repo: {id: 7}}};
    const github = {rest: {pulls: {get: async ({pull_number}) => {assert.equal(pull_number, 23); return {data: pr};}}, actions: {
      listWorkflowRuns: async () => ({data: {workflow_runs: [{id: 42, head_sha: revision, pull_requests: [{number: 23}],
        status: 'completed', conclusion: scenario === 'failed-tests' ? 'failure' : 'success', html_url: 'https://github.com/gaulatti/mistify/actions/runs/42'}]}}),
      listJobsForWorkflowRun: async () => ({data: [{name: 'verify', conclusion: 'success',
        steps: config.verification.steps.map(name => ({name, conclusion: 'success'}))}]}),
    }}, paginate: async (method, args) => (await method(args)).data};
    const messages = [];
    const invoke = runInNewContext(`(async () => {${step.with.script}\n})()`, {github, context,
      process: {env: {REQUESTED_PR: '23'}}, core: {info: value => messages.push(value)},
      require: name => name === 'node:path' ? require(name) : name.endsWith('/src/config.cjs') ?
        {loadConfig: () => config} : name.endsWith('/src/central-version.cjs') ? require(`${engine}/src/central-version.cjs`) : require(`${engine}/src/policy.cjs`)});
    if (scenario === 'ready') {await invoke; assert.match(messages[0], /Clouseau acceptance can now start/);}
    else await assert.rejects(invoke, /changed|verification concluded/);
  }
});
