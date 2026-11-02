import assert from 'node:assert/strict';
import {readFileSync, existsSync, symlinkSync, unlinkSync, mkdtempSync, mkdirSync, writeFileSync, rmSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import {parse} from 'yaml';
import {evaluate, allowed} from './workflow-expressions.mjs';

const workflow = parse(readFileSync(new URL('../workflows/ci.yml', import.meta.url), 'utf8'));
const require = createRequire(import.meta.url);
const engine = fileURLToPath(new URL('../../.clouseau-ci/', import.meta.url));
const config = require(`${engine}/src/config.cjs`).loadConfig(fileURLToPath(new URL('../clouseau.json', import.meta.url)));
const head = 'a'.repeat(40);
const baseExpression = '${{ github.event.pull_request.base.sha || github.event.before }}';

function context(event, {draft = false, pullRequest = '', author = false, fork = false, assessment = 'success', record = true, repair = false, gate = 'success', ref = 'refs/heads/main', deleted = false} = {}) {
  return {
    github: {event_name: event, ref, repository: 'gaulatti/mistify', run_id: 1000, event: {deleted, pull_request: {number: 20, draft, head: {sha: head, repo: {full_name: fork ? 'external/contribution' : 'gaulatti/mistify'}}}}},
    inputs: {pull_request: pullRequest, check_author: author},
    needs: {unit: {result: 'success'}, build: {result: 'success'}, 'assess-intent': {result: assessment, outputs: {assessment: JSON.stringify({record_digest: record ? 'a'.repeat(64) : '', repair_digest: repair ? 'b'.repeat(64) : ''})}}, 'approve-and-merge': {result: gate}},
  };
}

for (const [label, c, verify, assess] of [
  ['PR', context('pull_request'), true, false],
  ['fork PR', context('pull_request', {fork: true}), true, false],
  ['draft PR', context('pull_request', {draft: true}), true, false],
  ['main', context('push'), true, false],
  ['owner installation push', context('push', {ref: 'refs/heads/policy-install/pr-24'}), false, true],
  ['deleted installer branch', context('push', {ref: 'refs/heads/policy-install/pr-24', deleted: true}), false, false],
  ['unrelated branch push', context('push', {ref: 'refs/heads/feature'}), false, false],
  ['PR metadata', context('pull_request_target'), false, true],
  ['fork metadata', context('pull_request_target', {fork: true}), false, true],
  ['draft metadata', context('pull_request_target', {draft: true}), false, false],
  ['comment', context('issue_comment'), false, false],
  ['manual verification', context('workflow_dispatch'), true, false],
  ['manual reassessment', context('workflow_dispatch', {pullRequest: '20'}), false, false],
  ['author diagnostic', context('workflow_dispatch', {author: true}), false, false],
]) {
  test(`${label}: verification and privileged review are isolated with readable native checks`, () => {
    for (const name of ['unit', 'build']) assert.equal(allowed(workflow.jobs[name].if, c), verify, name);
    assert.equal(evaluate(workflow.jobs.verify.name, c).coerceString(), verify ? 'verify' : 'metadata-only');
    assert.ok(allowed(workflow.jobs.verify.if, c, {success: false}));
    for (const step of workflow.jobs.verify.steps) assert.equal(allowed(step.if, c), verify);
    assert.equal(allowed(workflow.jobs['assess-intent'].if, c), assess);
    const approve = workflow.jobs['approve-and-merge'];
    assert.equal(evaluate(approve.name, c).coerceString(), assess ? 'approve-and-merge' : 'metadata-only-approval');
    assert.ok(allowed(approve.if, c, {success: false}));
    for (const step of approve.steps.slice(1)) assert.equal(allowed(step.if, c), assess);
  });
}

test('unit and build verification remain parallel and their combined check fails closed', () => {
  assert.equal(workflow.jobs.unit.needs, 'resolve-clouseau');
  assert.equal(workflow.jobs.build.needs, undefined);
  assert.deepEqual(workflow.jobs.verify.needs, ['unit', 'build', 'resolve-clouseau']);
  const step = workflow.jobs.verify.steps[0];
  assert.equal(step.name, 'Require passing unit and build checks');
  assert.match(step.run, /test "\$UNIT_RESULT" = success && test "\$BUILD_RESULT" = success/);
  for (const result of ['failure', 'cancelled', 'skipped']) {
    const c = context('pull_request'); c.needs.build.result = result;
    assert.ok(allowed(step.if, c, {success: false}));
    assert.equal(evaluate(step.env.BUILD_RESULT, c).coerceString(), result);
    for (const later of workflow.jobs.verify.steps.slice(1)) assert.equal(allowed(later.if, c, {success: false}), false);
  }
});

test('native coverage requires original application sources and this run/attempt artifact', () => {
  const unit = workflow.jobs.unit;
  assert.ok(!JSON.stringify(unit).includes('secrets.'));
  assert.ok(!JSON.stringify(workflow.jobs.build).includes('secrets.'));
  const upload = unit.steps.find(step => step.uses?.startsWith('actions/upload-artifact@'));
  const download = workflow.jobs.verify.steps.find(step => step.uses?.startsWith('actions/download-artifact@'));
  assert.equal(upload.with.name, download.with.name);
  assert.match(upload.with.name, /github.run_id.*github.run_attempt/);
  const check = workflow.jobs.verify.steps.find(step => step.with?.operation === 'coverage');
  assert.equal(check.with['coverage-format'], "coveragepy");
  assert.equal(check.with['coverage-report'], "coverage/coverage.json");
  assert.equal(check.with['source-directory'], "src");
  assert.equal(check.with['coverage-base-ref'], baseExpression);
  assert.equal(check.with['coverage-baseline'], '${{ github.workspace }}/coverage/baseline.json');
  assert.equal(check.with.config, '${{ runner.temp }}/clouseau-config.json');
  assert.deepEqual(config.coverage, {mode: 'changed_lines'});
  assert.match(upload.with.path, /coverage\/baseline\.json/);
  for (const name of config.verification.steps) assert.ok(workflow.jobs.verify.steps.some(step => step.name === name));
});

for (const result of ['failure', 'cancelled', 'skipped']) {
  test(`${result} assessment fails the required approval check`, () => {
    const job = workflow.jobs['approve-and-merge'];
    assert.ok(allowed(job.steps[0].if, context('pull_request_target', {assessment: result})));
    assert.equal(job.steps[0].run.trim(), 'exit 1');
  });
}

test('trusted jobs load main and provider keys exist only after the explicit verification wait', () => {
  for (const name of ['assess-intent', 'approve-and-merge', 'apply-repair', 'finalize-merge', 'request-reassessment']) {
    const checkout = workflow.jobs[name].steps.find(step => step.uses?.startsWith('actions/checkout@'));
    assert.equal(checkout.with.ref, 'main', name);
    assert.equal(checkout.with['persist-credentials'], false, name);
  }
  const assess = workflow.jobs['assess-intent'];
  const wait = assess.steps.findIndex(step => step.name === 'Wait for tests, exact coverage, and build');
  const model = assess.steps.findIndex(step => step.with?.operation === 'assess');
  assert.ok(wait > 0 && model > wait);
  assert.ok(!JSON.stringify(assess.steps.slice(0, model)).includes('secrets.'));
  assert.equal(assess.steps[model].with['wait-for-verification'], 'false');
  assert.deepEqual(assess.permissions, {actions: 'read', contents: 'read', 'pull-requests': 'read', issues: 'read'});
});

test('approval and repair consume complete evidence from their own run, without provider credentials', () => {
  for (const name of ['approve-and-merge', 'apply-repair']) {
    const job = workflow.jobs[name];
    assert.ok(!JSON.stringify(job).includes('provider-key'));
    const load = job.steps.find(step => step.uses?.startsWith('actions/download-artifact@'));
    assert.equal(load.with.name, 'clouseau-acceptance-${{ github.run_id }}-${{ github.run_attempt }}');
  }
  const gate = workflow.jobs['approve-and-merge'].steps.find(step => step.with?.operation === 'gate');
  assert.equal(gate.with['native-gate'], 'true');
  const load = workflow.jobs['approve-and-merge'].steps.find(step => step.uses?.startsWith('actions/download-artifact@'));
  assert.equal(allowed(load.if, context('pull_request_target', {record: false})), false);
});

test('forks, deleted head repositories and failed assessments cannot load the repair author credential', () => {
  const repair = workflow.jobs['apply-repair'];
  const eligible = context('pull_request_target', {repair: true, gate: 'failure'});
  assert.ok(allowed(repair.if, eligible, {success: false}));
  const deleted = structuredClone(eligible); deleted.github.event.pull_request.head.repo = null;
  for (const c of [deleted, context('pull_request_target', {fork: true, repair: true, gate: 'failure'}), context('pull_request_target', {repair: true, gate: 'failure', assessment: 'failure'}), context('pull_request_target', {repair: true, gate: 'success'}), context('pull_request_target', {repair: false, gate: 'failure'})]) assert.equal(allowed(repair.if, c, {success: false}), false);
  const mutant = repair.if.replace('github.event.pull_request.head.repo.full_name == github.repository &&', '');
  assert.equal(allowed(mutant, context('pull_request_target', {fork: true, repair: true, gate: 'failure'}), {success: false}), true);
  assert.equal(repair.permissions.contents, 'read');
  for (const [name, job] of Object.entries(workflow.jobs)) {
    if (['apply-repair', 'check-author'].includes(name)) continue;
    assert.ok(!JSON.stringify(job).includes('secrets.DEPLOYMENT_TOKEN'), name);
  }
});

test('the scored route, repair bound, protected purpose and deployment handoff remain explicit', () => {
  assert.equal(config.version, 2);
  assert.deepEqual(config.acceptance, {examiner: 'deepseek-flash', challenger: 'deepseek-v4-pro', grader: 'routed', grading_model: 'deepseek-flash', repair_attempts: 2});
  assert.ok(config.sensitive_paths.includes(".clouseau-local.json"));
  const contract = readFileSync(new URL('../clouseau.md', import.meta.url), 'utf8');
  assert.match(contract, /changed-line\s+coverage/); assert.match(contract, /explicit\s+owner\s+confirmation/);
  assert.equal(config.deployment, null);
  assert.equal(workflow.jobs['finalize-merge'].permissions.actions, 'write');
});

test('real pinned policy makes no model call for pending, failed, stale or incomplete verification', async () => {
  for (const scenario of ['pending', 'failed', 'stale', 'missing step']) {
    const pr = {number: 20, state: 'open', draft: false, head: {sha: head, ref: 'fix/cors', repo: {id: 1}}, base: {sha: 'b'.repeat(40), ref: 'main'}};
    const run = {id: 9, head_sha: scenario === 'stale' ? 'c'.repeat(40) : head, pull_requests: [{number: 20}], status: scenario === 'pending' ? 'in_progress' : 'completed', conclusion: scenario === 'failed' ? 'failure' : 'success'};
    const github = {rest: {pulls: {get: async () => ({data: pr})}, actions: {
      listWorkflowRuns: async (args) => { assert.equal(args.workflow_id, 'ci.yml'); assert.equal(args.event, 'pull_request'); return {data: {workflow_runs: [run]}}; },
      listJobsForWorkflowRun: async () => ({data: [{name: 'verify', conclusion: 'success', steps: config.verification.steps.filter(name => scenario !== 'missing step' || name !== config.verification.steps[0]).map(name => ({name, conclusion: 'success'}))}]}),
    }}, paginate: async (method, args) => (await method(args)).data};
    const c = {repo: {owner: 'gaulatti', repo: 'mistify'}, payload: {pull_request: {number: 20}}};
    const result = await require(`${engine}/src/policy.cjs`).createPolicy(config, {engine: require(`${engine}/src/central-version.cjs`).environmentEngine()}).assess({github, context: c, fetchImpl: async () => assert.fail('no model request before genuine verification')});
    assert.equal(result.decision, ['pending', 'stale'].includes(scenario) ? 'pending_technical' : 'blocked_technical');
  }
});

test('the actual wait step uses fresh PR verification, reports pending work and fails before any model execution', async (t) => {
  const project = fileURLToPath(new URL('../../', import.meta.url));
  const reviewCheckout = fileURLToPath(new URL('../../.clouseau-review', import.meta.url));
  // The unit job checks out the same exact engine at .clouseau-ci; the
  // privileged job uses a separately owned .clouseau-review checkout.
  if (!existsSync(reviewCheckout)) {
    symlinkSync(engine, reviewCheckout, 'dir');
    t.after(() => unlinkSync(reviewCheckout));
  }
  const originalCwd = process.cwd();
  process.chdir(project);
  t.after(() => process.chdir(originalCwd));
  const job = workflow.jobs['assess-intent'];
  const step = job.steps.find(step => step.name === 'Wait for tests, exact coverage, and build');
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  const invoke = new AsyncFunction('github', 'context', 'core', 'require', 'setTimeout', step.with.script);
  const pr = {number: 20, state: 'open', draft: false, head: {sha: head, ref: 'fix/cors', repo: {id: 1}}};
  const c = {repo: {owner: 'gaulatti', repo: 'mistify'}, payload: {pull_request: {number: 20, head: {sha: head}}}};
  for (const scenario of ['ready', 'pending', 'failed', 'stale', 'closed', 'missing step', 'never-ready']) {
    const messages = [];
    let polls = 0;
    const github = {rest: {pulls: {get: async () => ({data: {...pr, state: scenario === 'closed' ? 'closed' : 'open', head: {...pr.head, sha: scenario === 'stale' ? 'c'.repeat(40) : head}}})}, actions: {
      listWorkflowRuns: async () => { polls += 1; return {data: {workflow_runs: [{id: 3, head_sha: head, pull_requests: [{number: 20}], status: scenario === 'never-ready' || scenario === 'pending' && polls === 1 ? 'in_progress' : 'completed', conclusion: scenario === 'failed' ? 'failure' : 'success', html_url: 'https://github.com/gaulatti/mistify/actions/runs/3'}]}}; },
      listJobsForWorkflowRun: async () => ({data: [{name: 'verify', conclusion: 'success', steps: config.verification.steps.filter(name => scenario !== 'missing step' || name !== config.verification.steps[0]).map(name => ({name, conclusion: 'success'}))}]}),
    }}, paginate: async (method, args) => (await method(args)).data};
    const operation = invoke(github, c, {info: line => messages.push(line)}, require, (callback, ms) => { assert.equal(ms, 15000); callback(); });
    if (['ready', 'pending'].includes(scenario)) {
      await operation;
      assert.ok(messages.some(line => line.includes('Clouseau acceptance can now start')));
      assert.equal(polls, scenario === 'pending' ? 2 : 1);
    } else await assert.rejects(operation, /verification|changed|closed/i);
  }
});

test('strict owner-only automatic merge is selected and backed by repository-wide ownership', () => {
  assert.equal(config.merge_mode, "automatic");
  assert.equal(readFileSync(new URL('../CODEOWNERS', import.meta.url), 'utf8'), '* @gaulatti\n');
});

test('every coverage base consumer uses the pull request base, or the push before revision', () => {
  const unitStep = workflow.jobs.unit.steps.find(step => step.name === 'Measure base revision coverage for changed-line mode');
  const selectStep = workflow.jobs.verify.steps.find(step => step.name === 'Select accepted coverage configuration');
  const gateStep = workflow.jobs.verify.steps.find(step => step.with?.operation === 'coverage');
  const pr = context('pull_request'); pr.github.event.pull_request.base = {sha: 'b'.repeat(40)}; pr.github.event.before = undefined;
  const push = context('push'); push.github.event.pull_request = undefined; push.github.event.before = 'd'.repeat(40);
  const manual = context('workflow_dispatch'); manual.github.event.pull_request = undefined;
  for (const [expression, c, expected] of [
    [unitStep.env.BASE_SHA, pr, 'b'.repeat(40)], [unitStep.env.BASE_SHA, push, 'd'.repeat(40)], [unitStep.env.BASE_SHA, manual, ''],
    [selectStep.env.BASE_SHA, pr, 'b'.repeat(40)], [selectStep.env.BASE_SHA, push, 'd'.repeat(40)], [selectStep.env.BASE_SHA, manual, ''],
    [gateStep.with['coverage-base-ref'], pr, 'b'.repeat(40)], [gateStep.with['coverage-base-ref'], push, 'd'.repeat(40)], [gateStep.with['coverage-base-ref'], manual, ''],
  ]) assert.equal(evaluate(expression, c).coerceString(), expected);
  assert.equal(unitStep.env.BASE_SHA, baseExpression);
  assert.equal(selectStep.env.BASE_SHA, baseExpression);
  assert.equal(unitStep.env.EVENT_NAME, '${{ github.event_name }}');
  assert.equal(selectStep.env.EVENT_NAME, '${{ github.event_name }}');
});

function git(cwd, ...args) {
  const result = spawnSync('git', args, {cwd, encoding: 'utf8', env: {...process.env, GIT_AUTHOR_NAME: 't', GIT_AUTHOR_EMAIL: 't@t', GIT_COMMITTER_NAME: 't', GIT_COMMITTER_EMAIL: 't@t'}});
  assert.equal(result.status, 0, result.stderr);
  return result.stdout.trim();
}

// Runs the real `run` script of a workflow step, in a throwaway repository, with the step's own env keys.
function runStep(step, repository, env) {
  const temp = mkdtempSync(join(tmpdir(), 'ci-step-'));
  if (!existsSync(join(repository, '.clouseau-ci'))) symlinkSync(engine, join(repository, '.clouseau-ci'), 'dir');
  const result = spawnSync('bash', ['-eo', 'pipefail', '-c', step.run], {cwd: repository, encoding: 'utf8', env: {PATH: process.env.PATH, RUNNER_TEMP: temp, ...env}});
  const config = join(temp, 'clouseau-config.json');
  const selected = existsSync(config) ? readFileSync(config, 'utf8') : null;
  rmSync(temp, {recursive: true, force: true});
  return {...result, selected};
}

function fixtureRepository() {
  const repository = mkdtempSync(join(tmpdir(), 'ci-repo-'));
  git(repository, 'init', '-q', '-b', 'main');
  mkdirSync(join(repository, '.github'));
  writeFileSync(join(repository, '.github/clouseau.json'), '{"coverage":{"mode":"exact"}}');
  git(repository, 'add', '-A'); git(repository, 'commit', '-qm', 'base');
  const base = git(repository, 'rev-parse', 'HEAD');
  writeFileSync(join(repository, '.github/clouseau.json'), '{"coverage":{"mode":"changed_lines"}}');
  git(repository, 'commit', '-qam', 'head');
  const bare = mkdtempSync(join(tmpdir(), 'ci-repo-'));
  git(bare, 'init', '-q', '-b', 'main');
  writeFileSync(join(bare, 'readme'), 'x'); git(bare, 'add', '-A'); git(bare, 'commit', '-qm', 'no config');
  return {repository, base, noConfig: git(bare, 'rev-parse', 'HEAD'), bare};
}

test('a push selects the accepted configuration from the before revision and fails clearly when it is unusable', (t) => {
  const {repository, base, bare, noConfig} = fixtureRepository();
  t.after(() => { rmSync(repository, {recursive: true, force: true}); rmSync(bare, {recursive: true, force: true}); });
  const select = workflow.jobs.verify.steps.find(step => step.name === 'Select accepted coverage configuration');
  const accepted = runStep(select, repository, {EVENT_NAME: 'push', BASE_SHA: base});
  assert.equal(accepted.status, 0, accepted.stderr);
  assert.equal(accepted.selected, '{"coverage":{"mode":"exact"}}');
  for (const [label, sha, pattern] of [
    ['first push', '0'.repeat(40), /created it/],
    ['malformed before', 'not-a-sha', /not a commit SHA/],
    ['empty before', '', /not a commit SHA/],
    ['unknown before', 'e'.repeat(40), /not in the checkout/],
  ]) {
    const failed = runStep(select, repository, {EVENT_NAME: 'push', BASE_SHA: sha});
    assert.notEqual(failed.status, 0, label);
    assert.match(failed.stdout + failed.stderr, pattern, label);
    assert.equal(failed.selected, null, `${label}: no configuration is selected`);
  }
  git(repository, 'fetch', '-q', bare, 'main');
  const missing = runStep(select, repository, {EVENT_NAME: 'push', BASE_SHA: noConfig});
  assert.notEqual(missing.status, 0);
  assert.match(missing.stdout + missing.stderr, /no accepted \.github\/clouseau\.json/);
});

test('pull request configuration selection is unchanged, including the base without a configuration', (t) => {
  const {repository, base, bare, noConfig} = fixtureRepository();
  t.after(() => { rmSync(repository, {recursive: true, force: true}); rmSync(bare, {recursive: true, force: true}); });
  const select = workflow.jobs.verify.steps.find(step => step.name === 'Select accepted coverage configuration');
  assert.equal(runStep(select, repository, {EVENT_NAME: 'pull_request', BASE_SHA: base}).selected, '{"coverage":{"mode":"exact"}}');
  const bootstrap = runStep(select, repository, {EVENT_NAME: 'pull_request', BASE_SHA: 'f'.repeat(40)});
  assert.equal(bootstrap.status, 0);
  assert.equal(bootstrap.selected, '{"coverage":{"mode":"changed_lines"}}');
  git(repository, 'fetch', '-q', bare, 'main');
  assert.equal(runStep(select, repository, {EVENT_NAME: 'pull_request', BASE_SHA: noConfig}).selected, '{"coverage":{"mode":"changed_lines"}}');
  assert.equal(runStep(select, repository, {EVENT_NAME: 'workflow_dispatch', BASE_SHA: ''}).selected, '{"coverage":{"mode":"changed_lines"}}');
});

test('the base coverage measurement refuses an unusable push base but stays silent for other events', (t) => {
  const {repository, base, bare} = fixtureRepository();
  t.after(() => { rmSync(repository, {recursive: true, force: true}); rmSync(bare, {recursive: true, force: true}); });
  const measure = workflow.jobs.unit.steps.find(step => step.name === 'Measure base revision coverage for changed-line mode');
  for (const sha of ['0'.repeat(40), 'not-a-sha', '', 'e'.repeat(40)]) {
    const failed = runStep(measure, repository, {EVENT_NAME: 'push', BASE_SHA: sha});
    assert.notEqual(failed.status, 0, sha);
    assert.match(failed.stdout + failed.stderr, /::error::Changed-line coverage needs/);
  }
  // An exact-mode base needs no baseline, so a push on it measures nothing and succeeds.
  const exact = runStep(measure, repository, {EVENT_NAME: 'push', BASE_SHA: base});
  assert.equal(exact.status, 0, exact.stderr);
  for (const event of ['pull_request', 'workflow_dispatch']) assert.equal(runStep(measure, repository, {EVENT_NAME: event, BASE_SHA: ''}).status, 0);
});
