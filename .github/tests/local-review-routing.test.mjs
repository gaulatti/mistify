import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {parse} from 'yaml';
import {allowed, evaluate} from './workflow-expressions.mjs';

const workflow = parse(readFileSync(new URL('../workflows/ci.yml', import.meta.url), 'utf8'));
const action = parse(readFileSync(new URL('../../.clouseau-ci/action.yml', import.meta.url), 'utf8'));
const review = workflow.jobs['assess-intent'];

test('the actual receipt route cannot run the CI AI step, including for a malformed or duplicated stamp', async () => {
  const route = review.steps.find(step => step.id === 'review-route');
  const invoke = new (Object.getPrototypeOf(async function() {}).constructor)('context', 'core', route.with.script);
  const local = review.steps.find(step => step.with?.operation === 'verify-local');
  const remote = review.steps.find(step => step.with?.operation === 'assess');
  for (const [body, expected] of [[undefined, false], ['A normal PR.', false],
    [`<!-- clouseau-local-review:v1 ${'a'.repeat(40)} -->`, true],
    ['clouseau-local-review:v1 malformed', true],
    ['clouseau-local-review:v1 duplicated clouseau-local-review:v1', true]]) {
    let selected;
    await invoke({payload: {pull_request: {body}}}, {setOutput: (name, value) => {
      assert.equal(name, 'local'); selected = value;
    }});
    const context = {steps: {'review-route': {outputs: {local: selected}}}};
    assert.equal(allowed(local.if, context), expected);
    assert.equal(allowed(remote.if, context), !expected);
    if (expected) {
      assert.equal(local.with['provider-key'], undefined);
      assert.equal(local.with['typesafe-key'], undefined);
      assert.ok(!JSON.stringify(local).includes('secrets.'));
    }
  }
  assert.equal(local.uses, remote.uses);
  assert.equal(local.with['wait-for-verification'], 'false');
  assert.match(review.outputs.assessment, /steps.local-review.outputs.assessment \|\| steps.assess.outputs.assessment/);
});

test('the real action removes provider credentials in receipt-only mode and forbids paid fallback', () => {
  const step = action.runs.steps.find(step => step.id === 'assess');
  for (const [operation, key] of [['verify-local', ''], ['assess', 'fictional-key']]) {
    const context = {inputs: {operation, 'provider-key': 'fictional-key', 'typesafe-key': 'fictional-key'}};
    assert.equal(evaluate(step.env.DEEPSEEK_API_KEY, context).coerceString(), key);
    assert.equal(evaluate(step.env.TYPESAFE_API_KEY, context).coerceString(), key);
    assert.equal(evaluate(step.env.CLOUSEAU_RECEIPT_ONLY, context).value, operation === 'verify-local');
  }
  assert.match(step.with.script, /receiptOnly: process.env.CLOUSEAU_RECEIPT_ONLY === 'true'/);
});

test('local agent installation has its own verification recipe and the Storybook deployment stays unselected', () => {
  const recipe = JSON.parse(readFileSync(new URL('../../.clouseau-local.json', import.meta.url)));
  const config = JSON.parse(readFileSync(new URL('../clouseau.json', import.meta.url)));
  assert.ok(recipe.verification.length > 0 && recipe.verification.every(c => Array.isArray(c)));
  assert.equal(recipe.coverage.format, "coveragepy");
  assert.equal(recipe.coverage.source_directory, "src");
  assert.ok(!JSON.stringify(recipe).match(/provider|secret|api_key|deepseek|typesafe/i));
  assert.equal(config.deployment, null);
  assert.equal(config.local_review.keys[0].github_actor, config.manual_reviewer);
});
