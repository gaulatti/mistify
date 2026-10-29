import {data, Evaluator, Lexer, Parser} from '@actions/expressions';
import {Expr, FunctionCall} from '@actions/expressions/ast';
import {truthy} from '@actions/expressions/result';

const statuses = new Set(['always', 'success', 'failure', 'cancelled']);

function expressionData(value) {
  if (typeof value === 'boolean') return new data.BooleanData(value);
  if (typeof value === 'number') return new data.NumberData(value);
  if (typeof value === 'string') return new data.StringData(value);
  if (value == null) return new data.Null();
  return new data.Dictionary(...Object.entries(value).map(([key, value]) => ({key, value: expressionData(value)})));
}

function hasStatusFunction(node) {
  if (node instanceof FunctionCall && statuses.has(node.functionName.lexeme.toLowerCase())) return true;
  return Object.values(node).some(value =>
    value instanceof Expr ? hasStatusFunction(value) :
      Array.isArray(value) && value.some(child => child instanceof Expr && hasStatusFunction(child)));
}

// Use GitHub's expression engine on the actual YAML, including its implicit
// success() condition. This does not execute workflow commands or replace CI.
export function evaluate(expression, context, {condition = false, success = true, cancelled = false} = {}) {
  const source = String(expression ?? 'success()').trim().replace(/^\$\{\{\s*|\s*\}\}$/g, '');
  const functions = new Map([...statuses].map(name => [name, {
    name, minArgs: 0, maxArgs: 0,
    call: () => new data.BooleanData({always: true, success, cancelled, failure: !success && !cancelled}[name]),
  }]));
  const ast = new Parser(new Lexer(source).lex().tokens, Object.keys(context), [...functions.values()]).parse();
  if (condition && !hasStatusFunction(ast) && !success) return new data.BooleanData(false);
  return new Evaluator(ast, expressionData(context), functions).evaluate();
}

export function allowed(expression, context, options = {}) {
  return truthy(evaluate(expression, context, {condition: true, ...options}));
}
