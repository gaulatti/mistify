import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import { MistifyStack } from '../lib';

describe('Mistify infrastructure', () => {
  test('owns no CloudWatch application logging resources or writer grants', () => {
    const app = new cdk.App();
    const stack = new MistifyStack(app, 'Mistify');
    const template = Template.fromStack(stack);

    template.resourceCountIs('AWS::Logs::LogGroup', 0);
    expect(JSON.stringify(template.toJSON())).not.toContain('"logs:');
  });
});
