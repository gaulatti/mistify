import * as cdk from 'aws-cdk-lib';
import { LogGroup, RetentionDays } from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';

/**
 * Represents the MistifyStack, an AWS CDK stack that provisions infrastructure resources.
 *
 * This stack creates a CloudWatch Log Group named `/services/monitor` with a retention period of one week.
 * The log group is configured to be destroyed upon stack deletion.
 *
 * @remarks
 * - The log group is intended for monitoring service logs.
 * - Uses AWS CDK constructs and removal policies.
 *
 * @param scope - The parent construct.
 * @param id - The unique identifier for this stack.
 * @param props - Optional stack properties.
 */
export class MistifyStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    /**
     * Log Group
     */
    new LogGroup(this, `${this.stackName}ServiceLogGroup`, {
      logGroupName: '/services/mistify',
      retention: RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
  }
}
