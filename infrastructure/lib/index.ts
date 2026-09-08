import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';

/**
 * Compatibility stack retained so the next normal CDK deployment removes the
 * former `/services/mistify` CloudWatch log group from the live stack.
 *
 * Mistify application logs are intentionally host-local and bounded by the
 * Docker logging options in both deployment workflows. This stack must not
 * provision CloudWatch Logs resources or grant application log-writer access.
 */
export class MistifyStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);
  }
}
