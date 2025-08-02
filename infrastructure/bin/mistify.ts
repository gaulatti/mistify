#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { config } from 'dotenv';
import { MistifyStack } from '../lib';

/**
 * Loads the environment variables from the .env file.
 */
config();

/**
 * I need perfection!
 * https://www.youtube.com/watch?v=26ELh7P2ZtA
 */
const app = new cdk.App();
new MistifyStack(app, 'Mistify', {
  /* If you don't specify 'env', this stack will be environment-agnostic.
   * Account/Region-dependent features and context lookups will not work,
   * but a single synthesized template can be deployed anywhere. */
  /* Uncomment the next line to specialize this stack for the AWS Account
   * and Region that are implied by the current CLI configuration. */
  // env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
  /* Uncomment the next line if you know exactly what Account and Region you
   * want to deploy the stack to. */
  // env: { account: '123456789012', region: 'us-east-1' },
  /* For more information, see https://docs.aws.amazon.com/cdk/latest/guide/environments.html */
});
