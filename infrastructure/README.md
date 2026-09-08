# Mistify infrastructure

Mistify keeps its CDK stack as a compatibility shell so a normal stack update
can remove the former `/services/mistify` CloudWatch log group. The synthesized
template intentionally contains no CloudWatch Logs resources or application
log-writer grants.

Production application logs remain on the Mistify host through Docker's
`json-file` driver. Both deployment workflows cap each container at five 50 MB
rotated files, and operators can inspect bounded output with:

```bash
docker logs --tail 200 mistify
```

Use operation, callback, retry, and queue metrics for ongoing diagnosis. Do not
copy media locations, object keys, credentials, or payload contents into
operational evidence.

## Verification

Run the following from this directory:

```bash
npm ci
npm run build
npm test -- --runInBand
npm run cdk:synth
```

The test and synthesized template must show zero `AWS::Logs::LogGroup`
resources and no `logs:` IAM actions. From the repository root, the required
Python suite verifies that both deployment paths keep bounded local logging;
the existing media and callback retry tests remain in the same required gate.

After an approved deployment, verify `/health`, one representative media
operation, and its downstream callback/persistence result. Confirm that
`/services/mistify` is absent and that `AWS/Logs` `IncomingBytes` receives no
new data without reading or copying log events.

## Rollback

Revert the removal commit and deploy the `Mistify` CDK stack through the normal
reviewed workflow to recreate `/services/mistify`. Reverting the infrastructure
change does not redirect the container by itself: the deployment workflows
remain on bounded host-local logging unless a separately reviewed change alters
their Docker log driver.
