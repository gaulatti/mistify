# Mistify review contract

Mistify provides unified AI helpers behind a FastAPI and gRPC service: language
detection, classification, translation, embeddings, clustering, editorial
scoring, analysis and media operations, with Prometheus observability and a CDK
infrastructure package. Callers rely on its endpoint and protobuf contracts,
deterministic fixture-driven behaviour, and bounded, content-free metrics.

Preserve the public HTTP and gRPC contracts, the generated protobuf bindings
(regenerated wholesale from `proto/`), offline deterministic behaviour of the
required test suite (model loading or downloading is a failure, not a slow
path), bounded metric labels, and the deployment gate that requires the
required tests before any image or stack is deployed. Bug fixes, meaningful
tests, observability, documentation and dependency maintenance serve this
existing service without a new product decision. A new endpoint, operation or
model dependency needs documentation or a linked requirement that establishes
its need and limits. Deployment stays a separate, explicitly authorized
workflow outside this review contract.

Assess purpose and evidence rather than the number of changed files or lines.
Ask gaulatti for a second opinion for a product-direction or authority decision.
Ordinary implementation uncertainty and repairable bugs belong to automated
investigation, not routine human code review. Changes to review controls (this
contract, `.github/clouseau.json`, the CI workflow, the local recipe) require
explicit owner confirmation.

Every material capability and mechanism must independently score at least 80
for fit, need and proportionality under Clouseau's shared protocol. A causal
technical finding at or above 70 blocks; unrelated features or good scores
cannot average away a defect. Integrity and review-completeness gates remain
independent.

Tests must assert actual behaviour. Reject fabricated coverage, disabled tests,
coverage pragmas or exclusions, swallowed errors, unconditional assertions and
mocks that replace the behaviour being claimed. The required pytest suite under
coverage.py with branch measurement and no exclusions, and the infrastructure
build, tests and synthesis, remain mandatory. Every pull request must keep exactly
100% line and branch coverage of `src`, including the generated protobuf
bindings, which the shared gate does not allow to be omitted; a pull request
below that stays for human review.
