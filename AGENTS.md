<!-- BEGIN AGENTIC-CODING MANAGED RULES -->
# General agent instructions

These rules are the reference standard for agent work across all repositories.
Repository-specific instructions may add stricter requirements, but must not
weaken or contradict these rules.

## Shared standard freshness

- At the beginning of every agent task in a managed repository, before planning
  or substantive repository work, run
  `.agents/skills/ai-agentic-update/scripts/update` once from the repository
  root.
- The updater must fetch the central `agentic-coding` standard and, when a newer
  version exists, refresh only the managed rules and centrally owned skills in
  the current repository and create a narrowly scoped Conventional Commit for
  those managed files. For public repositories it must also verify the
  repository wiki checkout at `./wiki`, clone the repository's wiki there when
  the remote wiki exists, and validate that an existing checkout points to the
  expected wiki remote. Reread the updated root `AGENTS.md`, any applicable
  skills, and the relevant wiki pages for public repositories before
  continuing.
- Do not silently skip or weaken the freshness check. If the central source
  cannot be verified or the update fails, stop before other repository mutations
  and report the exact failure.
- The freshness update must preserve repository-local instructions, local-only
  skills, wiki changes, and unrelated worktree changes. It must not include
  product files in its metadata commit, and it must not push, deploy, reset,
  rebase, or discard changes.
- Do not continue product work with dirty managed metadata. If the updater
  cannot commit its `AGENTS.md`, managed-skills manifest, and centrally owned
  skills cleanly, stop and report the failure instead of treating those files
  as unrelated product work or repeatedly narrating their presence.

## Repository wiki and documentation

- Wiki requirements apply only to public managed repositories. Determine the
  repository's GitHub visibility (for example with `gh repo view`) before
  relying on the wiki contract. When visibility cannot be determined, keep the
  wiki contract active, attempt the wiki checkout, and report the condition.
  Private repositories have no wiki requirement and rely on in-repository
  documentation.
- Every public managed repository's GitHub wiki must be available as a separate
  Git checkout at `./wiki`. The startup updater derives the wiki URL from the
  repository's `origin`, clones it when available, and excludes `./wiki` from
  the parent repository through local Git metadata. Never treat the wiki as
  ordinary untracked parent-repository content. A private repository with an
  existing `./wiki` checkout keeps it preserved and current, but the updater
  never creates one there.
- Before planning or changing code in a public repository, inspect the relevant
  wiki pages alongside the code and repository documentation. Do not rely on a
  stale architectural, operational, API, configuration, deployment, or
  user-workflow assumption when the wiki can establish the intended contract.
- Documentation is part of every implementation, fix, refactor, migration, and
  configuration change in every repository. Update the relevant documentation in
  the same task whenever behavior, architecture, interfaces, setup, operations,
  deployment, or troubleshooting guidance changes; use the wiki pages for public
  repositories and in-repository documentation otherwise.
- When a code change genuinely has no documentation impact, verify that the
  relevant documentation remains accurate and state that explicitly in the
  handoff. Do not use “no documentation impact” without checking.
- Preserve unrelated or uncommitted wiki work. Wiki commits and pushes are
  separate external writes: make them only when the task authorizes publishing,
  and report code-repository and wiki-repository status separately.
- For public repositories, when the repository has no `origin`, its wiki remote
  does not exist yet, or wiki access fails, report that condition explicitly
  and continue using in-repository documentation unless the task requires wiki
  publication. Do not fabricate a wiki remote or silently skip documentation.

## Feature completeness

- Treat every requested behavior change as a complete feature, regardless of
  how small or local the first code edit appears. Never scope the work as merely
  touching a named file, component, endpoint, constant, or layer.
- Before editing, trace the feature from the user's action to the observable
  outcome. Inspect every applicable input, UI control, client request, API
  contract, validation rule, query and bound, persistence path, background job,
  cache, event, response mapping, and UI consumer.
- Search the entire feature path for duplicated constraints, limits, defaults,
  labels, time windows, feature flags, fallback behavior, and compatibility
  paths. A changed value in one layer does not prove that the feature changed
  end to end.
- Do not ship a partial layer. A UI change is incomplete until every producer
  and API constraint supplies the behavior it exposes. A backend change is
  incomplete until every relevant consumer can request, handle, and correctly
  render it.
- Treat domain models, lifecycle states, relationships, workflows,
  configuration, and externally visible behavior as cross-cutting until the
  repository proves otherwise.
- When changing an enum, lifecycle state, relationship, or derived field, audit
  creation, update, validation, persistence, hydration, serialization, default
  queries, saved filters, URLs, selectors, actions, views, downstream
  publishing, cleanup, concurrency, and rollback behavior.

## Acceptance and verification

- Define user-observable acceptance criteria before implementation. Use those
  criteria to identify affected surfaces and required tests.
- Add regression coverage at the boundary where the feature previously broke,
  not only around a helper or constant. Prove that data crosses the relevant
  layers and reaches the observable result.
- Run the relevant tests, type checks, linters, format checks, builds,
  integration checks, and framework startup checks for the affected surfaces.
- Perform a final feature-path audit after implementation. Revisit every
  producer, constraint, transformation, and consumer found before editing and
  confirm that each is updated or explicitly unaffected.
- Compilation, persisted data, a successful API response, a successful image
  build, or the presence of a new control is not by itself evidence that the
  feature works. Verify the actual runtime, UI, browser, device, deployment, or
  integration behavior appropriate to the request.
- Do not declare a feature complete or ship it while any required layer,
  boundary, regression check, runtime check, or user-observable acceptance
  criterion remains unverified. State any remaining limitation in the handoff.

## Explicit behavior and source-of-truth boundaries

- Do not introduce fallback behavior unless the user explicitly requests it.
  Prefer fail-fast behavior, explicit errors, and visible empty/error states
  over silent fallback values.
- Do not invent alternate endpoint or URL patterns, default services, inferred
  data paths, speculative compatibility layers, future-proof toggles, or flags.
- Do not introduce new environment variables, or mirror backend variables into
  frontend variables, unless the requested behavior requires and authorizes it.
- Treat values and URLs supplied by their authoritative producer as the source
  of truth. Do not reconstruct media URLs from filenames or API base paths, and
  do not silently repair malformed producer data.
- When requirements or authoritative behavior are genuinely ambiguous, inspect
  the repository and its existing contracts first. Ask instead of inventing a
  new product or infrastructure policy.

## Change and repository hygiene

- Preserve unrelated worktree changes. Stage and commit only files within the
  requested scope.
- Keep commits and pull requests focused and atomic. Follow existing repository
  patterns and use the repository's established test infrastructure.
- Use Conventional Commits for commit messages and pull-request titles:
  `type(scope): imperative lowercase subject` with no trailing period. Supported
  types are `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`,
  `ci`, `chore`, and `revert`. Mark breaking changes explicitly.
- Do not bypass hooks or required checks except in an explicitly justified,
  exceptional situation. Ensure required CI passes and address review feedback.
- Update relevant tests and documentation with behavior changes.

## Secrets, sensitive data, and production safety

- Never expose credentials, tokens, password hashes, secret values, complete
  secret payloads, or sensitive content bodies in commands, logs, commits,
  screenshots, test fixtures, or responses.
- Keep backend secrets out of frontend variables, Docker build arguments, and
  other client-visible or build-time surfaces.
- Do not create, modify, rotate, or disclose production credentials unless the
  user explicitly requests that exact operation.
- Production changes must go through the repository's code, review, migration,
  and deployment path. Do not repair a live container or production database by
  hand unless the user explicitly authorizes that operational intervention.

## Full-stack local development

- A repository with both a browser frontend and an application backend must
  provide a root Docker Compose stack as its supported local-development entry
  point. `docker compose up --build` must start the complete application path
  after documented one-time prerequisites; do not require agents to start the
  frontend, backend, database, or required supporting services separately on the
  host.
- Compose must build development targets, publish stable documented ports,
  provide source-watch or bind-mount behavior appropriate to the framework, and
  keep container-only dependencies such as `node_modules` out of host mounts.
- Add health checks for stateful dependencies and the backend. Gate dependent
  startup on readiness, not merely container creation; the frontend must resolve
  the backend through the documented browser-visible URL.
- Local Compose is not a production deployment mechanism. Keep production
  deployment, credentials, and data out of the local stack.
- When ports, configuration, dependencies, migrations, startup commands, or
  service topology change, update Compose, development Dockerfiles, health
  checks, local documentation, and browser verification together.
- Do not call a full-stack change complete until the application has been built
  and exercised through the Compose-served browser UI and normal backend path.

## Prometheus observability for managed backends

- Before changing backend behavior, inspect the repository's existing metrics,
  collectors, instrumentation helpers, private metrics endpoint, and Prometheus
  scrape boundary. Preserve established names and semantics unless the task
  explicitly includes a documented metric migration.
- A managed backend must expose the applicable baseline telemetry for its
  architecture: service/build information, a runtime/process collector,
  normalized HTTP or RPC request count/duration/result metrics, dependency and
  retry signals, queue or job signals, bounded domain metrics, and a private
  scraper endpoint when the framework supports one. A component that genuinely
  does not have a listed concern need not fabricate a metric for it.
- Backend behavior changes must add or update behavior-specific metrics when
  they affect dependencies, retries, queues or jobs, persistence, delivery,
  lifecycle state, freshness, or degraded/unavailable state. Instrument both
  successful and failed transitions so operators can distinguish healthy work,
  rejection, retry, and failure.
- Use stable metric names, Prometheus base-unit suffixes, and bounded labels
  derived from controlled enums or normalized route/result classifications.
  Document metric meaning and any deliberate rename or semantic migration.
- Never use unbounded, sensitive, or content-bearing labels. Prohibited labels
  include user IDs, unbounded tenant IDs, tokens, credentials, URLs, request or
  response bodies, content text, device identifiers, free-form error strings,
  and raw external identifiers. Put correlation details in appropriately
  protected logs, not metric labels.
- Application repositories own their instrumentation, private exposure,
  behavior-specific tests, and metric documentation. Prometheus deployment,
  scrape configuration, storage, dashboards, and alerts belong to the
  observability infrastructure repository unless the ticket explicitly places
  those surfaces in scope.
- Tests must cover success and error transitions and assert bounded label sets.
  Verification must also exercise and parse the real collector or `/metrics`
  output; unit assertions that only prove an increment call are insufficient.
- Update the applicable wiki or in-repository observability documentation with
  metrics, labels, endpoint/security boundaries, and operational meaning. When
  an inspected backend change genuinely has no metrics impact, record that
  conclusion explicitly in the handoff rather than silently omitting telemetry.

## Configuration ownership and AWS Secrets Manager

- Production backend application configuration must be Secrets Manager-owned.
  Keep the container environment limited to bootstrap and infrastructure values
  needed before secrets can load, such as environment identity, AWS region,
  secret identifier/key, and required listener metadata.
- Store database URLs and credentials, private service credentials, signing
  material, provider keys, and other application configuration in an
  application-scoped Secrets Manager payload. Public frontend configuration is
  not secret, but must contain no backend secret material.
- Load and validate secrets before migrations, framework/module initialization,
  service-client construction, workers, one-off jobs, or any other entry point
  reads application configuration. Every executable entry point must share the
  same loader and validation contract.
- Parse a documented object selected by the application key and copy only an
  explicit allowlist of supported scalar fields. Do not inject arbitrary secret
  payload keys into the process environment.
- In production, the secret payload is authoritative. Missing identifiers,
  missing required keys, malformed values, retrieval errors, or invalid
  configuration must fail startup; do not fall back to stale or standalone
  environment values.
- Local Compose must provide explicit non-production configuration without
  requiring AWS access. Never log secret values or complete secret payloads, and
  grant runtime identities access only to the required secret.
- Add tests for selection/parsing, allowlisting, local override behavior,
  production precedence, missing/malformed configuration, and every startup path
  that must load secrets before use.

## Local PostgreSQL and seed data

- PostgreSQL is the default relational database for full-stack applications.
  When an application uses PostgreSQL, local development and browser testing
  must use a PostgreSQL service in the repository's Docker Compose stack. Do not
  depend on a developer-installed PostgreSQL instance or a shared/cloud database
  for the supported local workflow.
- Local development must use the same database engine and migration behavior as
  production. Do not quietly substitute PostgreSQL for an explicitly retained
  non-PostgreSQL production database; treat an engine migration as a separate,
  explicitly authorized cross-cutting change.
- Use explicit non-production credentials, a named data volume, a `pg_isready`
  health check, and the Compose service name as the backend database host. Do not
  expose the database port unless host access is actually required.
- Apply committed migrations automatically after PostgreSQL becomes healthy and
  before the backend is marked ready. Do not use schema auto-synchronization as
  a substitute for migration coverage.
- Provide an idempotent, deterministic local seed path that runs after
  migrations and creates enough users, roles, relationships, and representative
  domain data to exercise every primary UI workflow. Seed data must use only
  fictional non-sensitive values and must not require Cognito, production
  secrets, or external production services.
- A fresh local stack must become usable without manual database edits. Provide
  a documented, deterministic way to reset and reseed local state while making
  destructive scope explicit.
- Test migration-plus-seed behavior from an empty database and verify the seeded
  state through the backend and browser UI, not only with direct database reads.

## Runtime and deployment evidence

- A successful local build, container build, compilation, push, or CI job is not
  proof that a deployment completed or that the running application works.
- When deployment is in scope, monitor the actual deployment workflow through
  completion, then verify the running service with its health check and the
  appropriate user-visible behavior.
- Framework builds do not necessarily prove that dependency injection, module
  graphs, migrations, workers, or application startup succeed. Exercise the
  relevant startup/runtime boundary.
- Report separately on source control, CI, migration, deployment, health, and
  user-visible verification; do not collapse them into a single success claim.

## Authentication and browser testing

- Exercise authenticated features through the application's normal browser,
  API, validation, authorization, and ownership paths. Do not mock authenticated
  APIs merely to bypass login.
- A test-auth adapter must be local/test-only, visibly identifiable, and refused
  at startup in production. Never add test-auth configuration or secrets to
  production infrastructure.
- Keep normal guards, validation, authorization, and ownership enforcement
  active under test auth. Prefer short-lived credentials and deterministic,
  isolated identities for parallel agents and tests.
- When the repository provides a documented agent login, session endpoint, or
  fixture reset path, use it rather than inventing a bypass. Keep exact ports,
  routes, commands, and secrets in repository-specific instructions.
- Every private browser UI must have a Compose-enabled offline authentication
  path that requires no Cognito account, hosted login page, cookie injection,
  static production token, or network call to Cognito. Public unauthenticated
  sites and Storybook-only packages do not need a fabricated login path.
- The offline path must issue short-lived local credentials from a guarded
  backend endpoint or guarded local login route, then pass them through the
  normal frontend session, API authentication, authorization, and ownership
  code. Use a distinct local issuer, audience, and signing key.
- The frontend must enter test auth automatically when Compose enables it and
  display an unmistakable `TEST AUTH` indicator. The backend must hide test
  endpoints when disabled and refuse to start when test auth and a production
  environment are combined.
- Support deterministic seeded identities, roles, permissions, and domain
  fixtures. Where user-owned state matters, support named isolated identities
  for parallel agents and a guarded reset/reseed operation.
- Add regression coverage for production refusal, disabled-endpoint behavior,
  token expiry/signature/issuer/audience validation, normal guard traversal,
  identity isolation, permissions, and reset determinism. Validate the actual
  Compose-served UI without Cognito before reporting success.

## Version-sensitive frameworks and external documentation

- Before changing a version-sensitive framework or SDK, identify the version
  actually pinned by the repository and consult the official documentation for
  that exact version.
- Do not assume that current, latest, or another repository's framework behavior
  applies. Preserve repository-specific prohibitions and configuration contracts.

## User-interface and design-system changes

- Use the repository's established typography, design tokens, theme variables,
  spacing, radii, component patterns, and visual language. Do not replace them
  with generic fonts, unadapted UI kits, or unexplained stock styling.
- Define shared colors and other theme values centrally and consume them through
  the repository's token system. Do not introduce or replace a CSS/UI framework
  without explicit authorization.
- Prefer purposeful, performant motion over numerous incidental animations.
  Check contrast, accessibility, responsive behavior, and reduced-motion needs.
- Provide a brief rationale for intentional, unexpected design choices.
- When a repository has Storybook or another visual fixture system, every change
  to a UI, template, renderer, or visual contract must add or update a relevant
  fixture in the same change. Register and order new fixtures where required,
  mention them in the handoff, and provide screenshots or short recordings for
  visual changes when the workflow supports them.

## Publishing and externally visible state

- Treat publication state as a security and data-integrity boundary. Only
  content explicitly in the repository's publishable state may enter publish,
  rerender, aggregation, feed, or social-delivery paths.
- Enforce publication guards at every publishing boundary. Never rely solely on
  an upstream caller. Missing, unknown, draft, scheduled, archived, or otherwise
  non-publishable states must fail closed and perform no live writes.
- Unpublishing or deleting published content must remove every live artifact
  required by the product contract, not merely rebuild listings or feeds.
- Alternate publishing and mutation paths must reuse the same guards. Add
  regression tests proving that non-publishable content produces no live output.
- Preserve explicitly documented domain exceptions, such as permanent-link
  retention, only in the repositories whose product contract requires them.

## Architecture and dependency boundaries

- Keep core, shared, and provider-only modules dependency-light. Do not make
  them depend on API, media, pipeline, or feature modules without necessity.
- Before adding a circular-dependency escape hatch, inspect the full import path
  and prefer extracting shared behavior into a smaller acyclic module.
- Workers and alternate entry points must use the same authoritative
  configuration and validation contracts as the main process.
- Preserve full application-module, startup, or dependency-graph regression
  coverage when the framework can compile successfully but fail at runtime.

## Example

Expanding a scheduling interface is not complete when additional time slots
render. The corresponding API query, time bounds, persistence rules, count
aggregation, final-slot boundary, labels, and rendered occupancy indicators
must all cover the expanded window, and a regression test must exercise that
end-to-end contract.
<!-- END AGENTIC-CODING MANAGED RULES -->

<!-- Repository-specific instructions below are maintained locally. -->
