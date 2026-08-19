---
name: full-stack-repository-contract
description: Audit or implement the required local and production foundation for repositories with a browser frontend and backend, including Docker Compose, Secrets Manager configuration, Dockerized PostgreSQL with deterministic seed data, and Cognito-free offline UI authentication. Use for full-stack repository setup, modernization, or compliance work; do not trigger for frontend-only, backend-only, public static, Storybook-only, or native-only projects unless the user asks to add the missing full-stack surface.
---

# Full Stack Repository Contract

Bring an applicable repository to a one-command, browser-verifiable local state
without weakening its production security boundaries.

## Establish applicability

Confirm that the repository owns both a browser frontend and an application
backend. Identify its database, authentication model, production configuration
source, workers/jobs, migrations, deployment path, and existing local tooling.
PostgreSQL is the default for new relational persistence. Do not silently replace
an explicitly retained production database with PostgreSQL only for local use,
or add an authentication bypass to a public UI; database-engine migration needs
explicit scope and full data/deployment planning.

Read [references/reference-contracts.md](references/reference-contracts.md)
before designing or reviewing an implementation. Use the concrete repositories
as patterns, then adapt names, frameworks, ports, and domain fixtures to the
target rather than copying code blindly.

## Define acceptance before editing

The target state is observable:

1. From a fresh local database, `docker compose up --build` starts the required
   dependencies, applies migrations, loads deterministic seed data, starts the
   backend only after dependencies are ready, and serves the frontend.
2. An agent opens the documented local URL, enters a private UI without Cognito,
   sees a visible test-auth indicator, and exercises primary workflows through
   the real API, guards, authorization, ownership, and persisted seed data.
3. Production entry points load an allowlisted, validated application payload
   from Secrets Manager before configuration consumers initialize and fail
   closed when required configuration is unavailable.
4. Test auth cannot exist in production, and local startup needs no production
   credentials or AWS access.

## Work through the complete path

- Audit root Compose files, development Dockerfiles, health checks, dependency
  readiness, mounts/watch behavior, ports, browser-visible URLs, migrations,
  seeds, backend/bootstrap entry points, workers/jobs, configuration modules,
  frontend session bootstrap, backend guards, test endpoints, fixtures, reset
  behavior, tests, and local documentation.
- Preserve existing product authentication and authorization. Test auth is an
  alternate local credential issuer, not an alternate permission system.
- Make database seeds idempotent and deterministic. Include roles,
  relationships, and representative domain states needed by primary screens;
  avoid random identifiers where tests or URLs need stable references.
- Keep production configuration authoritative in Secrets Manager and local
  configuration explicit in Compose. Do not create a hidden production-env
  fallback or inject every secret payload property indiscriminately.
- If the repository is missing several parts, implement and verify the whole
  contract in one feature path. Do not stop after adding only a Compose file,
  database container, seed script, token endpoint, or frontend flag.

## Required evidence

- Run focused unit tests for secret loading and test-auth security boundaries.
- Run migration and seed tests against an empty disposable PostgreSQL database.
- Build the backend and frontend development images.
- Start the complete Compose stack and wait for health, then inspect container
  status and relevant logs without exposing secrets.
- Exercise the seeded private UI in a browser, confirm API persistence, use a
  second identity when ownership exists, and prove reset/reseed behavior.
- Exercise a production-mode configuration test proving test auth is rejected
  and missing/malformed Secrets Manager configuration fails before application
  startup.
- Report exact validated surfaces and any repository-specific limitation. Do not
  treat compilation, container creation, or direct database contents as proof of
  the user workflow.
