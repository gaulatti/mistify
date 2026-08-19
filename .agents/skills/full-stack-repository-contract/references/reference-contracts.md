# Proven reference contracts

Use these repositories as evidence for architecture and safety invariants. Read
only the files relevant to the target framework and revalidate them because the
reference repositories can evolve.

## Docker Compose and PostgreSQL

### Pompeii

- `~/source/gaulatti/pompeii/compose.yml`
- Complete local `postgres`, `backend`, and `frontend` topology.
- PostgreSQL named volume and `pg_isready` health check.
- Backend waits for healthy PostgreSQL, runs migrations, then starts its watcher.
- Backend health gates frontend startup.
- Test authentication is Compose-owned and production is deployed separately.

### Angelina

- `~/source/gaulatti/angelina/compose.yaml`
- Development build targets and source mounts for backend/frontend.
- Dockerized PostgreSQL with named volume and readiness check.
- Migrations run before the development backend starts.
- Frontend Compose configuration selects the local test-auth path.

### Centinela

- `~/source/gaulatti/centinela/compose.yaml`
- Strong readiness chain: PostGIS health gates migrations/backend, and backend
  health gates dependent services and the Nginx-served frontend.
- Useful reference when PostgreSQL extensions or a production-built frontend
  image are required locally.

Do not copy exact ports, images, credentials, or commands. Match the target's
framework and persistence contract.

## Secrets Manager bootstrap

### Pompeii

- `backend/src/config/secrets-loader.ts`
- `backend/src/config/secrets-loader.spec.ts`
- `backend/src/main.ts`
- `backend/src/run-production.ts`
- `backend/src/run-migrations.ts`
- `backend/src/run-bootstrap-admin.ts`

This is the preferred contract: minimal `SECRET_ARN`/application-key bootstrap,
selection of an application-scoped object, explicit supported-key allowlist,
production precedence, required-key validation, and shared loading before every
entry point consumes configuration.

### Angelina

- `backend/src/config/secrets-loader.ts`
- `backend/src/run-production.ts`

Use its load-before-migrate-and-start sequence as a reference. Improve on its
broad key injection by using Pompeii's explicit allowlist and validation model.

Local Compose must bypass AWS by supplying explicit non-production values. In
production, never fall back from a configured secret to standalone sensitive
environment values.

## Offline authenticated UI

### Pompeii

- `backend/src/authentication/test-auth.ts`
- `backend/src/authentication/test-auth.controller.ts`
- `frontend/app/auth/session.ts`

Reference for a short-lived local JWT, distinct issuer/audience, minimum local
signing-key requirements, automatic frontend session bootstrap, hidden disabled
endpoint, and explicit production refusal.

### Celesti/Mattone

- `backend/src/auth/test-auth.service.ts`
- `backend/src/auth/test-auth.controller.ts`
- frontend test-auth bootstrap and visible `TEST AUTH` indicator

Reference for deterministic named identities, seeded user-owned domain data,
normal ownership enforcement, and reset/reseed behavior. Prefer this shape when
parallel agents need isolated state.

### Monitor and Ariston

- Monitor: guarded `GET /__test/session` plus automatic frontend session.
- Ariston: guarded `/test-auth/login` signs in a seeded local administrator and
  redirects through the normal CMS session path.

Choose the route/session shape that matches the target framework. Do not use a
static production token, mock the protected API, disable guards, or require a
real Cognito request.

## Minimum regression matrix

| Boundary | Required proof |
| --- | --- |
| Compose | Fresh stack reaches healthy database, backend, and frontend |
| Migrations | Empty database reaches current schema before backend readiness |
| Seed | Repeated execution is safe and primary UI states exist |
| Secrets | Allowlist, production precedence, malformed/missing failure |
| Entry points | API, migrations, workers, and jobs load secrets before use |
| Test auth disabled | Endpoint is absent or 404 |
| Production safety | Production plus test auth refuses startup |
| Token validation | Signature, expiry, issuer, and audience are enforced |
| Authorization | Real roles, permissions, and ownership remain active |
| Isolation/reset | Named identities do not leak state; reset is deterministic |
| Browser | Agent completes a primary persisted workflow without Cognito |
