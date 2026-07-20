# Supabase Metadata Store

Use Supabase Postgres for production metadata:

```env
RAG_METADATA_BACKEND=postgres
RAG_DATABASE_URL=postgresql://postgres:<password>@db.<project>.supabase.co:5432/postgres
RAG_AUTH_MODE=supabase
RAG_SUPABASE_URL=https://<project>.supabase.co
RAG_SUPABASE_JWT_SECRET=<supabase-jwt-secret>
```

Bootstrap tables by running the metadata store bootstrap from a one-off Render
shell or local machine with the same `RAG_DATABASE_URL`.

The schema includes `users`, `organizations`, `api_keys`, `documents`,
`ingestion_jobs`, `chat_sessions`, and `usage_events`. Uploaded document bytes
can stay in local storage or S3-compatible storage; Postgres stores metadata,
job state, chat/session references, and usage records.

The API accepts Supabase Auth access tokens as `Authorization: Bearer <token>`.
On first request it upserts the user and organization from the JWT claims.
Machine clients can use `X-API-Key`; store `sha256(key)` in `api_keys.key_hash`,
put the workspace in `organization_id`, and put roles such as `public` or
`workspace-admin` in `scopes`.
