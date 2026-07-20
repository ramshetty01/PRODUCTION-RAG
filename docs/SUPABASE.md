# Supabase Metadata Store

Use Supabase Postgres for production metadata:

```env
RAG_METADATA_BACKEND=postgres
RAG_DATABASE_URL=postgresql://postgres:<password>@db.<project>.supabase.co:5432/postgres
```

Bootstrap tables by running the metadata store bootstrap from a one-off Render
shell or local machine with the same `RAG_DATABASE_URL`.

The schema includes `users`, `organizations`, `api_keys`, `documents`,
`ingestion_jobs`, `chat_sessions`, and `usage_events`. Uploaded document bytes
can stay in local storage or S3-compatible storage; Postgres stores metadata,
job state, chat/session references, and usage records.
