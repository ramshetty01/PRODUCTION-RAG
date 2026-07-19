# Secret Management

`.env` is local-only and ignored by Git. Staging and production should inject
secrets from the deployment platform or a secret manager.

Required production secrets:

- `RAG_API_KEYS` when `RAG_AUTH_MODE=api_key`.
- `RAG_JWT_SECRET` when `RAG_AUTH_MODE=jwt`.
- `RAG_LLM_API_KEY` for hosted LLM providers such as OpenRouter.
- `RAG_QDRANT_API_KEY` when Qdrant requires an API key.
- `RAG_REDIS_URL` when Redis requires credentials.
- `RAG_OBSERVABILITY_EXPORT_API_KEY` when using managed observability export.

Preferred setup:

1. Store secrets in Render environment variables, Kubernetes Secret, AWS Secrets
   Manager, GCP Secret Manager, Doppler, Vault, or 1Password.
2. Inject them as environment variables at runtime.
3. Keep `deploy/*.env.example` as non-secret references only.
4. Rotate keys after incidents, employee changes, and provider migrations.

Mounted secret file setup is also supported. Set `RAG_SECRETS_FILE` to an env or
JSON file mounted by your secret manager:

```bash
RAG_SECRETS_FILE=/run/secrets/production-rag.env
```

The app loads local `.env`, then the mounted secrets file, then real environment
variables. Real environment variables win.
