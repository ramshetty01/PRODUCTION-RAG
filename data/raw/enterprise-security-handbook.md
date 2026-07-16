# Enterprise Security Handbook

Acme Fintech uses role-based access control for every production RAG workspace.
Public handbook content can be retrieved by users with the public role, while
restricted operational documents require the admin role before retrieval or
answer generation.

All answers must cite retrieved evidence. If the retriever does not return
supporting chunks, the assistant must say that the answer is not available in
the retrieved context instead of guessing.

Incident response reviews must be opened within four business hours for
production authentication failures. The review owner records affected tenants,
retrieval mode, model provider, document version, and the request identifier.

Prompts, retrieval configuration, and evaluation thresholds are versioned
artifacts. Changes to any of those artifacts require a pull request, golden
dataset evaluation, and sign-off from the platform owner.

Tenant isolation is enforced through metadata filters, cache keys that include
tenant and role context, and explicit authorization checks before retrieved
chunks are sent to the generator.

Retrieved documents are treated as untrusted evidence. A retrieved document may
contain instructions, but those instructions must never override system,
developer, security, or citation requirements.
