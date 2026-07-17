const credential = document.querySelector("#adminCredential");
const refresh = document.querySelector("#adminRefresh");
const statusText = document.querySelector("#adminStatus");
const health = document.querySelector("#adminHealth");
const index = document.querySelector("#adminIndex");
const failures = document.querySelector("#adminFailures");
const documents = document.querySelector("#adminDocuments");
const auditEvents = document.querySelector("#auditEvents");
const auditCsv = document.querySelector("#auditCsv");

function headers() {
  return credential.value.trim() ? {"X-API-Key": credential.value.trim()} : {};
}

function row(document) {
  return `
    <div class="admin-row">
      <div>
        <strong>${document.filename || document.document_id}</strong>
        <span>${document.document_id} · ${document.status} · ${document.chunk_count} chunks</span>
      </div>
      <div class="admin-actions">
        <button type="button" data-action="reindex" data-id="${document.document_id}">Reindex</button>
        <button type="button" data-action="delete" data-id="${document.document_id}">Delete</button>
      </div>
    </div>
  `;
}

async function adminRequest(path, options = {}) {
  const response = await fetch(path, {...options, headers: {...headers(), ...(options.headers || {})}});
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `HTTP ${response.status}`);
  }
  return payload;
}

async function load() {
  statusText.textContent = "Loading";
  const payload = await adminRequest("/admin/status");
  const audit = await adminRequest("/audit");
  health.textContent = `API ${payload.health.api}`;
  index.textContent = `${payload.index.document_count} docs`;
  failures.textContent = `${payload.failed_jobs.length} failures`;
  documents.innerHTML = payload.documents.length ? payload.documents.map(row).join("") : "<p>No documents indexed.</p>";
  auditCsv.href = "/audit?format=csv";
  auditEvents.innerHTML = audit.events.length
    ? audit.events.map((event) => `<div class="admin-row"><div><strong>${event.user}</strong><span>${event.query}</span></div><span>${event.latency_ms || 0} ms</span></div>`).join("")
    : "<p>No audit events.</p>";
  statusText.textContent = "Loaded";
}

documents.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const action = button.dataset.action;
  const id = button.dataset.id;
  await adminRequest(`/documents/${id}${action === "reindex" ? "/reindex" : ""}`, {method: action === "delete" ? "DELETE" : "POST"});
  await load();
});

refresh.addEventListener("click", () => load().catch((error) => {
  statusText.textContent = error.message;
}));
