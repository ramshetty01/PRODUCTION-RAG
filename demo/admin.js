const credential = document.querySelector("#adminCredential");
const refresh = document.querySelector("#adminRefresh");
const statusText = document.querySelector("#adminStatus");
const health = document.querySelector("#adminHealth");
const index = document.querySelector("#adminIndex");
const failures = document.querySelector("#adminFailures");
const documents = document.querySelector("#adminDocuments");
const auditEvents = document.querySelector("#auditEvents");
const auditCsv = document.querySelector("#auditCsv");
const feedbackEvents = document.querySelector("#feedbackEvents");
const feedbackCsv = document.querySelector("#feedbackCsv");
const dashboard = document.querySelector("#observabilityDashboard");
const dashboardWindow = document.querySelector("#dashboardWindow");

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
  const observability = await adminRequest("/observability/dashboard?window_minutes=60");
  const audit = await adminRequest("/audit");
  const feedback = await adminRequest("/feedback/events");
  health.textContent = `API ${payload.health.api}`;
  index.textContent = `${payload.index.document_count} docs`;
  failures.textContent = `${payload.failed_jobs.length} failures`;
  documents.innerHTML = payload.documents.length ? payload.documents.map(row).join("") : "<p>No documents indexed.</p>";
  dashboardWindow.textContent = `${observability.window.minutes} min`;
  dashboard.innerHTML = [
    ["Requests", observability.metrics.request_count],
    ["Avg latency", `${observability.request_latency.avg_ms} ms`],
    ["P95 latency", `${observability.request_latency.p95_ms} ms`],
    ["Retrieval chunks", observability.retrieval.total_chunks],
    ["Index health", observability.index_health.status],
    ["Model errors", observability.model.errors],
    ["Ingestion failures", observability.ingestion.failed_jobs + observability.ingestion.failed_documents],
  ].map(([label, value]) => `<div class="admin-row"><strong>${label}</strong><span>${value}</span></div>`).join("");
  auditCsv.href = "/audit?format=csv";
  auditEvents.innerHTML = audit.events.length
    ? audit.events.map((event) => `<div class="admin-row"><div><strong>${event.user}</strong><span>${event.query}</span></div><span>${event.latency_ms || 0} ms</span></div>`).join("")
    : "<p>No audit events.</p>";
  feedbackCsv.href = "/feedback/events?format=csv";
  feedbackEvents.innerHTML = feedback.events.length
    ? feedback.events.map((event) => `<div class="admin-row"><div><strong>${event.helpful ? "Helpful" : "Unhelpful"}</strong><span>${event.query}</span></div><span>${event.request_id}</span></div>`).join("")
    : "<p>No feedback yet.</p>";
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
