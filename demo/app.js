const form = document.querySelector("#queryForm");
const uploadForm = document.querySelector("#uploadForm");
const queryInput = document.querySelector("#queryInput");
const retrievalMode = document.querySelector("#retrievalMode");
const topK = document.querySelector("#topK");
const filterDocument = document.querySelector("#filterDocument");
const filterType = document.querySelector("#filterType");
const filterRole = document.querySelector("#filterRole");
const activeFilters = document.querySelector("#activeFilters");
const authType = document.querySelector("#authType");
const credential = document.querySelector("#credential");
const authBadge = document.querySelector("#authBadge");
const askButton = document.querySelector("#askButton");
const uploadInput = document.querySelector("#documentUpload");
const uploadButton = document.querySelector("#uploadButton");
const uploadStatus = document.querySelector("#uploadStatus");
const apiStatus = document.querySelector("#apiStatus");
const answerText = document.querySelector("#answerText");
const requestId = document.querySelector("#requestId");
const citations = document.querySelector("#citations");
const citationCount = document.querySelector("#citationCount");
const cacheBadge = document.querySelector("#cacheBadge");
const traceMode = document.querySelector("#traceMode");
const traceChunks = document.querySelector("#traceChunks");
const traceSubject = document.querySelector("#traceSubject");
const traceCost = document.querySelector("#traceCost");
const evalStatus = document.querySelector("#evalStatus");
const evalFaithfulness = document.querySelector("#evalFaithfulness");
const evalCitations = document.querySelector("#evalCitations");
const evalRefusals = document.querySelector("#evalRefusals");
const evalDataset = document.querySelector("#evalDataset");
const metricRequests = document.querySelector("#metricRequests");
const metricLatency = document.querySelector("#metricLatency");
const metricStatus = document.querySelector("#metricStatus");
const workspaceId = localStorage.getItem("rag_workspace_id") || crypto.randomUUID();
const sessionId = localStorage.getItem("rag_session_id") || crypto.randomUUID();
const savedAuthType = localStorage.getItem("rag_auth_type") || "none";
const savedCredential = sessionStorage.getItem("rag_credential") || "";

localStorage.setItem("rag_workspace_id", workspaceId);
localStorage.setItem("rag_session_id", sessionId);
authType.value = savedAuthType;
credential.value = savedCredential;

const scenarios = {
  vendor: {
    query: "What evidence is required before vendor onboarding?",
    mode: "hybrid",
  },
  payroll: {
    query: "Can I retrieve protected payroll data?",
    mode: "semantic",
  },
  incident: {
    query: "How fast must production authentication incidents be reviewed?",
    mode: "semantic",
  },
};

function text(value) {
  return value === null || value === undefined || value === "" ? "-" : String(value);
}

function setStatus(label, variant = "") {
  apiStatus.textContent = label;
  apiStatus.className = `metric-value ${variant}`.trim();
}

function saveAuthState() {
  localStorage.setItem("rag_auth_type", authType.value);
  sessionStorage.setItem("rag_credential", credential.value.trim());
}

function authHeaders(base = {}) {
  const headers = {...base};
  const authValue = credential.value.trim();
  if (authType.value === "api_key" && authValue) {
    headers["X-API-Key"] = authValue;
  }
  if (authType.value === "bearer" && authValue) {
    headers.Authorization = `Bearer ${authValue}`;
  }
  return headers;
}

function authErrorMessage(status, payload) {
  if (status === 401) return "Sign in with a valid API key or bearer token.";
  if (status === 403) return "Your current identity is not allowed to perform this action.";
  return typeof payload.detail === "string" ? payload.detail : payload.detail?.message;
}

function renderCitations(items) {
  citations.innerHTML = "";
  citationCount.textContent = String(items.length);

  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "citation";
    empty.innerHTML = "<strong>No citations</strong><p>No retrieved evidence supported this response.</p>";
    citations.append(empty);
    return;
  }

  for (const item of items) {
    const citation = document.createElement("article");
    citation.className = "citation";
    citation.tabIndex = 0;
    citation.innerHTML = `
      <button type="button" class="citation-toggle" aria-expanded="false">
        <strong>${text(item.label || item.source)} </strong>
        <span>${text(item.snippet || item.quote)}</span>
      </button>
      <div class="citation-detail" hidden>
        <dl>
          <div><dt>Source</dt><dd>${text(item.source)}</dd></div>
          <div><dt>Page</dt><dd>${text(item.page)}</dd></div>
        </dl>
        <p>${text(item.context || item.quote)}</p>
        ${item.source_url ? `<a href="${item.source_url}" target="_blank" rel="noreferrer">Open source</a>` : ""}
      </div>
    `;
    const toggle = citation.querySelector(".citation-toggle");
    const detail = citation.querySelector(".citation-detail");
    toggle.addEventListener("click", () => {
      const expanded = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!expanded));
      detail.hidden = expanded;
    });
    citations.append(citation);
  }
}

function renderResult(payload) {
  answerText.textContent = payload.answer || "The answer is not available in the retrieved context.";
  requestId.textContent = payload.request_id || "No request";
  cacheBadge.textContent = payload.cached ? "Cached" : "Fresh";
  cacheBadge.className = `pill ${payload.cached ? "" : "muted"}`.trim();
  renderCitations(payload.citations || []);

  const retrieval = payload.retrieval || {};
  const trace = payload.trace || {};
  traceMode.textContent = text(retrieval.mode);
  traceChunks.textContent = text((retrieval.chunk_ids || []).join(", "));
  traceSubject.textContent = text(retrieval.auth_subject);
  authBadge.textContent = `${text(retrieval.auth_subject)} · ${text((retrieval.auth_roles || []).join("|"))}`;
  traceCost.textContent = text(trace.token_usage?.estimated_cost);
}

function renderError(error) {
  answerText.textContent = error;
  requestId.textContent = "Request failed";
  requestId.className = "pill error";
  renderCitations([]);
  traceMode.textContent = "-";
  traceChunks.textContent = "-";
  traceSubject.textContent = "-";
  traceCost.textContent = "-";
  cacheBadge.textContent = "Error";
  cacheBadge.className = "pill error";
}

function selectedFilters() {
  const filters = {};
  const documentId = filterDocument.value.trim();
  const parser = filterType.value.trim();
  const role = filterRole.value.trim();
  if (documentId) {
    filters.document_id = documentId;
  }
  if (parser) {
    filters.parser = parser;
  }
  if (role) {
    filters.access_roles = role;
  }
  return filters;
}

function renderActiveFilters(filters) {
  const entries = Object.entries(filters).map(([key, value]) => `${key}:${value}`);
  activeFilters.textContent = entries.length ? entries.join(" · ") : "No filters";
  activeFilters.className = `pill ${entries.length ? "" : "muted"}`.trim();
}

function queryBody() {
  const filters = selectedFilters();
  renderActiveFilters(filters);
  const body = {
    query: queryInput.value.trim(),
    workspace_id: workspaceId,
    session_id: sessionId,
    retrieval_mode: retrievalMode.value,
    top_k: Number(topK.value),
  };
  if (Object.keys(filters).length) {
    body.metadata_filters = filters;
  }
  return {
    ...body,
  };
}

function parseSseMessage(message) {
  const event = {type: "message", data: ""};
  for (const line of message.split("\n")) {
    if (line.startsWith("event:")) {
      event.type = line.slice("event:".length).trim();
    }
    if (line.startsWith("data:")) {
      event.data += line.slice("data:".length).trim();
    }
  }
  return event;
}

async function readStreamingAnswer(response) {
  if (!response.body) {
    throw new Error("Streaming is unavailable");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalPayload = null;
  answerText.textContent = "";
  renderCitations([]);

  while (true) {
    const {value, done} = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, {stream: true});
    const messages = buffer.split("\n\n");
    buffer = messages.pop() || "";

    for (const message of messages) {
      if (!message.trim()) {
        continue;
      }
      const event = parseSseMessage(message);
      const data = event.data ? JSON.parse(event.data) : {};
      if (event.type === "token") {
        answerText.textContent += data.text || "";
      }
      if (event.type === "complete") {
        finalPayload = data;
      }
      if (event.type === "error") {
        throw new Error(data.message || `HTTP ${data.status_code || 500}`);
      }
    }
  }

  if (!finalPayload) {
    throw new Error("Streaming response ended without completion");
  }
  renderResult(finalPayload);
}

async function postStandardQuery(headers, body) {
  const response = await fetch("/query", {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) {
    const message = authErrorMessage(response.status, payload);
    throw new Error(message || `HTTP ${response.status}`);
  }
  renderResult(payload);
}

function parseMetric(body, metricName) {
  const match = body.match(new RegExp(`^${metricName}\\\\s+([^\\\\n]+)$`, "m"));
  return match ? match[1] : "0";
}

function percent(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${Math.round(numeric * 100)}%` : "-";
}

async function refreshEvaluation() {
  try {
    const response = await fetch("/evaluation");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }
    evalStatus.textContent = payload.quality_gate?.passed ? "Passing" : "Failing";
    evalStatus.className = `pill ${payload.quality_gate?.passed ? "" : "error"}`.trim();
    evalFaithfulness.textContent = percent(payload.metrics?.faithfulness);
    evalCitations.textContent = percent(payload.metrics?.citation_coverage);
    evalRefusals.textContent = percent(payload.metrics?.refusal_accuracy);
    evalDataset.textContent = `${payload.dataset?.verified_cases || 0}/${payload.dataset?.total_cases || 0}`;
  } catch {
    evalStatus.textContent = "Offline";
    evalStatus.className = "pill error";
  }
}

async function refreshMetrics() {
  try {
    const response = await fetch("/metrics");
    const body = await response.text();
    metricRequests.textContent = parseMetric(body, "rag_api_requests_total");
    const latency = Number(parseMetric(body, "rag_api_request_latency_ms_total"));
    metricLatency.textContent = Number.isFinite(latency) ? `${Math.round(latency)} ms` : "0 ms";
    metricStatus.textContent = response.ok ? "Ready" : "Error";
  } catch {
    metricStatus.textContent = "Offline";
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    setStatus(response.ok ? "Online" : "Degraded", response.ok ? "" : "error");
  } catch {
    setStatus("Offline", "error");
  }
}

async function refreshDocuments() {
  const response = await fetch(`/documents?workspace_id=${encodeURIComponent(workspaceId)}`);
  return response.ok ? response.json() : {documents: []};
}

async function askQuestion(event) {
  event.preventDefault();
  askButton.disabled = true;
  askButton.lastChild.textContent = " Asking";
  requestId.className = "pill muted";
  setStatus("Running");

  const headers = authHeaders({"Content-Type": "application/json"});

  try {
    const body = queryBody();
    const response = await fetch("/query/stream", {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      await postStandardQuery(headers, body);
    } else {
      await readStreamingAnswer(response);
    }

    setStatus("Online");
  } catch (error) {
    renderError(error.message);
    setStatus("Error", "error");
  } finally {
    askButton.disabled = false;
    askButton.lastChild.textContent = " Ask";
    refreshMetrics();
  }
}

function applyAuthPreset(preset) {
  authType.value = "api_key";
  credential.value = preset === "admin" ? "admin-key" : "public-key";
  saveAuthState();
  authBadge.textContent = preset === "admin" ? "api-key:admin- · admin|public" : "api-key:public · public";
}

async function uploadDocument(event) {
  event.preventDefault();
  const file = uploadInput.files?.[0];
  if (!file) {
    uploadStatus.textContent = "Choose a file";
    return;
  }

  uploadButton.disabled = true;
  uploadButton.textContent = "Indexing";
  uploadStatus.textContent = file.name;
  setStatus("Indexing");

  const body = new FormData();
  body.append("file", file);
  body.append("workspace_id", workspaceId);
  body.append("access_roles", authType.value === "api_key" && credential.value.trim() === "admin-key" ? "admin" : "public");

  try {
    const response = await fetch("/upload", {
      method: "POST",
      headers: authHeaders(),
      body,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(authErrorMessage(response.status, payload) || `HTTP ${response.status}`);
    }
    uploadStatus.textContent = `${payload.chunks_created} chunks indexed`;
    await refreshDocuments();
    metricStatus.textContent = "Indexed";
    setStatus("Online");
  } catch (error) {
    uploadStatus.textContent = error.message;
    metricStatus.textContent = "Upload error";
    setStatus("Error", "error");
  } finally {
    uploadButton.disabled = false;
    uploadButton.textContent = "Index";
    refreshMetrics();
  }
}

document.querySelectorAll("[data-scenario]").forEach((button) => {
  button.addEventListener("click", () => {
    const scenario = scenarios[button.dataset.scenario];
    queryInput.value = scenario.query;
    retrievalMode.value = scenario.mode;
    queryInput.focus();
  });
});

document.querySelectorAll("[data-auth-preset]").forEach((button) => {
  button.addEventListener("click", () => applyAuthPreset(button.dataset.authPreset));
});
authType.addEventListener("change", saveAuthState);
credential.addEventListener("input", saveAuthState);

filterDocument.addEventListener("input", () => renderActiveFilters(selectedFilters()));
filterType.addEventListener("change", () => renderActiveFilters(selectedFilters()));
filterRole.addEventListener("input", () => renderActiveFilters(selectedFilters()));
form.addEventListener("submit", askQuestion);
uploadForm.addEventListener("submit", uploadDocument);
renderActiveFilters(selectedFilters());
checkHealth();
refreshMetrics();
refreshEvaluation();
