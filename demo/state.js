export function createInitialState() {
  return {
    auth: {
      type: "none",
      subject: "dev-public",
      roles: ["public"],
    },
    upload: {
      status: "idle",
      message: "No upload",
      jobId: null,
      chunks: 0,
    },
    indexing: {
      ready: false,
      status: "empty",
      message: "Upload and index a corpus before asking.",
    },
    chat: {
      busy: false,
      error: null,
      lastPayload: null,
    },
    citations: [],
    errors: [],
  };
}

export const appState = createInitialState();

export function mergeState(patch) {
  for (const [key, value] of Object.entries(patch)) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      appState[key] = {...appState[key], ...value};
    } else {
      appState[key] = value;
    }
  }
  return appState;
}
