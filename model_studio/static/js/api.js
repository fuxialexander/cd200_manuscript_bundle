export class APIClient {
  async request(path, options = {}) {
    const response = await fetch(path, options);
    const text = await response.text();
    const payload = text ? JSON.parse(text) : {};
    if (!response.ok) {
      const error = new Error(payload.message || `Request failed (${response.status})`);
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  get(path) {
    return this.request(path);
  }

  post(path, payload) {
    return this.request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  async uploadCheckpoint(file) {
    const data = new FormData();
    data.append("checkpoint", file);
    return this.request("/api/uploads/checkpoint", {
      method: "POST",
      body: data,
    });
  }

  async uploadTable(file) {
    const data = new FormData();
    data.append("table", file);
    return this.request("/api/uploads/table", {
      method: "POST",
      body: data,
    });
  }
}
