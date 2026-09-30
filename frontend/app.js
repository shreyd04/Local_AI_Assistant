const API_BASE = "http://127.0.0.1:8000";

const state = {
  structured: false,
  busy: false,
};

const chat = document.getElementById("chat");
const composer = document.getElementById("composer");
const promptInput = document.getElementById("promptInput");
const sendButton = document.getElementById("sendButton");
const sendLabel = document.getElementById("sendLabel");
const sendIcon = document.getElementById("sendIcon");
const temperature = document.getElementById("temperature");
const temperatureValue = document.getElementById("temperatureValue");
const structuredToggle = document.getElementById("structuredToggle");
const clearChat = document.getElementById("clearChat");
const connectionStatus = document.getElementById("connectionStatus");
const errorBanner = document.getElementById("errorBanner");

const metricsEmpty = document.getElementById("metricsEmpty");
const metricsContent = document.getElementById("metricsContent");
const metricTtft = document.getElementById("metricTtft");
const metricLatency = document.getElementById("metricLatency");
const metricTokens = document.getElementById("metricTokens");
const metricOutputTokens = document.getElementById("metricOutputTokens");
const metricModel = document.getElementById("metricModel");
const metricTemperature = document.getElementById("metricTemperature");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdown(text) {
  let safe = escapeHtml(text);

  const codeBlocks = [];

  // Extract fenced code blocks first
  safe = safe.replace(/```([\s\S]*?)```/g, (_, code) => {
    const index = codeBlocks.push(code.trim()) - 1;
    return `@@CODEBLOCK${index}@@`;
  });

  const lines = safe.split("\n");
  const html = [];

  let listType = null; // "ol" or "ul"

  function closeList() {
    if (listType) {
      html.push(`</${listType}>`);
      listType = null;
    }
  }

  for (const rawLine of lines) {
    const line = rawLine.trim();

    // Blank line:
    // keep an existing list open because LLMs commonly
    // put blank lines between list items.
    if (!line) {
      if (!listType) {
        html.push("<div class=\"md-spacer\"></div>");
      }
      continue;
    }

    // Headings
    if (/^#{1,3}\s+/.test(line)) {
      closeList();

      const heading = line.replace(/^#{1,3}\s+/, "");
      html.push(`<h3>${heading}</h3>`);
      continue;
    }

    // Unordered list
    if (/^[-*]\s+/.test(line)) {
      if (listType !== "ul") {
        closeList();
        html.push("<ul>");
        listType = "ul";
      }

      const content = line.replace(/^[-*]\s+/, "");
      html.push(`<li>${content}</li>`);
      continue;
    }

    // Ordered list
    if (/^\d+\.\s+/.test(line)) {
      if (listType !== "ol") {
        closeList();
        html.push("<ol>");
        listType = "ol";
      }

      const content = line.replace(/^\d+\.\s+/, "");
      html.push(`<li>${content}</li>`);
      continue;
    }

    // Inline code
    let formatted = line.replace(
      /`([^`]+)`/g,
      "<code>$1</code>"
    );

    // Bold
    formatted = formatted.replace(
      /\*\*(.+?)\*\*/g,
      "<strong>$1</strong>"
    );

    formatted = formatted.replace(
      /__(.+?)__/g,
      "<strong>$1</strong>"
    );

    // Italic
    formatted = formatted.replace(
      /\*([^*\n]+)\*/g,
      "<em>$1</em>"
    );

    // Normal paragraph
    closeList();
    html.push(`<p>${formatted}</p>`);
  }

  closeList();

  let rendered = html.join("");

  // Restore code blocks
  codeBlocks.forEach((code, index) => {
    rendered = rendered.replace(
      `@@CODEBLOCK${index}@@`,
      `<pre><code>${code}</code></pre>`
    );
  });

  return rendered;
}

function addUserMessage(text) {
  const wrapper = document.createElement("div");
  wrapper.className = "message user";

  wrapper.innerHTML = `
    <div>
      <div class="message-bubble">${renderMarkdown(text)}</div>
      <div class="message-meta">YOU</div>
    </div>
    <div class="avatar">Y</div>
  `;

  chat.appendChild(wrapper);
  scrollToBottom();
}

function addTypingMessage() {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";
  wrapper.id = "typingMessage";

  wrapper.innerHTML = `
    <div class="avatar">AI</div>
    <div>
      <div class="message-bubble">
        <div class="typing">Generating<span class="typing-dots"></span></div>
      </div>
    </div>
  `;

  chat.appendChild(wrapper);
  scrollToBottom();
}

function removeTypingMessage() {
  document.getElementById("typingMessage")?.remove();
}

function addAssistantMessage(data) {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";

  if (state.structured && data.result) {
    const result = data.result;
    const points = Array.isArray(result.key_points) ? result.key_points : [];

    wrapper.innerHTML = `
      <div class="avatar">AI</div>
      <div>
        <div class="message-bubble">
          <div class="structured-card">
            <span class="structured-category">${escapeHtml(result.category)}</span>
            <div class="structured-answer">${renderMarkdown(result.answer)}</div>
            ${
              points.length
                ? `<ul class="key-points">${points.map(point => `<li>${escapeHtml(point)}</li>`).join("")}</ul>`
                : ""
            }
          </div>
        </div>
        <div class="message-meta">LOCAL • STRUCTURED • ${escapeHtml(data.attempts ?? 1)} ATTEMPT(S)</div>
      </div>
    `;
  } else {
    wrapper.innerHTML = `
      <div class="avatar">AI</div>
      <div>
        <div class="message-bubble">${renderMarkdown(data.response ?? "")}</div>
        <div class="message-meta">LOCAL • ${escapeHtml(data.model ?? "mistral:7b-instruct-v0.3-q5_K_M")}</div>
      </div>
    `;
  }

  chat.appendChild(wrapper);
  scrollToBottom();
}

function updateMetrics(data) {
  metricsEmpty.classList.add("hidden");
  metricsContent.classList.remove("hidden");

  metricTtft.textContent =
    data.ttft_ms != null ? `${Number(data.ttft_ms).toFixed(1)} ms` : "—";

  metricLatency.textContent =
    data.total_latency_ms != null
      ? `${(Number(data.total_latency_ms) / 1000).toFixed(2)} s`
      : "—";

  metricTokens.textContent =
    data.tokens_per_second != null
      ? `${Number(data.tokens_per_second).toFixed(2)} t/s`
      : "—";

  metricOutputTokens.textContent =
    data.output_tokens != null
      ? `${data.output_tokens}`
      : "—";

  metricModel.textContent =
    data.model ?? data.result?.model ?? "mistral:7b-instruct-v0.3-q5_K_M";

  metricTemperature.textContent = Number(temperature.value).toFixed(1);
}

function setBusy(busy) {
  state.busy = busy;
  sendButton.disabled = busy;

  if (busy) {
    sendLabel.textContent = "Running";
    sendIcon.textContent = "•";
  } else {
    sendLabel.textContent = "Send";
    sendIcon.textContent = "↑";
  }
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.remove("hidden");
}

function clearError() {
  errorBanner.textContent = "";
  errorBanner.classList.add("hidden");
}

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

async function checkBackend() {
  try {
    const response = await fetch(`${API_BASE}/health`, {
      method: "GET",
    });

    if (!response.ok) {
      throw new Error("Backend health check failed");
    }

    connectionStatus.className = "status-pill status-online";
    connectionStatus.innerHTML =
      '<span class="status-dot"></span> Backend online';
  } catch {
    connectionStatus.className = "status-pill status-offline";
    connectionStatus.innerHTML =
      '<span class="status-dot"></span> Backend offline';
  }
}

async function sendMessage() {
  if (state.busy) return;

  const prompt = promptInput.value.trim();
  if (!prompt) return;

  clearError();

  const temp = Number(temperature.value);

  addUserMessage(prompt);
  promptInput.value = "";
  autoResize();

  setBusy(true);
  addTypingMessage();

  try {
    const endpoint = state.structured
      ? `${API_BASE}/structured-chat`
      : `${API_BASE}/chat`;

    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt,
        temperature: temp,
      }),
    });

    const raw = await response.text();

    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      throw new Error(`Backend returned invalid JSON (${response.status}).`);
    }

    if (!response.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        `Request failed with HTTP ${response.status}.`
      );
    }

    removeTypingMessage();
    addAssistantMessage(data);

    if (!state.structured) {
      updateMetrics(data);
    } else {
      // Structured endpoint returns structured content but not timing fields.
      // Keep the previous metric panel and update the mode temperature.
      metricTemperature.textContent = temp.toFixed(1);
    }
  } catch (error) {
    removeTypingMessage();

    const message =
      error?.message ||
      "Unable to reach the local backend.";

    showError(
      `${message} Make sure FastAPI is running on http://127.0.0.1:8000 and Ollama is running.`
    );
  } finally {
    setBusy(false);
    promptInput.focus();
  }
}

function autoResize() {
  promptInput.style.height = "auto";
  promptInput.style.height =
    `${Math.min(promptInput.scrollHeight, 180)}px`;
}

temperature.addEventListener("input", () => {
  temperatureValue.textContent = Number(temperature.value).toFixed(1);
});

structuredToggle.addEventListener("click", () => {
  state.structured = !state.structured;

  structuredToggle.classList.toggle("active", state.structured);
  structuredToggle.setAttribute(
    "aria-pressed",
    String(state.structured)
  );
});

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage();
});

promptInput.addEventListener("input", autoResize);

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

clearChat.addEventListener("click", () => {
  chat.innerHTML = `
    <div class="welcome-card">
      <div class="welcome-icon">✦</div>
      <h2>Private, local, measurable.</h2>
      <p>
        Ask a question and inspect the actual inference metrics from your local model.
        Nothing needs to leave your machine.
      </p>
      <div class="suggestion-grid">
        <button class="suggestion">Explain why database indexes improve query performance.</button>
        <button class="suggestion">Compare a process and a thread in operating systems.</button>
        <button class="suggestion">Give three principles of REST API architecture.</button>
        <button class="suggestion">Explain overfitting and two ways to reduce it.</button>
      </div>
    </div>
  `;

  wireSuggestions();
});

function wireSuggestions() {
  document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
      promptInput.value = button.textContent.trim();
      autoResize();
      promptInput.focus();
    });
  });
}

wireSuggestions();
checkBackend();
promptInput.focus();
