import type {
  PlanningRequest,
  StreamResult,
  FinalTravelPackage,
  HumanInterveneResult,
  RejectedResult,
  NodeState,
  SSEEventType,
} from "./types";

// ---- Constants ----
const STEP_NAMES = [
  "shangshu_preflight", "zhongshu_itinerary", "menxia_review",
  "shangshu_review_gate", "shangshu_dispatch_liubu",
  "liubu_weather", "liubu_budget", "liubu_accommodation",
  "liubu_flight_transport", "liubu_calendar", "shangshu_assemble",
] as const;

// ---- DOM helpers ----
function $<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

// ---- State ----
let currentRequestId: string | null = null;
let sseEventType: SSEEventType | "" = "";

// ---- DOM refs ----
const workflowSection = $<HTMLElement>("workflow-section");
const resultSection = $<HTMLElement>("result-section");
const interveneSection = $<HTMLElement>("intervene-section");
const rejectedSection = $<HTMLElement>("rejected-section");
const logContent = $<HTMLDivElement>("log-content");
const resultContent = $<HTMLDivElement>("result-content");
const planForm = $<HTMLFormElement>("plan-form");
const interveneForm = $<HTMLFormElement>("intervene-form");
const submitBtn = $<HTMLButtonElement>("submit-btn");

// ---- Set default dates ----
(function setDefaultDates(): void {
  const today = new Date();
  const start = new Date(today);
  start.setDate(today.getDate() + 14);
  const end = new Date(start);
  end.setDate(start.getDate() + 3);
  $<HTMLInputElement>("start_date").value = fmtDate(start);
  $<HTMLInputElement>("end_date").value = fmtDate(end);
})();

function fmtDate(d: Date): string {
  return d.toISOString().split("T")[0];
}

// ---- Util ----
function esc(s: unknown): string {
  if (s == null) return "";
  const d = document.createElement("div");
  d.textContent = String(s);
  return d.innerHTML;
}

function splitTags(s: string): string[] {
  return s.split(/[,，]/).map(t => t.trim()).filter(Boolean);
}

function show(el: HTMLElement): void { el.style.display = "block"; }
function hide(el: HTMLElement): void { el.style.display = "none"; }

// ---- Node state ----
function resetNodes(): void {
  document.querySelectorAll<HTMLElement>(".node").forEach(n => {
    n.classList.remove("node-active", "node-done", "node-error");
  });
}

function setNodeState(step: string, state: NodeState): void {
  const node = document.querySelector<HTMLElement>(`.node[data-step="${step}"]`);
  if (!node) return;
  node.classList.remove("node-active", "node-done", "node-error");
  node.classList.add(`node-${state}`);
}

function updateNodeFromLog(line: string): void {
  const m = line.match(/\[\d{2}:\d{2}:\d{2}\]\s+\[(\w+)\]\s+(\w+)/);
  if (!m) return;
  const phase = m[1].toLowerCase();
  const step = m[2];

  if (step === "shangshu_dispatch_liubu" || (STEP_NAMES as readonly string[]).includes(step)) {
    if (phase === "start") setNodeState(step, "active");
    else if (phase === "done") setNodeState(step, "done");
    else if (phase === "error") setNodeState(step, "error");
  }
}

// ---- Log ----
function appendLog(text: string, phase: string): void {
  const div = document.createElement("div");
  div.className = "log-line" + (phase ? ` phase-${phase}` : "");
  div.textContent = text;
  logContent.appendChild(div);
  logContent.scrollTop = logContent.scrollHeight;
}

function parsePhase(line: string): string {
  const m = line.match(/\[([A-Z_]+)\]/);
  if (!m) return "";
  const p = m[1].toLowerCase();
  if (p === "start" || p === "done" || p === "error") return p;
  return "";
}

// ---- Generic SSE Stream ----
function startSSEStream(url: string, body: unknown): void {
  resetNodes();
  logContent.innerHTML = "";
  resultContent.innerHTML = "";
  hide(resultSection);
  hide(interveneSection);
  hide(rejectedSection);
  show(workflowSection);
  submitBtn.disabled = true;
  submitBtn.textContent = "规划中...";

  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
    .then(response => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      function pump(): Promise<void> {
        return reader.read().then(({ done, value }) => {
          if (done) { onStreamEnd(); return; }
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop()!;
          for (const line of lines) {
            processSSELine(line);
          }
          return pump();
        });
      }
      return pump();
    })
    .catch(err => {
      appendLog(`ERROR: ${(err as Error).message}`, "error");
      onStreamEnd();
    });
}

function processSSELine(line: string): void {
  if (line.startsWith("event: ")) {
    sseEventType = line.substring(7).trim() as SSEEventType;
    return;
  }
  if (line.startsWith("data: ")) {
    handleSSEEvent(sseEventType, line.substring(6));
    sseEventType = "";
  }
}

function handleSSEEvent(type: SSEEventType | "", data: string): void {
  if (type === "progress") {
    try {
      const obj = JSON.parse(data) as { line?: string };
      const line = obj.line || data;
      appendLog(line, parsePhase(line));
      updateNodeFromLog(line);
    } catch {
      appendLog(data, "");
    }
  } else if (type === "result") {
    try {
      const result = JSON.parse(data) as StreamResult;
      renderResult(result);
    } catch (e) {
      appendLog(`Failed to parse result: ${(e as Error).message}`, "error");
    }
  } else if (type === "error") {
    try {
      const err = JSON.parse(data) as { error?: string };
      appendLog(`ERROR: ${err.error || data}`, "error");
    } catch {
      appendLog(`ERROR: ${data}`, "error");
    }
  } else if (type === "done") {
    setNodeState("done", "done");
  }
}

function onStreamEnd(): void {
  submitBtn.disabled = false;
  submitBtn.textContent = "\u{1F680} 开始规划";
}

// ---- Build request ----
function buildRequestBody(): PlanningRequest {
  const ts = Date.now();
  currentRequestId = `web_${ts}`;
  const budgetVal = $<HTMLInputElement>("total_budget").value;

  return {
    request_id: currentRequestId,
    user_message: $<HTMLTextAreaElement>("user_message").value,
    profile: {
      origin_city: $<HTMLInputElement>("origin_city").value,
      destination_preferences: [$<HTMLInputElement>("destination").value],
      start_date: $<HTMLInputElement>("start_date").value,
      end_date: $<HTMLInputElement>("end_date").value,
      adults: parseInt($<HTMLInputElement>("adults").value) || 1,
      children: parseInt($<HTMLInputElement>("children").value) || 0,
      budget_level: $<HTMLSelectElement>("budget_level").value as "budget" | "mid_range" | "luxury",
      total_budget: budgetVal ? parseFloat(budgetVal) : null,
      currency: $<HTMLSelectElement>("currency").value,
      interests: splitTags($<HTMLInputElement>("interests").value),
      constraints: splitTags($<HTMLInputElement>("constraints").value),
      pace: $<HTMLSelectElement>("pace").value,
    },
  };
}

// ---- Result rendering ----
function renderResult(result: StreamResult): void {
  const status = (result as { status?: string }).status
    || (result as FinalTravelPackage).workflow_state
    || "DONE";

  if (status === "HUMAN_INTERVENE") {
    renderIntervene(result as HumanInterveneResult);
    return;
  }
  if (status === "REJECTED") {
    renderRejected(result as RejectedResult);
    return;
  }

  const pkg = result as FinalTravelPackage;
  show(resultSection);
  let html = "";

  // Overview
  const it = pkg.itinerary || ({} as FinalTravelPackage["itinerary"]);
  html += '<div class="result-overview">';
  html += `<h3>${esc(it.destination || pkg.destination || "")} 旅行方案</h3>`;
  html += `<p>${esc(it.overview || "")}</p>`;
  if (it.trip_style) html += `<p>旅行风格: ${esc(it.trip_style)}</p>`;
  html += "</div>";

  // Daily plan
  for (const day of it.daily_plan || []) {
    html += '<div class="day-card">';
    html += `<h4>Day ${day.day_index} - ${day.date} (${esc(day.theme)}) - ${esc(day.city)}</h4>`;
    for (const a of day.activities || []) {
      html += '<div class="activity-item">';
      html += `<span class="activity-time">${esc(a.start_time)}-${esc(a.end_time)}</span>`;
      html += `${esc(a.title)} @ ${esc(a.location_name)}`;
      if (a.estimated_cost && a.estimated_cost > 0) {
        html += `<span class="activity-cost">$${a.estimated_cost}</span>`;
      }
      html += "</div>";
    }
    html += "</div>";
  }

  // Info grid
  html += '<div class="info-grid">';

  // Budget
  if (pkg.budget) {
    html += '<div class="info-card"><h4>&#x1F4B0; 预算明细</h4>';
    html += '<table class="budget-table"><thead><tr><th>类别</th><th>项目</th><th>费用</th></tr></thead><tbody>';
    for (const bi of pkg.budget.budget_breakdown || []) {
      html += `<tr><td>${esc(bi.category)}</td><td>${esc(bi.item)}</td><td>${bi.estimated_cost} ${esc(bi.currency || pkg.budget!.currency)}</td></tr>`;
    }
    html += "</tbody></table>";
    html += `<p class="budget-total" style="margin-top:0.5rem">总计: ${pkg.budget.total_estimated_cost || 0} ${esc(pkg.budget.currency)}</p>`;
    html += "</div>";
  }

  // Weather
  if (pkg.weather) {
    html += '<div class="info-card"><h4>&#x1F324;&#xFE0F; 天气预报</h4><ul>';
    for (const wd of pkg.weather.forecast_days || []) {
      const temp = (wd.min_temp_c != null && wd.max_temp_c != null)
        ? `${wd.min_temp_c}~${wd.max_temp_c}°C`
        : (wd.temp_range || "");
      html += `<li>${esc(wd.date)}: ${esc(wd.condition)} ${temp}</li>`;
    }
    html += "</ul></div>";
  }

  // Accommodation
  if (pkg.accommodation) {
    html += '<div class="info-card"><h4>&#x1F3E8; 住宿推荐</h4><ul>';
    for (const ht of (pkg.accommodation.hotel_options || []).slice(0, 3)) {
      const price = ht.nightly_rate || ht.price_per_night || "";
      html += `<li>${esc(ht.name)}`;
      if (price) html += ` - $${price}/晚`;
      if (ht.booking_link) html += ` <a href="${esc(ht.booking_link)}" target="_blank">预订</a>`;
      html += "</li>";
    }
    html += "</ul></div>";
  }

  // Flights
  if (pkg.flight_transport) {
    html += '<div class="info-card"><h4>&#x2708;&#xFE0F; 航班信息</h4><ul>';
    for (const fl of (pkg.flight_transport.flight_options || []).slice(0, 3)) {
      const route = fl.route || `${fl.departure_airport} \u2192 ${fl.arrival_airport}`;
      html += `<li>${esc(route)} ${esc(fl.departure_time)}-${esc(fl.arrival_time)}`;
      if (fl.booking_link) html += ` <a href="${esc(fl.booking_link)}" target="_blank">预订</a>`;
      html += "</li>";
    }
    html += "</ul></div>";
  }

  html += "</div>"; // close info-grid

  // Packing list
  if (pkg.packing_list?.length) {
    html += '<div class="info-card" style="margin-top:1rem"><h4>&#x1F9F3; 装箱清单</h4><ul>';
    for (const item of pkg.packing_list) {
      html += `<li>${esc(item)}</li>`;
    }
    html += "</ul></div>";
  }

  // Download bar
  const rid = pkg.request_id || currentRequestId || "";
  html += '<div class="download-bar">';
  html += `<a class="btn-download" href="/download/${encodeURIComponent(rid)}" target="_blank">&#x1F4E5; 下载 Markdown</a>`;
  html += "</div>";

  resultContent.innerHTML = html;
}

function renderIntervene(result: HumanInterveneResult): void {
  show(interveneSection);
  $<HTMLDivElement>("intervene-question").textContent =
    result.question || "系统需要您提供更多信息。";
  currentRequestId = result.request_id || currentRequestId;
}

function renderRejected(result: RejectedResult): void {
  show(rejectedSection);
  let html = "";
  const review = result.review;
  if (review?.summary) html += `<p><strong>审核摘要:</strong> ${esc(review.summary)}</p>`;
  const issues = review?.blocking_issues || result.blocking_issues || [];
  if (issues.length) {
    html += "<ul>";
    for (const issue of issues) html += `<li>${esc(issue)}</li>`;
    html += "</ul>";
  }
  if (result.reason) html += `<p>${esc(result.reason)}</p>`;
  if (result.error) html += `<p>${esc(result.error)}</p>`;
  $<HTMLDivElement>("rejected-content").innerHTML = html;
}

// ---- Form submit ----
planForm.addEventListener("submit", (e: Event) => {
  e.preventDefault();
  const body = buildRequestBody();
  startSSEStream("/plan/stream", body);
});

// ---- Intervene form (uses SSE resume stream) ----
interveneForm.addEventListener("submit", (e: Event) => {
  e.preventDefault();
  const payload: { user_message: string | null; profile_updates: Record<string, unknown> } = {
    user_message: $<HTMLTextAreaElement>("intervene-message").value || null,
    profile_updates: {},
  };
  const budgetVal = $<HTMLInputElement>("intervene-budget").value;
  if (budgetVal) payload.profile_updates.total_budget = parseFloat(budgetVal);

  hide(interveneSection);
  startSSEStream(
    `/resume/${encodeURIComponent(currentRequestId || "")}/stream`,
    payload,
  );
});

// ---- Reset (exposed globally for onclick) ----
(window as unknown as Record<string, unknown>).resetForm = function resetForm(): void {
  hide(workflowSection);
  hide(resultSection);
  hide(interveneSection);
  hide(rejectedSection);
  resetNodes();
  logContent.innerHTML = "";
  resultContent.innerHTML = "";
  window.scrollTo({ top: 0, behavior: "smooth" });
};
