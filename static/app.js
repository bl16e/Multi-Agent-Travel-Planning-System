"use strict";
(() => {
  // src/app.ts
  var STEP_NAMES = [
    "shangshu_preflight",
    "zhongshu_itinerary",
    "menxia_review",
    "shangshu_review_gate",
    "shangshu_dispatch_liubu",
    "liubu_weather",
    "liubu_budget",
    "liubu_accommodation",
    "liubu_flight_transport",
    "liubu_calendar",
    "shangshu_assemble"
  ];
  function $(id) {
    return document.getElementById(id);
  }
  var currentRequestId = null;
  var sseEventType = "";
  var workflowSection = $("workflow-section");
  var resultSection = $("result-section");
  var interveneSection = $("intervene-section");
  var rejectedSection = $("rejected-section");
  var logContent = $("log-content");
  var resultContent = $("result-content");
  var planForm = $("plan-form");
  var interveneForm = $("intervene-form");
  var submitBtn = $("submit-btn");
  (function setDefaultDates() {
    const today = /* @__PURE__ */ new Date();
    const start = new Date(today);
    start.setDate(today.getDate() + 14);
    const end = new Date(start);
    end.setDate(start.getDate() + 3);
    $("start_date").value = fmtDate(start);
    $("end_date").value = fmtDate(end);
  })();
  function fmtDate(d) {
    return d.toISOString().split("T")[0];
  }
  function esc(s) {
    if (s == null) return "";
    const d = document.createElement("div");
    d.textContent = String(s);
    return d.innerHTML;
  }
  function splitTags(s) {
    return s.split(/[,，]/).map((t) => t.trim()).filter(Boolean);
  }
  function show(el) {
    el.style.display = "block";
  }
  function hide(el) {
    el.style.display = "none";
  }
  function resetNodes() {
    document.querySelectorAll(".node").forEach((n) => {
      n.classList.remove("node-active", "node-done", "node-error");
    });
  }
  function setNodeState(step, state) {
    const node = document.querySelector(`.node[data-step="${step}"]`);
    if (!node) return;
    node.classList.remove("node-active", "node-done", "node-error");
    node.classList.add(`node-${state}`);
  }
  function updateNodeFromLog(line) {
    const m = line.match(/\[\d{2}:\d{2}:\d{2}\]\s+\[(\w+)\]\s+(\w+)/);
    if (!m) return;
    const phase = m[1].toLowerCase();
    const step = m[2];
    if (step === "shangshu_dispatch_liubu" || STEP_NAMES.includes(step)) {
      if (phase === "start") setNodeState(step, "active");
      else if (phase === "done") setNodeState(step, "done");
      else if (phase === "error") setNodeState(step, "error");
    }
  }
  function appendLog(text, phase) {
    const div = document.createElement("div");
    div.className = "log-line" + (phase ? ` phase-${phase}` : "");
    div.textContent = text;
    logContent.appendChild(div);
    logContent.scrollTop = logContent.scrollHeight;
  }
  function parsePhase(line) {
    const m = line.match(/\[([A-Z_]+)\]/);
    if (!m) return "";
    const p = m[1].toLowerCase();
    if (p === "start" || p === "done" || p === "error") return p;
    return "";
  }
  function startSSEStream(url, body) {
    resetNodes();
    logContent.innerHTML = "";
    resultContent.innerHTML = "";
    hide(resultSection);
    hide(interveneSection);
    hide(rejectedSection);
    show(workflowSection);
    submitBtn.disabled = true;
    submitBtn.textContent = "\u89C4\u5212\u4E2D...";
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      function pump() {
        return reader.read().then(({ done, value }) => {
          if (done) {
            onStreamEnd();
            return;
          }
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop();
          for (const line of lines) {
            processSSELine(line);
          }
          return pump();
        });
      }
      return pump();
    }).catch((err) => {
      appendLog(`ERROR: ${err.message}`, "error");
      onStreamEnd();
    });
  }
  function processSSELine(line) {
    if (line.startsWith("event: ")) {
      sseEventType = line.substring(7).trim();
      return;
    }
    if (line.startsWith("data: ")) {
      handleSSEEvent(sseEventType, line.substring(6));
      sseEventType = "";
    }
  }
  function handleSSEEvent(type, data) {
    if (type === "progress") {
      try {
        const obj = JSON.parse(data);
        const line = obj.line || data;
        appendLog(line, parsePhase(line));
        updateNodeFromLog(line);
      } catch {
        appendLog(data, "");
      }
    } else if (type === "result") {
      try {
        const result = JSON.parse(data);
        renderResult(result);
      } catch (e) {
        appendLog(`Failed to parse result: ${e.message}`, "error");
      }
    } else if (type === "error") {
      try {
        const err = JSON.parse(data);
        appendLog(`ERROR: ${err.error || data}`, "error");
      } catch {
        appendLog(`ERROR: ${data}`, "error");
      }
    } else if (type === "done") {
      setNodeState("done", "done");
    }
  }
  function onStreamEnd() {
    submitBtn.disabled = false;
    submitBtn.textContent = "\u{1F680} \u5F00\u59CB\u89C4\u5212";
  }
  function buildRequestBody() {
    const ts = Date.now();
    currentRequestId = `web_${ts}`;
    const budgetVal = $("total_budget").value;
    return {
      request_id: currentRequestId,
      user_message: $("user_message").value,
      profile: {
        origin_city: $("origin_city").value,
        destination_preferences: [$("destination").value],
        start_date: $("start_date").value,
        end_date: $("end_date").value,
        adults: parseInt($("adults").value) || 1,
        children: parseInt($("children").value) || 0,
        budget_level: $("budget_level").value,
        total_budget: budgetVal ? parseFloat(budgetVal) : null,
        currency: $("currency").value,
        interests: splitTags($("interests").value),
        constraints: splitTags($("constraints").value),
        pace: $("pace").value
      }
    };
  }
  function renderResult(result) {
    const status = result.status || result.workflow_state || "DONE";
    if (status === "HUMAN_INTERVENE") {
      renderIntervene(result);
      return;
    }
    if (status === "REJECTED") {
      renderRejected(result);
      return;
    }
    const pkg = result;
    show(resultSection);
    let html = "";
    const it = pkg.itinerary || {};
    html += '<div class="result-overview">';
    html += `<h3>${esc(it.destination || pkg.destination || "")} \u65C5\u884C\u65B9\u6848</h3>`;
    html += `<p>${esc(it.overview || "")}</p>`;
    if (it.trip_style) html += `<p>\u65C5\u884C\u98CE\u683C: ${esc(it.trip_style)}</p>`;
    html += "</div>";
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
    html += '<div class="info-grid">';
    if (pkg.budget) {
      html += '<div class="info-card"><h4>&#x1F4B0; \u9884\u7B97\u660E\u7EC6</h4>';
      html += '<table class="budget-table"><thead><tr><th>\u7C7B\u522B</th><th>\u9879\u76EE</th><th>\u8D39\u7528</th></tr></thead><tbody>';
      for (const bi of pkg.budget.budget_breakdown || []) {
        html += `<tr><td>${esc(bi.category)}</td><td>${esc(bi.item)}</td><td>${bi.estimated_cost} ${esc(bi.currency || pkg.budget.currency)}</td></tr>`;
      }
      html += "</tbody></table>";
      html += `<p class="budget-total" style="margin-top:0.5rem">\u603B\u8BA1: ${pkg.budget.total_estimated_cost || 0} ${esc(pkg.budget.currency)}</p>`;
      html += "</div>";
    }
    if (pkg.weather) {
      html += '<div class="info-card"><h4>&#x1F324;&#xFE0F; \u5929\u6C14\u9884\u62A5</h4><ul>';
      for (const wd of pkg.weather.forecast_days || []) {
        const temp = wd.min_temp_c != null && wd.max_temp_c != null ? `${wd.min_temp_c}~${wd.max_temp_c}\xB0C` : wd.temp_range || "";
        html += `<li>${esc(wd.date)}: ${esc(wd.condition)} ${temp}</li>`;
      }
      html += "</ul></div>";
    }
    if (pkg.accommodation) {
      html += '<div class="info-card"><h4>&#x1F3E8; \u4F4F\u5BBF\u63A8\u8350</h4><ul>';
      for (const ht of (pkg.accommodation.hotel_options || []).slice(0, 3)) {
        const price = ht.nightly_rate || ht.price_per_night || "";
        html += `<li>${esc(ht.name)}`;
        if (price) html += ` - $${price}/\u665A`;
        if (ht.booking_link) html += ` <a href="${esc(ht.booking_link)}" target="_blank">\u9884\u8BA2</a>`;
        html += "</li>";
      }
      html += "</ul></div>";
    }
    if (pkg.flight_transport) {
      html += '<div class="info-card"><h4>&#x2708;&#xFE0F; \u822A\u73ED\u4FE1\u606F</h4><ul>';
      for (const fl of (pkg.flight_transport.flight_options || []).slice(0, 3)) {
        const route = fl.route || `${fl.departure_airport} \u2192 ${fl.arrival_airport}`;
        html += `<li>${esc(route)} ${esc(fl.departure_time)}-${esc(fl.arrival_time)}`;
        if (fl.booking_link) html += ` <a href="${esc(fl.booking_link)}" target="_blank">\u9884\u8BA2</a>`;
        html += "</li>";
      }
      html += "</ul></div>";
    }
    html += "</div>";
    if (pkg.packing_list?.length) {
      html += '<div class="info-card" style="margin-top:1rem"><h4>&#x1F9F3; \u88C5\u7BB1\u6E05\u5355</h4><ul>';
      for (const item of pkg.packing_list) {
        html += `<li>${esc(item)}</li>`;
      }
      html += "</ul></div>";
    }
    const rid = pkg.request_id || currentRequestId || "";
    html += '<div class="download-bar">';
    html += `<a class="btn-download" href="/download/${encodeURIComponent(rid)}" target="_blank">&#x1F4E5; \u4E0B\u8F7D Markdown</a>`;
    html += "</div>";
    resultContent.innerHTML = html;
  }
  function renderIntervene(result) {
    show(interveneSection);
    $("intervene-question").textContent = result.question || "\u7CFB\u7EDF\u9700\u8981\u60A8\u63D0\u4F9B\u66F4\u591A\u4FE1\u606F\u3002";
    currentRequestId = result.request_id || currentRequestId;
  }
  function renderRejected(result) {
    show(rejectedSection);
    let html = "";
    const review = result.review;
    if (review?.summary) html += `<p><strong>\u5BA1\u6838\u6458\u8981:</strong> ${esc(review.summary)}</p>`;
    const issues = review?.blocking_issues || result.blocking_issues || [];
    if (issues.length) {
      html += "<ul>";
      for (const issue of issues) html += `<li>${esc(issue)}</li>`;
      html += "</ul>";
    }
    if (result.reason) html += `<p>${esc(result.reason)}</p>`;
    if (result.error) html += `<p>${esc(result.error)}</p>`;
    $("rejected-content").innerHTML = html;
  }
  planForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const body = buildRequestBody();
    startSSEStream("/plan/stream", body);
  });
  interveneForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const payload = {
      user_message: $("intervene-message").value || null,
      profile_updates: {}
    };
    const budgetVal = $("intervene-budget").value;
    if (budgetVal) payload.profile_updates.total_budget = parseFloat(budgetVal);
    hide(interveneSection);
    startSSEStream(
      `/resume/${encodeURIComponent(currentRequestId || "")}/stream`,
      payload
    );
  });
  window.resetForm = function resetForm() {
    hide(workflowSection);
    hide(resultSection);
    hide(interveneSection);
    hide(rejectedSection);
    resetNodes();
    logContent.innerHTML = "";
    resultContent.innerHTML = "";
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
})();
