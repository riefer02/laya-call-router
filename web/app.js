"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  chat: $("chat-log"),
  stages: $("stages"),
  personaBar: $("persona-bar"),
  composer: $("composer"),
  input: $("message-input"),
  send: $("send-btn"),
  tokenCount: $("token-count"),
  latency: $("latency-stat"),
  pace: $("pace-toggle"),
  reset: $("reset-btn"),
};

let sessionId = null;
let waiting = false;
let streaming = false;
let stageEls = {};
let totalStageMs = 0;
let lastStageEl = null;

// ---------------------------------------------------------------- tiny DOM helpers
function h(tag, props = {}, children = []) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) el.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c) el.appendChild(c);
  return el;
}

function confColor(c) {
  if (c >= 0.85) return "var(--ok)";
  if (c >= 0.6) return "var(--warn)";
  return "var(--bad)";
}

function clearUI() {
  sessionId = null;
  waiting = false;
  streaming = false;
  stageEls = {};
  totalStageMs = 0;
  lastStageEl = null;
  els.chat.innerHTML = "";
  els.stages.innerHTML = '<div class="empty">No run yet.</div>';
  els.tokenCount.textContent = "0";
  els.latency.textContent = "— ms";
  els.send.disabled = false;
  els.input.disabled = false;
}

// ---------------------------------------------------------------- chat
function addMessage(role, text, templated = false) {
  const who = role === "customer" ? "Customer" : "Agent";
  const head = h("span", { class: "who", text: who });
  if (templated) head.appendChild(h("span", { class: "templated", text: "  · templated, not generated" }));
  const bubble = h("div", { class: "msg " + role }, [head, document.createTextNode(text)]);
  els.chat.appendChild(bubble);
  els.chat.scrollTop = els.chat.scrollHeight;
  return bubble;
}

function addSuggestions(options, onPick) {
  const wrap = h("div", { class: "chips" });
  for (const opt of options) {
    wrap.appendChild(
      h("button", {
        class: "chip suggest",
        type: "button",
        text: opt,
        onclick: () => onPick(opt),
      })
    );
  }
  els.chat.appendChild(wrap);
  els.chat.scrollTop = els.chat.scrollHeight;
}

// ---------------------------------------------------------------- pipeline rendering
function startStage(ev) {
  if (els.stages.querySelector(".empty")) els.stages.innerHTML = "";
  const el = h("div", { class: "stage running" }, [
    h("div", { class: "stage-head" }, [
      h("span", { class: "stage-title", text: ev.title }),
      h("span", { class: "prim", text: ev.primitive }),
      h("span", { class: "stage-meta", text: "…" }),
    ]),
  ]);
  els.stages.appendChild(el);
  stageEls[ev.stage] = el;
  lastStageEl = el;
  els.stages.scrollTop = els.stages.scrollHeight;
}

function renderChoice(qid, ans) {
  const probs = ans.probabilities || {};
  const ranked = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  const rows = ranked.map(([label, p]) =>
    h("div", { class: "opt" + (label === ans.choice ? " top" : "") }, [
      h("span", { class: "label", text: label }),
      h("span", { class: "track" }, [
        h("span", { class: "fill", style: `width:${Math.round(p * 100)}%` }),
      ]),
      h("span", { class: "p", text: p.toFixed(2) }),
    ])
  );
  return h("div", { class: "ans" }, [
    h("div", { class: "ans-head" }, [
      h("span", { class: "qid", text: qid }),
      h("span", { class: "pick", text: ans.choice }),
      h("span", {
        class: "conf",
        style: `color:${confColor(ans.confidence)}`,
        text: `conf ${ans.confidence.toFixed(2)} · p ${ans.top_probability.toFixed(2)}`,
      }),
    ]),
    h("div", { class: "opts" }, rows),
  ]);
}

function renderScore(qid, ans) {
  const legend = ans.legend || {};
  const levels = Object.keys(legend).map((k) => legend[k]);
  const probs = Object.entries(ans.probabilities || {});
  const rows = probs.map(([k, p]) =>
    h("div", { class: "opt" + (Number(k) === Math.round(ans.score) ? " top" : "") }, [
      h("span", { class: "label", text: `${k}: ${legend[k] || k}` }),
      h("span", { class: "track" }, [
        h("span", { class: "fill", style: `width:${Math.round(p * 100)}%` }),
      ]),
      h("span", { class: "p", text: p.toFixed(2) }),
    ])
  );
  return h("div", { class: "ans" }, [
    h("div", { class: "ans-head" }, [
      h("span", { class: "qid", text: qid }),
      h("span", { class: "pick", text: `level ${ans.score.toFixed(2)} / ${levels.length - 1}` }),
      h("span", { class: "conf", style: `color:${confColor(ans.confidence)}`, text: `conf ${ans.confidence.toFixed(2)}` }),
    ]),
    h("div", { class: "opts" }, rows),
  ]);
}

function renderNoul(qid, ans) {
  const p = ans.noul;
  return h("div", { class: "ans" }, [
    h("div", { class: "ans-head" }, [
      h("span", { class: "qid", text: qid }),
      h("span", { class: "pick", text: `P(true) = ${p.toFixed(2)}` }),
      h("span", { class: "conf", style: `color:${confColor(ans.confidence)}`, text: `conf ${ans.confidence.toFixed(2)}` }),
    ]),
    h("div", { class: "opts" }, [
      h("div", { class: "opt" + (p >= 0.5 ? " top" : "") }, [
        h("span", { class: "label", text: "yes" }),
        h("span", { class: "track" }, [
          h("span", { class: "fill", style: `width:${Math.round(p * 100)}%` }),
        ]),
        h("span", { class: "p", text: p.toFixed(2) }),
      ]),
    ]),
  ]);
}

function renderStageResult(ev) {
  const el = stageEls[ev.stage] || lastStageEl;
  if (!el) return;
  el.classList.remove("running");
  el.classList.add(ev.status || "ok");
  const meta = el.querySelector(".stage-meta");
  if (meta) meta.textContent = `${ev.model || "policy"} · ${ev.latency_ms} ms`;
  totalStageMs += ev.latency_ms || 0;
  els.latency.textContent = `${totalStageMs.toFixed(0)} ms`;

  const answers = ev.answers || {};
  for (const [qid, ans] of Object.entries(answers)) {
    if (ans.primitive === "choice") el.appendChild(renderChoice(qid, ans));
    else if (ans.primitive === "score") el.appendChild(renderScore(qid, ans));
    else if (ans.primitive === "noul") el.appendChild(renderNoul(qid, ans));
  }
  if (ev.note) el.appendChild(h("div", { class: "stage-note", text: ev.note }));
}

function renderRouting(ev) {
  const flags = (ev.flags || []).map((f) => h("span", { class: "flag", text: f }));
  const el = h("div", { class: "stage route-card" }, [
    h("div", { class: "stage-head" }, [
      h("span", { class: "stage-title", text: "Route →" }),
      h("span", { class: "prim", text: "policy" }),
    ]),
    h("div", { class: "route-dest", text: ev.queue }),
    h("div", { class: "route-grid" }, [
      h("span", {}, [document.createTextNode("priority "), h("b", { text: ev.priority })]),
      h("span", {}, [document.createTextNode("handler "), h("b", { text: ev.handler })]),
    ]),
    flags.length ? h("div", {}, flags) : null,
    h("ul", { class: "route-reasons" }, (ev.reasons || []).map((r) => h("li", { text: r }))),
  ]);
  els.stages.appendChild(el);
  els.stages.scrollTop = els.stages.scrollHeight;
}

function renderDone(ev) {
  els.tokenCount.textContent = String(ev.tokens_generated ?? 0);
  els.latency.textContent = `${Math.round(ev.compute_ms ?? ev.total_ms)} ms`;
  els.stages.appendChild(
    h("div", { class: "done-strip" }, [
      document.createTextNode(`${ev.stages_run} stages · `),
      h("b", { text: `${Math.round(ev.compute_ms ?? ev.total_ms)} ms compute` }),
      document.createTextNode(` · ${ev.tokens_generated} tokens generated`),
    ])
  );
  setBusy(false);
}

function renderError(ev) {
  els.stages.appendChild(h("div", { class: "stage rejected" }, [
    h("div", { class: "stage-head", text: "Error" }),
    h("div", { class: "stage-note", text: ev.message }),
  ]));
  setBusy(false);
}

// ---------------------------------------------------------------- event dispatch
function handleEvent(ev) {
  switch (ev.type) {
    case "stage_start": startStage(ev); break;
    case "stage_result": renderStageResult(ev); break;
    case "clarify": handleClarify(ev); break;
    case "routing": renderRouting(ev); break;
    case "done": renderDone(ev); break;
    case "error": renderError(ev); break;
  }
}

function handleClarify(ev) {
  waiting = true;
  setBusy(false);
  addMessage("agent", ev.prompt, true);
  addSuggestions(ev.chips, answer);
  els.input.focus();
}

// ---------------------------------------------------------------- streaming
function parseBlock(raw) {
  let data = "";
  let name = "message";
  for (const line of raw.split("\n")) {
    if (line.startsWith("data:")) data += line.slice(5).trim();
    else if (line.startsWith("event:")) name = line.slice(6).trim();
  }
  if (name === "close" || !data) return null;
  try {
    return JSON.parse(data);
  } catch {
    return null;
  }
}

async function streamSession(id) {
  streaming = true;
  const res = await fetch(`/api/session/${id}/stream`);
  if (!res.body) return;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const ev = parseBlock(raw);
      if (ev) handleEvent(ev);
    }
  }
  streaming = false;
}

// ---------------------------------------------------------------- run control
function setBusy(busy) {
  els.send.disabled = busy;
  els.input.disabled = busy;
}

async function startRun(message) {
  clearUI();
  addMessage("customer", message);
  setBusy(true);
  const res = await fetch("/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, pace: els.pace.checked }),
  });
  if (!res.ok) {
    renderError({ message: `could not start session (${res.status})` });
    return;
  }
  const data = await res.json();
  sessionId = data.session_id;
  await streamSession(sessionId);
}

async function answer(text) {
  if (!sessionId || !waiting) return;
  waiting = false;
  addMessage("customer", text);
  setBusy(true);
  await fetch(`/api/session/${sessionId}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  await streamSession(sessionId);
}

// ---------------------------------------------------------------- bootstrap
async function loadPersonas() {
  const res = await fetch("/api/personas");
  const data = await res.json();
  for (const p of data.personas) {
    els.personaBar.appendChild(
      h("button", {
        class: "chip persona",
        type: "button",
        title: p.expect,
        text: p.label,
        onclick: () => startRun(p.message),
      })
    );
  }
}

els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = els.input.value.trim();
  if (!text) return;
  els.input.value = "";
  if (waiting) answer(text);
  else if (!streaming) startRun(text);
});

els.reset.addEventListener("click", clearUI);
loadPersonas();
