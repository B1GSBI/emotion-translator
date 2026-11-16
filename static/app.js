(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const textEl = $("text"), btn = $("analyze"), counter = $("counter"), errEl = $("error");
  const resultPane = $("result-pane"), emptyEl = $("empty"), resultEl = $("result");
  const maxLen = textEl.maxLength;
  const history = [];

  const updateCounter = () => { counter.textContent = `${textEl.value.length} / ${maxLen}`; };

  const showError = (msg) => { errEl.textContent = msg; errEl.hidden = !msg; };

  function renderResult(data) {
    resultPane.style.setProperty("--accent", data.color);
    resultPane.style.setProperty("--tint", data.color + "1f");

    const emoji = $("emoji");
    emoji.textContent = data.emoji;
    emoji.classList.remove("pop"); void emoji.offsetWidth; emoji.classList.add("pop");

    $("label").textContent = data.label;
    $("confidence").textContent = `впевненість ${Math.round(data.confidence * 100)}%`;
    $("message").textContent = data.message;
    $("advice").textContent = data.advice;
    $("uncertain").hidden = !data.low_confidence;

    const bars = $("bars");
    bars.replaceChildren();
    for (const p of data.probabilities) {
      const li = document.createElement("li");
      const e = document.createElement("span"); e.textContent = p.emoji; e.setAttribute("aria-hidden", "true");
      const l = document.createElement("span"); l.textContent = p.label;
      const track = document.createElement("div"); track.className = "bar-track";
      const fill = document.createElement("div"); fill.className = "bar-fill";
      fill.style.background = p.emotion === data.emotion ? data.color : "var(--muted)";
      track.append(fill);
      const pct = document.createElement("span"); pct.className = "pct";
      pct.textContent = `${Math.round(p.probability * 100)}%`;
      li.append(e, l, track, pct);
      bars.append(li);
      requestAnimationFrame(() => { fill.style.width = `${Math.round(p.probability * 100)}%`; });
    }

    emptyEl.hidden = true;
    resultEl.hidden = false;
  }

  function addHistory(text, data) {
    history.unshift({ text, emoji: data.emoji, label: data.label });
    history.splice(5);
    const ul = $("history");
    ul.replaceChildren();
    for (const item of history) {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button"; b.className = "chip"; b.title = "Повернути цей текст у поле";
      const em = document.createElement("span"); em.textContent = item.emoji;
      const tx = document.createElement("span"); tx.textContent = `${item.label}: ${item.text}`;
      b.append(em, tx);
      b.addEventListener("click", () => { textEl.value = item.text; updateCounter(); textEl.focus(); });
      li.append(b); ul.append(li);
    }
    $("history-wrap").hidden = false;
  }

  async function analyze() {
    const text = textEl.value.trim();
    showError("");
    if (!text) { showError("Введіть текст для аналізу."); textEl.focus(); return; }

    btn.disabled = true; const label = btn.textContent; btn.textContent = "Аналізую…";
    try {
      const resp = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) { showError(data.error || "Не вдалося виконати аналіз. Спробуйте ще раз."); return; }
      renderResult(data);
      addHistory(text, data);
    } catch (e) {
      showError("Немає зв’язку із сервером. Перевірте інтернет і спробуйте ще раз.");
    } finally {
      btn.disabled = false; btn.textContent = label;
    }
  }

  textEl.addEventListener("input", updateCounter);
  textEl.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze(); });
  btn.addEventListener("click", analyze);
  document.querySelectorAll("[data-example]").forEach((b) =>
    b.addEventListener("click", () => {
      textEl.value = b.dataset.example; updateCounter(); showError(""); analyze();
    }));
  updateCounter();
})();
