/* Notifications (outbox) page. */
"use strict";

async function load() {
  let d;
  try {
    d = await api("/api/auth/notifications");
  } catch (err) {
    document.querySelector("#notif-table tbody").innerHTML =
      `<tr><td colspan="7">Please log in to see your messages.</td></tr>`;
    if (window.refreshNav) refreshNav();
    return;
  }
  const tb = document.querySelector("#notif-table tbody");
  tb.innerHTML = d.items.map(n => `
    <tr>
      <td>${n.id}</td>
      <td>${(n.created_at || "").replace("T", " ").slice(0, 16)}</td>
      <td><span class="badge muted">${n.channel}</span></td>
      <td>${n.destination || "-"}</td>
      <td><b>${n.subject || ""}</b><br><span class="note">${(n.body || "").replace(/\n/g, " | ")}</span></td>
      <td><span class="badge ${n.status === "sent" ? "good" : n.status === "failed" ? "bad" : "muted"}">${n.status}</span>
        ${n.error ? `<br><span class="note">${n.error}</span>` : ""}</td>
      <td>${n.status !== "sent"
        ? `<button class="btn secondary" data-retry="${n.id}">Retry</button>` : ""}</td>
    </tr>`).join("") || `<tr><td colspan="7">No messages yet - run a prediction first.</td></tr>`;

  tb.querySelectorAll("[data-retry]").forEach(btn =>
    btn.addEventListener("click", async () => {
      await api(`/api/auth/notifications/${btn.dataset.retry}/retry`, { method: "POST" });
      load();
    }));
}

load();
if (window.refreshNav) refreshNav();
