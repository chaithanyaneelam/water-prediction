/* Login page: register + login against /api/auth. */
"use strict";

function showErrors(id, errs) {
  const box = document.getElementById(id);
  box.textContent = (errs || []).join(" ");
  box.classList.remove("hidden");
}

async function refreshNav() {
  const me = await fetch("/api/auth/me").then(r => r.json());
  const nav = document.querySelector(".navbar nav");
  let authArea = document.getElementById("auth-area");
  if (!authArea) {
    authArea = document.createElement("div");
    authArea.id = "auth-area";
    authArea.style.cssText = "display:flex;gap:0.5rem;align-items:center;";
    document.querySelector(".navbar").appendChild(authArea);
  }
  if (me.user) {
    authArea.innerHTML =
      `<span class="badge muted">${me.user.email || me.user.phone}</span>` +
      `<button class="btn secondary" id="btn-logout">Logout</button>`;
    document.getElementById("btn-logout").addEventListener("click", async () => {
      await api("/api/auth/logout", { method: "POST" });
      location.href = "/";
    });
  } else {
    authArea.innerHTML = `<a class="btn secondary" href="/login">Login / Register</a>`;
  }
}
window.refreshNav = refreshNav;

document.getElementById("btn-login").addEventListener("click", async () => {
  try {
    await api("/api/auth/login", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        identifier: document.getElementById("l-identifier").value,
        password: document.getElementById("l-password").value,
      }),
    });
    location.href = "/dashboard";
  } catch (err) { showErrors("login-errors", [err.message]); }
});

document.getElementById("btn-register").addEventListener("click", async () => {
  try {
    await api("/api/auth/register", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        email: document.getElementById("r-email").value,
        phone: document.getElementById("r-phone").value,
        display_name: document.getElementById("r-name").value,
        password: document.getElementById("r-password").value,
        preferred_channel: document.getElementById("r-channel").value,
      }),
    });
    location.href = "/dashboard";
  } catch (err) { showErrors("register-errors", [err.message]); }
});

refreshNav();
