/* Login page: register + login against /api/auth (hosted static version). */
"use strict";

function showErrors(id, errs) {
  const box = document.getElementById(id);
  box.textContent = (errs || []).join(" ");
  box.classList.remove("hidden");
}

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
    const channelSel = document.getElementById("r-channel");
    await api("/api/auth/register", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        email: document.getElementById("r-email").value,
        display_name: document.getElementById("r-name").value,
        password: document.getElementById("r-password").value,
      }),
    });
    location.href = "/dashboard";
  } catch (err) { showErrors("register-errors", [err.message]); }
});
