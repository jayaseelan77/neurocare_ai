import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import {
  getAuth,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";

const firebaseConfig = window.FIREBASE_WEB_CONFIG || null;
const authErrorBox = document.getElementById("authError");

function showAuthError(message) {
  if (!authErrorBox) {
    return;
  }
  authErrorBox.textContent = message;
  authErrorBox.classList.remove("d-none");
}

function clearAuthError() {
  if (!authErrorBox) {
    return;
  }
  authErrorBox.textContent = "";
  authErrorBox.classList.add("d-none");
}

async function createServerSession(idToken, is_registering = false, role = 'patient') {
  const response = await fetch("/session_login", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ idToken, is_registering, role })
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data.success) {
    throw new Error(data.message || "Unable to create authenticated session.");
  }

  return data;
}

if (!firebaseConfig) {
  showAuthError("Firebase configuration is missing.");
} else {
  const app = initializeApp(firebaseConfig);
  const auth = getAuth(app);

  const loginForm = document.getElementById("loginForm");
  if (loginForm) {
    loginForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      clearAuthError();

      const submitBtn = document.getElementById("loginSubmit");
      if (submitBtn) {
        submitBtn.disabled = true;
      }

      try {
        const email = document.getElementById("loginEmail").value.trim();
        const password = document.getElementById("loginPassword").value;
        const credentials = await signInWithEmailAndPassword(auth, email, password);
        const idToken = await credentials.user.getIdToken();
        const sessionData = await createServerSession(idToken, false);
        window.location.assign(sessionData.redirect || "/dashboard");
      } catch (error) {
        showAuthError(error.message || "Login failed.");
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
        }
      }
    });
  }

  const registerForm = document.getElementById("registerForm");
  if (registerForm) {
    registerForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      clearAuthError();

      const submitBtn = document.getElementById("registerSubmit");
      if (submitBtn) {
        submitBtn.disabled = true;
      }

      try {
        const email = document.getElementById("registerEmail").value.trim();
        const password = document.getElementById("registerPassword").value;
        const roleElement = document.getElementById("registerRole");
        const role = roleElement ? roleElement.value : "patient";
        
        const credentials = await createUserWithEmailAndPassword(auth, email, password);
        const idToken = await credentials.user.getIdToken();
        const sessionData = await createServerSession(idToken, true, role);
        window.location.assign(sessionData.redirect || "/dashboard");
      } catch (error) {
        showAuthError(error.message || "Registration failed.");
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
        }
      }
    });
  }
}
