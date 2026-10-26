
const userText = document.getElementById("userText");
const analyzeBtn = document.getElementById("analyzeBtn");
const btnText = document.getElementById("btnText");
const clearBtn = document.getElementById("clearBtn");
const charCount = document.getElementById("charCount");
const errorMessage = document.getElementById("errorMessage");

const serverStatus = document.getElementById("serverStatus");
const statusDot = document.querySelector(".status-dot");

const emptyState = document.getElementById("emptyState");
const resultContent = document.getElementById("resultContent");

const emotionInfo = {
  sadness: {
    emoji: "😢",
    description: "A tone associated with sadness or disappointment",
  },
  joy: {
    emoji: "😄",
    description: "A tone associated with happiness or excitement",
  },
  love: {
    emoji: "❤️",
    description: "A tone associated with affection or care",
  },
  anger: {
    emoji: "😠",
    description: "A tone associated with anger or frustration",
  },
  fear: {
    emoji: "😨",
    description: "A tone associated with fear or worry",
  },
  surprise: {
    emoji: "😲",
    description: "A tone associated with surprise or amazement",
  },
};

function updateCharCount() {
  charCount.textContent = `${userText.value.length} / 2000 characters`;
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.hidden = true;
}

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  userText.disabled = isLoading;
  btnText.textContent = isLoading ? "Analyzing your emotions..." : "Analyze my emotions";
}

async function checkServer() {
  try {
    const response = await fetch("/health");

    if (!response.ok) {
      throw new Error("Health check failed");
    }

    const data = await response.json();

    if (data.model_loaded) {
      serverStatus.textContent = "AI engine connected";
      statusDot.className = "status-dot online";
    } else {
      serverStatus.textContent = "Model not loaded";
      statusDot.className = "status-dot offline";
    }
  } catch (error) {
    serverStatus.textContent = "Server unavailable";
    statusDot.className = "status-dot offline";
  }
}

function formatEmotion(emotion) {
  return emotion.charAt(0).toUpperCase() + emotion.slice(1);
}

function formatPercentage(value) {
  // Model outputs are normally probabilities between 0 and 1.
  return `${(value * 100).toFixed(2)}%`;
}

function displayPrediction(data) {
  const emotion = data.predicted_emotion.toLowerCase();
  const info = emotionInfo[emotion];

  if (!info) {
    throw new Error("The API returned an unknown emotion label.");
  }

  const confidence = Number(data.confidence);

  if (!Number.isFinite(confidence)) {
    throw new Error("The API returned an invalid confidence score.");
  }

  document.getElementById("mainEmoji").textContent = info.emoji;
  document.getElementById("emotionName").textContent = formatEmotion(emotion);
  document.getElementById("emotionDescription").textContent = info.description;

  document.getElementById("confidenceValue").textContent =
    formatPercentage(confidence);

  document.getElementById("confidenceBar").style.width =
    `${Math.max(0, Math.min(100, confidence * 100))}%`;

  document.getElementById("analyzedText").textContent = data.text;

  const probabilityList = document.getElementById("probabilityList");
  probabilityList.replaceChildren();

  const probabilities = data.all_probablities;

  if (!probabilities || typeof probabilities !== "object") {
    throw new Error("The API did not return emotion probabilities.");
  }

  const sortedProbabilities = Object.entries(probabilities)
    .sort((a, b) => Number(b[1]) - Number(a[1]));

  for (const [label, rawProbability] of sortedProbabilities) {
    const probability = Number(rawProbability);
    const details = emotionInfo[label.toLowerCase()];

    if (!details || !Number.isFinite(probability)) {
      continue;
    }

    const percentage = Math.max(0, Math.min(100, probability * 100));

    const item = document.createElement("div");
    item.className = "probability-item";

    if (label.toLowerCase() === emotion) {
      item.classList.add("top-emotion");
    }

    const meta = document.createElement("div");
    meta.className = "probability-meta";

    const name = document.createElement("span");
    name.className = "probability-name";

    const emoji = document.createElement("span");
    emoji.className = "emotion-emoji";
    emoji.textContent = details.emoji;

    const labelText = document.createElement("span");
    labelText.textContent = formatEmotion(label);

    name.append(emoji, labelText);

    const value = document.createElement("span");
    value.className = "probability-value";
    value.textContent = formatPercentage(probability);

    meta.append(name, value);

    const track = document.createElement("div");
    track.className = "probability-track";

    const fill = document.createElement("div");
    fill.className = "probability-fill";

    track.appendChild(fill);
    item.append(meta, track);
    probabilityList.appendChild(item);

    // Animate each probability bar after inserting it into the page.
    requestAnimationFrame(() => {
      fill.style.width = `${percentage}%`;
    });
  }

  emptyState.hidden = true;
  resultContent.hidden = false;

  // Move results into view on small screens.
  if (window.innerWidth <= 900) {
    document.getElementById("resultPanel").scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }
}

async function analyzeEmotion() {
  clearError();

  const text = userText.value.trim();

  if (!text) {
    showError("Please enter a sentence before analyzing your emotions.");
    userText.focus();
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const detail = data.detail;

      const message = Array.isArray(detail)
        ? detail.map((item) => item.msg || "Invalid input").join(", ")
        : typeof detail === "string"
          ? detail
          : `Request failed with status ${response.status}.`;

      throw new Error(message);
    }

    displayPrediction(data);
  } catch (error) {
    if (error instanceof TypeError) {
      showError("Could not connect to the server. Check that FastAPI is running.");
    } else {
      showError(error.message || "Something went wrong. Please try again.");
    }
  } finally {
    setLoading(false);
  }
}

userText.addEventListener("input", updateCharCount);

userText.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    analyzeEmotion();
  }
});

analyzeBtn.addEventListener("click", analyzeEmotion);

clearBtn.addEventListener("click", () => {
  userText.value = "";
  updateCharCount();
  clearError();
  userText.focus();
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    userText.value = chip.dataset.text || "";
    updateCharCount();
    clearError();
    userText.focus();
  });
});

updateCharCount();
checkServer();
