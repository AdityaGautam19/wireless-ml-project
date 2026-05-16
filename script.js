const modulations = [
  {
    name: "8PSK",
    frequency: 3,
    phase: 0.2,
    description: "Eight Phase Shift Keying carries information through eight phase states for higher spectral efficiency."
  },
  {
    name: "AM-DSB",
    frequency: 2,
    phase: 0,
    description: "Amplitude Modulation Double Sideband varies carrier amplitude and preserves both sidebands."
  },
  {
    name: "AM-SSB",
    frequency: 2.4,
    phase: 0.6,
    description: "Amplitude Modulation Single Sideband transmits one sideband to improve bandwidth efficiency."
  },
  {
    name: "BPSK",
    frequency: 1.6,
    phase: 3.14,
    description: "Binary Phase Shift Keying uses two phase states and is robust in noisy digital links."
  },
  {
    name: "CPFSK",
    frequency: 4.1,
    phase: 0.7,
    description: "Continuous Phase FSK changes frequency while keeping phase transitions smooth."
  },
  {
    name: "GFSK",
    frequency: 4.7,
    phase: 1.1,
    description: "Gaussian FSK filters symbols before frequency modulation to reduce occupied bandwidth."
  },
  {
    name: "PAM4",
    frequency: 2.8,
    phase: 0.4,
    description: "Pulse Amplitude Modulation with four levels represents two bits per symbol through amplitude states."
  },
  {
    name: "QAM16",
    frequency: 3.4,
    phase: 0.9,
    description: "16-QAM combines amplitude and phase changes to encode four bits per symbol."
  },
  {
    name: "QAM64",
    frequency: 4.3,
    phase: 1.3,
    description: "64-QAM packs six bits per symbol and needs stronger SNR for reliable decoding."
  },
  {
    name: "QPSK",
    frequency: 2.2,
    phase: 0.8,
    description: "Quadrature Phase Shift Keying uses four phase states and is common in digital wireless links."
  },
  {
    name: "WBFM",
    frequency: 1.2,
    phase: 0.3,
    description: "Wideband FM represents information through frequency deviation and is common in broadcast-style signals."
  }
];

const snrLevels = Array.from({ length: 20 }, (_, index) => -20 + index * 2);
const modulationGrid = document.querySelector("#modulationGrid");
const selectedMod = document.querySelector("#selectedMod");
const selectedDescription = document.querySelector("#selectedDescription");
const snrStrip = document.querySelector("#snrStrip");
const canvas = document.querySelector("#signalCanvas");
const ctx = canvas.getContext("2d");

let activeModulation = modulations.find((item) => item.name === "QPSK") || modulations[0];

function drawGrid() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(1, 9, 12, 0.96)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(122, 245, 255, 0.1)";
  ctx.lineWidth = 1;
  for (let x = 40; x < canvas.width; x += 60) {
    ctx.beginPath();
    ctx.moveTo(x, 22);
    ctx.lineTo(x, canvas.height - 34);
    ctx.stroke();
  }
  for (let y = 36; y < canvas.height; y += 42) {
    ctx.beginPath();
    ctx.moveTo(34, y);
    ctx.lineTo(canvas.width - 22, y);
    ctx.stroke();
  }
}

function drawWave(color, offset, amplitude, phaseShift) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();

  for (let i = 0; i < 128; i += 1) {
    const x = 42 + (i / 127) * (canvas.width - 74);
    const t = (i / 127) * Math.PI * 2 * activeModulation.frequency;
    const noise = Math.sin(i * 0.41 + activeModulation.phase) * 4;
    const y = offset + Math.sin(t + phaseShift + activeModulation.phase) * amplitude + noise;

    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
}

function drawSignal() {
  drawGrid();
  drawWave("rgba(73, 224, 255, 0.96)", 104, 34, 0);
  drawWave("rgba(96, 240, 192, 0.92)", 176, 30, Math.PI / 2);

  ctx.fillStyle = "rgba(237, 249, 251, 0.9)";
  ctx.font = "700 13px Inter, sans-serif";
  ctx.fillText("I channel", 46, 34);
  ctx.fillStyle = "rgba(96, 240, 192, 0.92)";
  ctx.fillText("Q channel", 130, 34);
  ctx.fillStyle = "rgba(159, 183, 189, 0.9)";
  ctx.fillText(`${activeModulation.name} simulated 2 x 128 signal`, 46, canvas.height - 16);
}

function setActiveModulation(modulation) {
  activeModulation = modulation;
  selectedMod.textContent = modulation.name;
  selectedDescription.textContent = modulation.description;
  Array.from(modulationGrid.children).forEach((button) => {
    button.classList.toggle("active", button.dataset.name === modulation.name);
  });
  drawSignal();
}

function renderModulations() {
  modulationGrid.innerHTML = "";
  modulations.forEach((modulation) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.name = modulation.name;
    button.textContent = modulation.name;
    button.addEventListener("click", () => setActiveModulation(modulation));
    modulationGrid.appendChild(button);
  });
}

function renderSnrLevels() {
  snrStrip.innerHTML = "";
  snrLevels.forEach((level) => {
    const chip = document.createElement("span");
    chip.textContent = `${level} dB`;
    snrStrip.appendChild(chip);
  });
}

function wireContactForm() {
  const form = document.querySelector("#contactForm");
  form.addEventListener("submit", (event) => {
    event.preventDefault();

    const name = document.querySelector("#name").value.trim();
    const email = document.querySelector("#email").value.trim();
    const message = document.querySelector("#message").value.trim();
    const subject = encodeURIComponent("Wireless AI Analyzer Project Inquiry");
    const body = encodeURIComponent(`Name: ${name}\nEmail: ${email}\n\n${message}`);

    window.location.href = `mailto:manasgautam19@email.com?subject=${subject}&body=${body}`;
  });
}

renderModulations();
renderSnrLevels();
setActiveModulation(activeModulation);
wireContactForm();

window.addEventListener("resize", drawSignal);
