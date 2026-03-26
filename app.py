import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import Model

st.set_page_config(page_title="Wireless AI Analyzer", layout="wide")

# ---- CSS for continuous metallic waves ----
st.markdown("""
<style>

/* Wave 1 */
.metal-wave1 {
    position: absolute;
    top: -100%;
    left: 0;
    width: 100%;
    height: 300%;
    background: linear-gradient(
        120deg,
        rgba(255,255,255,0.02) 0%,
        rgba(255,255,255,0.06) 50%,
        rgba(255,255,255,0.02) 100%
    );
    transform: skewY(-15deg);
    animation: waveMove1 18s linear infinite;
    z-index: -1;
}

/* Wave 2 */
.metal-wave2 {
    position: absolute;
    top: -100%;
    left: 0;
    width: 100%;
    height: 300%;
    background: linear-gradient(
        140deg,
        rgba(255,255,255,0.01) 0%,
        rgba(255,255,255,0.05) 50%,
        rgba(255,255,255,0.01) 100%
    );
    transform: skewY(-25deg);
    animation: waveMove2 25s linear infinite;
    z-index: -2;
}

/* Wave 3 */
.metal-wave3 {
    position: absolute;
    top: -100%;
    left: 0;
    width: 100%;
    height: 300%;
    background: linear-gradient(
        100deg,
        rgba(255,255,255,0.015) 0%,
        rgba(255,255,255,0.04) 50%,
        rgba(255,255,255,0.015) 100%
    );
    transform: skewY(-10deg);
    animation: waveMove3 35s linear infinite;
    z-index: -3;
}

/* Animations: continuous vertical movement */
@keyframes waveMove1 {
    0% { transform: translateY(-100%) skewY(-15deg); }
    100% { transform: translateY(100%) skewY(-15deg); }
}
@keyframes waveMove2 {
    0% { transform: translateY(-100%) skewY(-25deg); }
    100% { transform: translateY(100%) skewY(-25deg); }
}
@keyframes waveMove3 {
    0% { transform: translateY(-100%) skewY(-10deg); }
    100% { transform: translateY(100%) skewY(-10deg); }
}

/* Header */
.title-container {
    text-align: center;
    margin-top: 30px;
}

.title-line1 {
    font-size: 34px;
    letter-spacing: 2px;
    color: #bbbbbb;
}

.title-line2 {
    font-size: 60px;
    font-weight: 800;
    color: #ccc;
    text-transform: uppercase;
}

/* Glass panel */
.block-container {
    background: rgba(255,255,255,0.03);
    border-radius: 15px;
    padding: 20px;
    backdrop-filter: blur(10px);
}

/* Buttons */
.stButton>button {
    background-color: #222;
    color: white;
    border-radius: 10px;
    border: 1px solid #555;
}

</style>

<!-- Metallic strips -->
<div class="metal-wave1"></div>
<div class="metal-wave2"></div>
<div class="metal-wave3"></div>

<!-- Header -->
<div class="title-container">
    <div class="title-line1">WELCOME TO THE WORLD OF</div>
    <div class="title-line2">Aditya's Intelligence</div>
</div>

<hr>
""", unsafe_allow_html=True)

# ---- LABELS ----
modulation_labels = [
    '8PSK','AM-DSB','AM-SSB','BPSK','CPFSK',
    'GFSK','PAM4','QAM16','QAM64','QPSK','WBFM'
]

# ---- LOAD MODEL ----
model = Model(num_classes=11)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# ---- PREDICT FUNCTION ----
def predict(signal):
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    mod_out, snr_out = model(x)
    probs = torch.softmax(mod_out, dim=1).detach().numpy()[0]
    snr = snr_out.item()
    top3_idx = probs.argsort()[-3:][::-1]
    return probs, top3_idx, snr

# ---- LAYOUT ----
col1, col2 = st.columns([2, 1])

# ---- LEFT PANEL ----
with col1:
    st.subheader("📥 Input Signals")
    uploaded_files = st.file_uploader(
        "Upload .npy signals (2x128)",
        type=["npy"],
        accept_multiple_files=True
    )

    signals = []
    if uploaded_files:
        for file in uploaded_files:
            signal = np.load(file)
            signals.append((file.name, signal))
    else:
        if st.button("🎲 Generate Random Signal"):
            signal = np.random.randn(2, 128)
            signals.append(("Random Signal", signal))

    for name, signal in signals:
        st.markdown(f"### 📈 {name}")
        fig, ax = plt.subplots()
        ax.plot(signal[0], label="I Channel")
        ax.plot(signal[1], label="Q Channel")
        ax.legend()
        st.pyplot(fig)

# ---- RIGHT PANEL ----
with col2:
    st.subheader("📊 Predictions")
    if signals:
        for name, signal in signals:
            probs, top3_idx, snr = predict(signal)
            top1 = top3_idx[0]
            confidence = probs[top1]

            st.success(f"Modulation: {modulation_labels[top1]}")
            st.info(f"SNR: {snr:.2f} dB")
            
            st.markdown(f"Confidence: **{confidence*100:.2f}%**")
            st.progress(float(confidence))

            st.markdown("### 🧠 Top 3 Predictions")
            for i in top3_idx:
                st.write(f"{modulation_labels[i]} : {probs[i]*100:.2f}%")

            st.markdown("### 📊 Probability Distribution")
            fig, ax = plt.subplots()
            ax.bar(modulation_labels, probs)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.markdown("---")
    else:
        st.info("Upload or generate signals")