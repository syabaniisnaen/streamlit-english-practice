#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aplikasi Latihan Speaking - Streamlit
Dengan DeepSeek API untuk generate kalimat
Versi cloud-ready (menggunakan st.audio_input, bukan sounddevice)
Alur: Listening → Writing (check) → Speaking
"""

import os
import json
import difflib
import tempfile
import base64
import requests
from datetime import datetime
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== KONFIGURASI ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
AUDIO_DIR = os.path.join(BASE_DIR, 'audio')
HISTORY_FILE = os.path.join(DATA_DIR, 'history.json')

# Buat folder
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# DeepSeek API - utamakan st.secrets (Streamlit Cloud), fallback ke .env (lokal)
try:
    DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY"))
except Exception:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# ==================== FUNGSI GENERATE KALIMAT ====================

def generate_sentence_with_deepseek(difficulty="easy"):
    """
    Generate kalimat menggunakan DeepSeek API
    difficulty: "very_easy" = 2-4 kata, "easy" = 10 kata,
                "moderate" = 15 kata, "difficult" = 20 kata
    """
    word_counts = {
        "very_easy": "2 to 4",
        "easy": "10",
        "moderate": "15",
        "difficult": "20",
    }
    target_words = word_counts.get(difficulty, "10")

    prompt = f"""Generate exactly one English sentence with exactly {target_words} words.

Requirements:
- Grammatically correct
- Natural and common in daily conversation
- Not too simple, not too complex
- Suitable for speaking practice

Respond with ONLY the sentence, nothing else.
No quotes, no numbering, no extra text.

Example format: "The quick brown fox jumps over the lazy dog and runs away"
"""

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates English sentences for language learning. You always respond with exactly one sentence and nothing else."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.8,
            "max_tokens": 150
        }

        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=15)

        if response.status_code == 200:
            result = response.json()
            sentence = result['choices'][0]['message']['content'].strip()

            # Validasi jumlah kata
            word_count = len(sentence.split())
            if difficulty == "very_easy":
                valid = 2 <= word_count <= 4
            else:
                valid = word_count == int(target_words)

            if not valid:
                for _ in range(3):
                    sentence = generate_sentence_with_deepseek(difficulty)
                    wc = len(sentence.split())
                    if difficulty == "very_easy":
                        if 2 <= wc <= 4:
                            return sentence
                    else:
                        if wc == int(target_words):
                            return sentence
                return get_fallback_sentence(difficulty)

            return sentence
        else:
            st.warning(f"⚠️ API Error: {response.status_code}. Menggunakan kalimat default.")
            return get_fallback_sentence(difficulty)

    except Exception as e:
        st.warning(f"⚠️ Error: {e}. Menggunakan kalimat default.")
        return get_fallback_sentence(difficulty)


def get_fallback_sentence(difficulty):
    """Kalimat fallback jika API gagal (berdasarkan difficulty key)"""
    fallback = {
        "very_easy": "I like coffee",
        "easy": "The quick brown fox jumps over the lazy dog",
        "moderate": "I enjoy learning new languages every single day",
        "difficult": "The weather today is very beautiful and sunny outside"
    }
    return fallback.get(difficulty, "Practice makes perfect in any skill")

# ==================== FUNGSI AUDIO ====================

def text_to_audio_base64(text):
    """Ubah teks ke audio (base64 untuk streamlit)"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
            tts.save(tmp.name)
            tmp_path = tmp.name

        with open(tmp_path, 'rb') as f:
            audio_bytes = f.read()

        os.unlink(tmp_path)
        return base64.b64encode(audio_bytes).decode()
    except Exception as e:
        st.error(f"❌ Error TTS: {e}")
        return None


def audio_to_text(audio_file):
    """Ubah audio ke teks"""
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio, language='en-US')
        return text
    except sr.UnknownValueError:
        return "[Tidak terdeteksi]"
    except sr.RequestError:
        return "[Error API]"
    except Exception as e:
        return f"[Error: {e}]"

# ==================== FUNGSI EVALUASI ====================

def evaluate_word_by_word(original, user_spoken):
    """Evaluasi kata per kata dengan detail"""

    if user_spoken.startswith("[") and user_spoken.endswith("]"):
        return [], 0, "❌ Suara tidak jelas atau terjadi error", "❌"

    orig_words = original.lower().split()
    user_words = user_spoken.lower().split()

    word_results = []
    correct_count = 0

    for i, orig_word in enumerate(orig_words):
        if i < len(user_words):
            user_word = user_words[i]
            if orig_word == user_word:
                is_correct = True
                similarity = 100
                correct_count += 1
            else:
                similarity = difflib.SequenceMatcher(None, orig_word, user_word).ratio()
                is_correct = similarity >= 0.8
                if is_correct:
                    correct_count += 1

            word_results.append({
                "original": orig_word,
                "user": user_word,
                "correct": is_correct,
                "similarity": round(similarity * 100, 2)
            })
        else:
            word_results.append({
                "original": orig_word,
                "user": "[tidak diucapkan]",
                "correct": False,
                "similarity": 0
            })

    total_words = len(orig_words)
    score = round((correct_count / total_words) * 100, 2) if total_words > 0 else 0

    if score >= 90:
        feedback = "🌟 Luar biasa! Semua kata diucapkan dengan sangat baik!"
        emoji = "🌟"
    elif score >= 75:
        feedback = "👍 Bagus! Sebagian besar kata sudah benar."
        emoji = "👍"
    elif score >= 60:
        feedback = "📖 Cukup baik. Beberapa kata perlu diperbaiki."
        emoji = "📖"
    elif score >= 40:
        feedback = "🔊 Perlu latihan lagi. Perhatikan pengucapan setiap kata."
        emoji = "🔊"
    else:
        feedback = "📚 Dengarkan audio contoh dan ulangi perlahan."
        emoji = "📚"

    return word_results, score, feedback, emoji

# ==================== FUNGSI RIWAYAT ====================

def save_history(original, user_spoken, score, feedback, word_results):
    """Simpan riwayat latihan"""
    history = load_history()

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original": original,
        "user_spoken": user_spoken,
        "score": score,
        "feedback": feedback,
        "word_results": word_results[:10]
    }

    history.append(entry)

    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)


def load_history():
    """Load riwayat latihan"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def get_statistics():
    """Dapatkan statistik"""
    history = load_history()
    if not history:
        return None

    scores = [h['score'] for h in history if isinstance(h['score'], (int, float))]

    return {
        "total": len(history),
        "avg": round(sum(scores) / len(scores), 2) if scores else 0,
        "max": max(scores) if scores else 0,
        "min": min(scores) if scores else 0
    }

# ==================== STREAMLIT UI ====================

def main():
    st.set_page_config(
        page_title="Speaking Practice",
        page_icon="🎯",
        layout="wide"
    )

    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1E88E5;
        padding: 0.5rem;
        font-size: 2.2rem;
    }
    .sentence-box {
        background: #e3f2fd;
        padding: 20px 25px;
        border-radius: 12px;
        border-left: 6px solid #1E88E5;
        margin: 10px 0;
    }
    .sentence-text {
        font-size: 1.4rem;
        font-weight: 600;
        color: #0d47a1;
        margin: 5px 0;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.8rem;
        border-radius: 20px;
        color: white;
        text-align: center;
    }
    .score-number {
        font-size: 4rem;
        font-weight: bold;
        margin: 0;
    }
    .feedback-card {
        background: #f5f5f5;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 10px 0;
    }
    .word-box {
        padding: 8px 12px;
        border-radius: 6px;
        margin: 4px 0;
        border: 1px solid #ccc;
        font-weight: 500;
        text-align: center;
        font-size: 0.9rem;
    }
    .word-correct {
        background: #c8e6c9;
        color: #1b5e20;
        border-color: #81c784;
    }
    .word-incorrect {
        background: #ffcdd2;
        color: #b71c1c;
        border-color: #e57373;
    }
    .word-missing {
        background: #fff3e0;
        color: #e65100;
        border-color: #ffb74d;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("🎯 Speaking Practice")
        st.markdown("---")

        menu = st.radio(
            "Menu",
            ["🏠 Home", "🎯 Latihan", "📊 Riwayat", "ℹ️ Tentang"],
            index=0
        )

        st.markdown("---")

        stats = get_statistics()
        if stats:
            st.caption(f"📊 Total latihan: {stats['total']}")
            st.caption(f"⭐ Rata-rata: {stats['avg']}/100")
            st.caption(f"🏆 Tertinggi: {stats['max']}/100")
        else:
            st.caption("📊 Belum ada latihan")

        st.markdown("---")

        api_status = "✅" if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" else "❌"
        st.caption(f"DeepSeek API: {api_status}")

        if api_status == "❌":
            st.warning("⚠️ Setup API Key di .env atau Streamlit Secrets")

    if menu == "🏠 Home":
        home_page()
    elif menu == "🎯 Latihan":
        practice_page()
    elif menu == "📊 Riwayat":
        history_page()
    elif menu == "ℹ️ Tentang":
        about_page()


def home_page():
    st.markdown('<h1 class="main-header">🎯 Selamat Datang di Speaking Practice</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    stats = get_statistics()
    if stats:
        with col1:
            st.metric("📚 Total Latihan", stats["total"])
        with col2:
            st.metric("⭐ Rata-rata Skor", f"{stats['avg']}/100")
        with col3:
            st.metric("🏆 Skor Tertinggi", f"{stats['max']}/100")
    else:
        with col1:
            st.metric("📚 Total Latihan", "0")
        with col2:
            st.metric("⭐ Rata-rata Skor", "-")
        with col3:
            st.metric("🏆 Skor Tertinggi", "-")

    st.markdown("---")

    st.subheader("🚀 Cara Penggunaan")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **1️⃣ Pilih Tingkat**
        - Very Easy (2-4 kata)
        - Easy (10 kata)
        - Moderate (15 kata)
        - Difficult (20 kata)
        - Klik Generate
        """)

    with col2:
        st.markdown("""
        **2️⃣ Dengarkan & Tulis**
        - Audio contoh otomatis
        - Tulis kalimat yang didengar
        - Klik Cek Jawaban
        """)

    with col3:
        st.markdown("""
        **3️⃣ Speaking**
        - Klik mikrofon di Step 3
        - Ucapkan kalimat
        - Lihat hasil per kata
        """)

    st.markdown("---")
    st.info("💡 **Tips:** Gunakan headset untuk hasil rekaman yang lebih baik!")

    if st.button("🎯 Mulai Latihan Sekarang!", use_container_width=True):
        st.session_state.menu = "🎯 Latihan"
        st.rerun()


def practice_page():
    st.markdown('<h1 class="main-header">🎯 Sesi Latihan</h1>', unsafe_allow_html=True)

    # ===== INISIALISASI SESSION STATE =====
    defaults = {
        "sentence": None,
        "audio_base64": None,
        "difficulty": "easy",
        "listening_checked": False,
        "listening_result": None,   # {user_text, score, feedback, emoji, word_results}
        "speaking_result": None,    # {user_text, score, feedback, emoji, word_results}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ===== STEP 1: PILIH TINGKAT & GENERATE =====
    st.subheader("📝 Step 1: Pilih Tingkat Kesulitan")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        difficulty = st.radio(
            "Pilih tingkat:",
            [
                "Very Easy (2-4 kata)",
                "Easy (10 kata)",
                "Moderate (15 kata)",
                "Difficult (20 kata)",
            ],
            index=1,  # default ke Easy
            horizontal=True,
            key="difficulty_radio",
        )
        diff_map = {
            "Very Easy (2-4 kata)": "very_easy",
            "Easy (10 kata)": "easy",
            "Moderate (15 kata)": "moderate",
            "Difficult (20 kata)": "difficult",
        }
        difficulty_key = diff_map[difficulty]
        word_count_map = {
            "Very Easy (2-4 kata)": "2-4",
            "Easy (10 kata)": 10,
            "Moderate (15 kata)": 15,
            "Difficult (20 kata)": 20,
        }
        target_words = word_count_map[difficulty]

    with col2:
        st.markdown(f"""
        <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; text-align: center;">
            <p style="margin: 0; font-size: 0.9rem; color: #666;">Jumlah kata</p>
            <p style="margin: 0; font-size: 1.5rem; font-weight: bold; color: #1E88E5;">{target_words}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if st.button("🔄 Generate Kalimat", use_container_width=True):
            with st.spinner("🤖 AI sedang membuat kalimat..."):
                sentence = generate_sentence_with_deepseek(difficulty_key)
                if sentence:
                    st.session_state.sentence = sentence
                    st.session_state.audio_base64 = text_to_audio_base64(sentence)
                    st.session_state.difficulty = difficulty_key
                    # Reset hasil tahap berikutnya
                    st.session_state.listening_checked = False
                    st.session_state.listening_result = None
                    st.session_state.speaking_result = None
                    st.success("✅ Kalimat siap! Silakan dengarkan audio di bawah.")
                    st.rerun()
                else:
                    st.error("❌ Gagal generate kalimat. Coba lagi.")

    st.markdown("---")

    # ===== JIKA BELUM ADA KALIMAT =====
    if not st.session_state.sentence:
        st.info("👆 Silakan generate kalimat terlebih dahulu di Step 1.")
        return

    # ===== STEP 2: LISTENING + WRITING =====
    st.subheader("🎧 Step 2: Dengarkan & Tulis")

    # Audio contoh (kalimat TIDAK ditampilkan)
    if st.session_state.audio_base64:
        audio_html = f'''
        <div style="margin: 15px 0;">
            <p>🔊 <strong>Audio contoh:</strong> (dengarkan berulang kali, kalimat disembunyikan)</p>
            <audio controls style="width: 100%; max-width: 500px;">
                <source src="data:audio/mp3;base64,{st.session_state.audio_base64}" type="audio/mpeg">
            </audio>
        </div>
        '''
        st.markdown(audio_html, unsafe_allow_html=True)

    st.markdown("**✍️ Tulis kalimat yang Anda dengar:**")

    user_written = st.text_input(
        "Ketik hasil listening Anda:",
        key="listening_input",
        label_visibility="collapsed",
        placeholder="Tulis kalimat yang Anda dengar di sini...",
        disabled=st.session_state.listening_checked,
    )

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        check_btn = st.button(
            "✅ Cek Jawaban",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.listening_checked or not user_written.strip(),
        )
    if check_btn:
        word_results, score, feedback, emoji = evaluate_word_by_word(
            st.session_state.sentence, user_written
        )
        st.session_state.listening_result = {
            "user_text": user_written,
            "score": score,
            "feedback": feedback,
            "emoji": emoji,
            "word_results": word_results,
        }
        st.session_state.listening_checked = True
        save_history(
            st.session_state.sentence,
            user_written,
            score,
            f"[LISTENING] {feedback}",
            word_results,
        )
        st.rerun()

    # ===== HASIL LISTENING =====
    if st.session_state.listening_checked and st.session_state.listening_result:
        res = st.session_state.listening_result

        # Tampilkan kalimat asli SETELAH dicek
        st.markdown(f"""
        <div class="sentence-box">
            <p style="margin: 0; color: #666; font-size: 0.9rem;">Kalimat asli:</p>
            <p class="sentence-text">"{st.session_state.sentence}"</p>
            <p style="margin: 0; color: #666; font-size: 0.85rem;">
                📊 {len(st.session_state.sentence.split())} kata
                <span style="margin-left: 20px;">📌 {st.session_state.difficulty.replace('_', ' ').title()}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div class="score-card">
                <p style="font-size: 1.2rem; margin: 0;">{res['emoji']}</p>
                <p class="score-number">{res['score']}</p>
                <p style="margin: 0; opacity: 0.8;">/100</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="feedback-card">
                <h4 style="margin: 0 0 8px 0;">📢 Hasil Listening:</h4>
                <p style="font-size: 1.05rem; margin: 0;">{res['feedback']}</p>
                <hr style="margin: 12px 0;">
                <p style="color: #666; font-style: italic; margin: 0;">
                    📝 Yang Anda tulis:<br>
                    <span style="font-weight: 500; color: #333;">"{res['user_text']}"</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Analisis kata per kata
        st.markdown("### 📝 Analisis Kata per Kata (Listening)")
        word_results = res["word_results"]
        if word_results:
            num_cols = min(len(word_results), 4)
            cols = st.columns(num_cols)
            for idx, word in enumerate(word_results):
                with cols[idx % num_cols]:
                    if word["correct"]:
                        sim_display = "100" if word["similarity"] >= 100 else f"{word['similarity']:.0f}"
                        st.markdown(f"""
                        <div class="word-box word-correct">
                            ✅ {word["original"]}
                            <br><small style="font-size: 0.65rem; opacity: 0.7;">({sim_display}%)</small>
                        </div>
                        """, unsafe_allow_html=True)
                    elif word["user"] == "[tidak diucapkan]":
                        st.markdown(f"""
                        <div class="word-box word-missing">
                            ❌ {word["original"]}
                            <br><small style="font-size: 0.65rem; opacity: 0.7;">(0%)</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        sim_display = "100" if word["similarity"] >= 100 else f"{word['similarity']:.0f}"
                        st.markdown(f"""
                        <div class="word-box word-incorrect">
                            ⚠️ {word["original"]}
                            <br><small style="font-size: 0.65rem; opacity: 0.7;">({sim_display}%)</small>
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown("---")

        # ===== STEP 3: SPEAKING (baru muncul setelah listening checked) =====
        st.subheader("🎤 Step 3: Latihan Speaking")
        st.info("📌 Klik tombol mikrofon di bawah, ucapkan kalimat sesuai audio contoh, lalu klik **Stop**.")

        audio_value = st.audio_input("Rekam suara Anda di sini", key="speaking_audio")

        if audio_value is not None:
            st.audio(audio_value, format="audio/wav")
            with st.spinner("🔄 Memproses suara..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_value.getbuffer())
                    tmp_path = tmp.name

                user_text = audio_to_text(tmp_path)

                word_results_sp, score_sp, feedback_sp, emoji_sp = evaluate_word_by_word(
                    st.session_state.sentence, user_text
                )
                st.session_state.speaking_result = {
                    "user_text": user_text,
                    "score": score_sp,
                    "feedback": feedback_sp,
                    "emoji": emoji_sp,
                    "word_results": word_results_sp,
                }
                save_history(
                    st.session_state.sentence,
                    user_text,
                    score_sp,
                    f"[SPEAKING] {feedback_sp}",
                    word_results_sp,
                )
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                st.rerun()

        # ===== HASIL SPEAKING =====
        if st.session_state.speaking_result:
            res_sp = st.session_state.speaking_result
            st.markdown("### 📊 Hasil Speaking")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                <div class="score-card">
                    <p style="font-size: 1.2rem; margin: 0;">{res_sp['emoji']}</p>
                    <p class="score-number">{res_sp['score']}</p>
                    <p style="margin: 0; opacity: 0.8;">/100</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="feedback-card">
                    <h4 style="margin: 0 0 8px 0;">📢 Hasil Speaking:</h4>
                    <p style="font-size: 1.05rem; margin: 0;">{res_sp['feedback']}</p>
                    <hr style="margin: 12px 0;">
                    <p style="color: #666; font-style: italic; margin: 0;">
                        🎤 Yang Anda ucapkan:<br>
                        <span style="font-weight: 500; color: #333;">"{res_sp['user_text']}"</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Analisis per kata speaking
            st.markdown("**Analisis kata per kata (Speaking):**")
            wr_sp = res_sp["word_results"]
            if wr_sp:
                num_cols = min(len(wr_sp), 4)
                cols = st.columns(num_cols)
                for idx, word in enumerate(wr_sp):
                    with cols[idx % num_cols]:
                        if word["correct"]:
                            sim_display = "100" if word["similarity"] >= 100 else f"{word['similarity']:.0f}"
                            st.markdown(f"""
                            <div class="word-box word-correct">
                                ✅ {word["original"]}
                                <br><small style="font-size: 0.65rem; opacity: 0.7;">({sim_display}%)</small>
                            </div>
                            """, unsafe_allow_html=True)
                        elif word["user"] == "[tidak diucapkan]":
                            st.markdown(f"""
                            <div class="word-box word-missing">
                                ❌ {word["original"]}
                                <br><small style="font-size: 0.65rem; opacity: 0.7;">(0%)</small>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            sim_display = "100" if word["similarity"] >= 100 else f"{word['similarity']:.0f}"
                            st.markdown(f"""
                            <div class="word-box word-incorrect">
                                ⚠️ {word["original"]}
                                <br><small style="font-size: 0.65rem; opacity: 0.7;">({sim_display}%)</small>
                            </div>
                            """, unsafe_allow_html=True)

        st.markdown("---")

        # ===== TOMBOL LATIHAN LAGI =====
        st.subheader("🔁 Selesai Satu Siklus")
        if st.button("🔄 Latihan Lagi dengan Kalimat Baru", use_container_width=True, type="primary"):
            st.session_state.sentence = None
            st.session_state.audio_base64 = None
            st.session_state.listening_checked = False
            st.session_state.listening_result = None
            st.session_state.speaking_result = None
            # Hapus key widget input agar tidak carry-over
            for k in ["listening_input", "speaking_audio"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()


def history_page():
    st.markdown('<h1 class="main-header">📊 Riwayat Latihan</h1>', unsafe_allow_html=True)

    history = load_history()
    stats = get_statistics()

    if not history:
        st.info("📊 Belum ada riwayat latihan.")
        if st.button("🎯 Mulai Latihan"):
            st.session_state.menu = "🎯 Latihan"
            st.rerun()
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📚 Total", stats["total"])
    with col2:
        st.metric("⭐ Rata-rata", f"{stats['avg']}/100")
    with col3:
        st.metric("🏆 Tertinggi", f"{stats['max']}/100")
    with col4:
        st.metric("📉 Terendah", f"{stats['min']}/100")

    st.markdown("---")

    st.subheader("📈 Progress Skor")
    scores = [h['score'] for h in history[-20:]]
    if scores:
        chart_data = {
            "Latihan": [f"#{i+1}" for i in range(len(scores))],
            "Skor": scores
        }
        st.bar_chart(chart_data, x="Latihan", y="Skor")

    st.markdown("---")

    st.subheader("📚 Daftar Latihan")

    for i, entry in enumerate(reversed(history[-30:]), 1):
        with st.expander(f"Latihan #{len(history)-i+1} - {entry['timestamp']}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                **📝 Kalimat:** {entry['original']}  
                **🎤 Ucapan Anda:** {entry['user_spoken']}
                """)
                if 'word_results' in entry and entry['word_results']:
                    words_html = ' '.join([
                        f'<span style="color: {"green" if w["correct"] else "red"};">{w["original"]}</span>'
                        for w in entry['word_results'][:10]
                    ])
                    st.markdown(f"**📊 Detail:** {words_html}", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                **⭐ Skor:** {entry['score']}/100  
                **📢 {entry['feedback'][:30]}...**
                """)


def about_page():
    st.markdown('<h1 class="main-header">ℹ️ Tentang Aplikasi</h1>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Speaking Practice App

    Aplikasi latihan listening & speaking bahasa Inggris berbasis **Streamlit** dengan **DeepSeek AI**.

    #### ✨ Fitur:
    - 🤖 **AI-Powered** - Generate kalimat random dengan DeepSeek API
    - 🎯 **4 Tingkat Kesulitan** - Very Easy (2-4 kata), Easy (10 kata), Moderate (15 kata), Difficult (20 kata)
    - 🎧 **Listening Practice** - Dengarkan audio, tulis apa yang didengar, dapatkan skor
    - 🎤 **Speaking Practice** - Rekam suara, cocokkan dengan kalimat asli
    - 📝 **Analisis Per Kata** - Evaluasi setiap kata dengan warna (✅ ⚠️ ❌)
    - ⭐ **Penilaian Otomatis** - Skor dan feedback detail untuk listening & speaking
    - 💾 **Riwayat Tersimpan** - Data tetap ada walau refresh

    #### 🛠️ Teknologi:
    - Python 3.12
    - Streamlit (`st.audio_input` untuk rekaman via browser)
    - DeepSeek API
    - gTTS (Text-to-Speech)
    - SpeechRecognition (Speech-to-Text)
    - JSON (Database)

    #### 🔑 Setup API Key:
    1. Daftar di https://platform.deepseek.com/
    2. Buat API Key
    3. Lokal: masukkan di file `.env` → `DEEPSEEK_API_KEY=sk-...`
    4. Cloud: masukkan di **Streamlit Secrets** dengan format TOML

    ---
    **Made with ❤️ using Streamlit & DeepSeek AI**
    """)


if __name__ == "__main__":
    main()