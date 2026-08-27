import streamlit as st


def apply_app_shell(page_title: str, subtitle: str, accent: str = "#0F9D58") -> None:
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

:root {{
    --accent: {accent};
    --accent-soft: rgba(15, 157, 88, 0.14);
    --panel: rgba(255, 255, 255, 0.72);
    --panel-border: rgba(15, 23, 42, 0.08);
    --text-main: #14213d;
    --text-muted: #5f6b7a;
    --shadow: 0 18px 45px rgba(20, 33, 61, 0.10);
}}

[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(circle at top left, rgba(15, 157, 88, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(41, 98, 255, 0.10), transparent 26%),
        linear-gradient(180deg, #f7fbf8 0%, #eef4ff 100%);
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #132238 0%, #1c3556 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}}

[data-testid="stSidebar"] * {{
    color: #f3f7fb;
}}

.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1180px;
}}

h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-main);
    letter-spacing: -0.02em;
}}

p, li, label, .stCaption {{
    color: var(--text-muted);
}}

.app-hero {{
    background: linear-gradient(135deg, rgba(11, 31, 58, 0.95), rgba(15, 157, 88, 0.92));
    border-radius: 28px;
    padding: 1.8rem 1.6rem;
    box-shadow: var(--shadow);
    color: white;
    margin-bottom: 1rem;
}}

.app-hero h1 {{
    color: white;
    margin: 0;
    font-size: 2.3rem;
}}

.app-hero p {{
    color: rgba(255, 255, 255, 0.86);
    margin-top: 0.75rem;
    margin-bottom: 0;
    font-size: 1rem;
}}

.app-card {{
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 22px;
    padding: 1.15rem 1.1rem;
    box-shadow: var(--shadow);
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}}

.app-card h3 {{
    margin-top: 0;
    margin-bottom: 0.6rem;
}}

.app-pill-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 0.7rem;
}}

.app-pill {{
    background: var(--accent-soft);
    color: var(--text-main);
    border: 1px solid rgba(15, 157, 88, 0.18);
    border-radius: 999px;
    padding: 0.35rem 0.7rem;
    font-size: 0.88rem;
    font-weight: 600;
}}

div[data-testid="stMetric"] {{
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 18px;
    padding: 0.85rem 0.9rem;
    box-shadow: 0 12px 30px rgba(20, 33, 61, 0.08);
}}

div[data-testid="stDataFrame"] {{
    background: rgba(255, 255, 255, 0.74);
    border-radius: 20px;
    padding: 0.35rem;
    border: 1px solid rgba(15, 23, 42, 0.08);
    box-shadow: var(--shadow);
}}

.stSelectbox > div > div,
.stTextInput > div > div,
.stRadio > div {{
    background: rgba(255, 255, 255, 0.8);
    border-radius: 16px;
}}

@media (max-width: 768px) {{
    .app-hero h1 {{
        font-size: 1.8rem;
    }}

    .block-container {{
        padding-top: 1.25rem;
    }}
}}
</style>
<section class="app-hero">
    <h1>{page_title}</h1>
    <p>{subtitle}</p>
</section>
""",
        unsafe_allow_html=True,
    )
