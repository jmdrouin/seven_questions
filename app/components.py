import plotly.express as px
import streamlit as st

def problem_solution(problem, solution):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="
                padding: 1.2rem;
                border-radius: 12px;
                background-color: #3A1E1E;
                border-left: 36px solid #B94A48;
            ">
            <strong>Problem</strong><br>
            <div>{problem}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="
                padding: 1.2rem;
                border-radius: 12px;
                background-color: #1E3A2A;
                border-left: 36px solid #4CAF50;
            ">
            <strong>Solution</strong><br>
            <div>{solution}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def section_title(text):
    st.markdown(
        f"""
        <span style="
            margin: 2.5rem 0 2rem -0.5em;
            padding: 0.5rem 1.5rem;
            border-radius: 20px;
            background-color: #AACCFF;
            border: 2px solid #000000; /* black */
            outline: 5px solid #6F9FD9;
            color: black;  
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        ">
            {text}
        </span>
        """,
        unsafe_allow_html=True,
    )

def big_divider():
    st.markdown(
        """
        <hr style="
            border: none;
            border-bottom: 5px solid black;
            height: 8px;
            background-color: #888;
            margin: 4rem 0;
            width: 100vw;
            position: relative;
            left: 50%;
            transform: translateX(-50%);
            border-radius: 3px;
        ">
        """,
        unsafe_allow_html=True,
    )

def table(items):
    st.markdown(f"""
        <div class="big-table">
        {items}
        </div>
        """, unsafe_allow_html=True)

def blob(text):
    return f"""<span style="
        display: inline-block;
        padding: 0.35rem 0.75rem;
        margin: 0.25rem 0.35rem 0 0;
        border-radius: 999px;
        background: #2B2F3A;
        color: #F9FAFB;
        font-size: 1.05rem;
        border: 1px solid #555;
        ">{text}</span>"""

def pill_box(title, items, accent="#6F9FD9"):
    pills_html = "".join(blob(it) for it in items)
    st.markdown(
        f"""
        <div style="
            padding: 1.4rem 1.6rem;
            border-radius: 16px;
            background-color: #111827;
            color: white;
            border: 3px solid {accent};
            box-shadow: 0 0 0 1px #000;
            ">
            <div style="
                font-size: 1.2rem;
                color: #BBBBBB;
                font-weight: 600;
                margin-bottom: 0.8rem;
                ">
                {title}
            </div>
            <div>{pills_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def phone_mockup():
    inner = """
        <div style="text-align:center; font-weight:800; font-size:1.1rem; margin-bottom:1rem;">
        What will you watch tonight?
        </div>
        <!-- Slider 1 -->
        <div style="margin-bottom:1.2rem;">
        <div style="display:flex; justify-content:space-between; font-size:0.8rem; opacity:0.85;">
            <span>Arthouse</span>
            <span>Blockbuster</span>
        </div>
        <div style="
            height: 8px;
            background: #444;
            border-radius: 4px;
            position: relative;
            margin-top: 6px;
        ">
            <div style="
            position: absolute;
            left: 30%;
            width: 14px;
            height: 14px;
            background: #6F9FD9;
            border-radius: 50%;
            top: -3px;
            box-shadow: 0 0 0 2px #111;
            "></div>
        </div>
        </div>
        <!-- Slider 2 -->
        <div style="margin-bottom:1.2rem;">
        <div style="display:flex; justify-content:space-between; font-size:0.8rem; opacity:0.85;">
            <span>Drama</span>
            <span>Comedy</span>
        </div>
        <div style="
            height: 8px;
            background: #444;
            border-radius: 4px;
            position: relative;
            margin-top: 6px;
        ">
            <div style="
            position: absolute;
            left: 55%;
            width: 14px;
            height: 14px;
            background: #6F9FD9;
            border-radius: 50%;
            top: -3px;
            box-shadow: 0 0 0 2px #111;
            "></div>
        </div>
        </div>
        <!-- Slider 3 -->
        <div style="margin-bottom:1.2rem;">
        <div style="display:flex; justify-content:space-between; font-size:0.8rem; opacity:0.85;">
            <span>Family-friendly</span>
            <span>Deep & complex</span>
        </div>
        <div style="
            height: 8px;
            background: #444;
            border-radius: 4px;
            position: relative;
            margin-top: 6px;
        ">
            <div style="
            position: absolute;
            left: 20%;
            width: 14px;
            height: 14px;
            background: #6F9FD9;
            border-radius: 50%;
            top: -3px;
            box-shadow: 0 0 0 2px #111;
            "></div>
        </div>
        </div>
        <div style="
        margin-top: 1.2rem;
        text-align: center;
        padding: 0.6rem;
        border-radius: 12px;
        background: #1F2937;
        font-weight: 700;
        font-size: 0.9rem;
        ">
        🎬 Show recommendations
        </div>
    """

    st.markdown(
        f"""
        <div style="
            width: 280px;
            margin: 2rem auto;
            padding: 16px 14px 20px 14px;
            border-radius: 36px;
            background: #999;
            box-shadow:
                0 20px 40px rgba(0,0,0,0.6),
                inset 0 0 0 2px #333;
        ">
        <!-- Screen -->
            <div style="
                background: #0E1117;
                border-radius: 24px;
                padding: 14px;
                height: 480px;
                box-shadow: inset 0 0 0 1px #222;
                color: white;
                font-size: 0.9rem;
            ">{inner}</div>
            <!-- Home button -->
            <div style="
                margin: 14px auto 0 auto;
                width: 42px;
                height: 42px;
                border-radius: 50%;
                background: #0a0a0a;
                box-shadow:
                    inset 0 0 0 2px #333,
                    0 0 0 1px #000;
            ">
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def pills(texts, color = "#110011"):
    style = f"display:inline-block;border:1px solid white;padding:4px 8px;margin:2px;border-radius:12px;background:{color};font-size:0.85em;"
    st.markdown(
        "".join(f"<span style='{style}'>{text}</span>" for text in texts),
        unsafe_allow_html=True
    )