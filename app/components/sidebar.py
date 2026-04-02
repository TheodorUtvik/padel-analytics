import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.divider()
        st.markdown(
            """
            <div style='font-size: 0.78rem; color: #888; line-height: 1.7'>
                <b style='color: #ccc'>Pages</b><br>
                — &nbsp;ELO Rankings<br>
                — &nbsp;Match Predictor<br>
                — &nbsp;Player Profile<br>
                — &nbsp;Tournament History<br>
                — &nbsp;Head to Head<br>
                — &nbsp;ELO Race<br>
                — &nbsp;Pair Chemistry
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            """
            <div style='font-size: 0.75rem; color: #666; line-height: 1.8'>
                <b style='color: #999'>About</b><br>
                Data collected from padelapi.org<br>
                891 labeled matches (free tier)<br>
                XGBoost · 77.1% accuracy<br>
                ELO · Rolling form · Pair chemistry<br><br>
                <b style='color: #999'>Author</b><br>
                Theodor Sjetnan Utvik<br>
                <a href='https://www.linkedin.com/in/theodorutvik/' target='_blank'
                   style='color: #4a9eda; text-decoration: none'>LinkedIn</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
