"""Minimal Streamlit learning coach: adaptive quiz and mastery dashboard."""

from __future__ import annotations

import os
import uuid

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 120
STATUS_LABELS = {"weak": "薄弱", "learning": "学习中", "mastered": "已掌握"}


def _request(method: str, path: str, **kwargs):
    response = requests.request(
        method,
        f"{API_BASE_URL}{path}",
        timeout=REQUEST_TIMEOUT,
        **kwargs,
    )
    response.raise_for_status()
    return response.json()


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


st.set_page_config(page_title="学习教练", page_icon="📚", layout="wide")
st.title("📚 有记忆的学习教练")
st.caption("按已有的薄弱知识点和到期复习记录自适应出题；答题结果会更新掌握度与复习计划。")

if "user_id" not in st.session_state:
    st.session_state.user_id = "default"
if "quiz_owner" not in st.session_state:
    st.session_state.quiz_owner = st.session_state.user_id
if "quiz_session_id" not in st.session_state:
    st.session_state.quiz_session_id = _new_session_id()
if "active_quiz" not in st.session_state:
    st.session_state.active_quiz = None
if "last_quiz_result" not in st.session_state:
    st.session_state.last_quiz_result = None

user_id = st.text_input("用户 ID", key="user_id", help="仅作为本地数据分区标识，不是登录凭据。")
if user_id != st.session_state.quiz_owner:
    st.session_state.quiz_owner = user_id
    st.session_state.quiz_session_id = _new_session_id()
    st.session_state.active_quiz = None
    st.session_state.last_quiz_result = None

try:
    dashboard = _request("GET", "/learning", params={"user_id": user_id})
except requests.RequestException as error:
    st.error(f"无法连接到 API（{API_BASE_URL}）：{error}")
    st.stop()

counts = dashboard["counts"]
weak_col, due_col, topic_col = st.columns(3)
weak_col.metric("薄弱知识点", counts["weak_topics"])
due_col.metric("到期复习", counts["due_reviews"])
topic_col.metric("已练习知识点", counts["practiced_topics"])

with st.expander("学习进度", expanded=True):
    topics = dashboard["topics"]
    if not topics:
        st.info("还没有答题记录。开始一道综合基础题后，知识点会出现在这里。")
    else:
        rows = [
            {
                "知识点": item["display_name"],
                "分类": item["category"],
                "掌握度": f"{item['mastery']:.0%}",
                "状态": STATUS_LABELS.get(item["status"], item["status"]),
                "答题次数": item["attempt_count"],
                "下次复习（UTC）": item["due_at"] or "—",
            }
            for item in topics
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

st.subheader("自适应练习")
topic_name = st.text_input("指定知识点（可留空，由系统选择）", placeholder="例如 python.generator")
if st.button("开始一道题", type="primary"):
    try:
        payload = _request(
            "POST",
            "/quiz/start",
            json={
                "user_id": user_id,
                "session_id": st.session_state.quiz_session_id,
                "topic_name": topic_name.strip() or None,
            },
        )
        st.session_state.active_quiz = payload["result"]
        st.session_state.last_quiz_result = None
    except requests.RequestException as error:
        st.error(f"开始答题失败：{error}")

active_quiz = st.session_state.active_quiz
if active_quiz:
    st.info(f"{active_quiz['reason']}\n\n**题目（{active_quiz['display_name']}）：** {active_quiz['question']}")
    with st.form("answer_form", clear_on_submit=True):
        answer = st.text_area("你的答案", height=120)
        submitted = st.form_submit_button("提交答案")
    if submitted:
        try:
            payload = _request(
                "POST",
                "/quiz/answer",
                json={
                    "user_id": user_id,
                    "session_id": st.session_state.quiz_session_id,
                    "answer": answer,
                },
            )
            result = payload["result"]
            st.session_state.last_quiz_result = result
            if result["status"] == "retry":
                st.session_state.active_quiz = result
            else:
                st.session_state.active_quiz = None
                st.rerun()
        except requests.RequestException as error:
            st.error(f"提交答案失败：{error}")

last_result = st.session_state.last_quiz_result
if last_result:
    if last_result["status"] == "retry":
        st.warning(f"首次回答需要补充。提示：{last_result['feedback']} 请再试一次。")
    elif last_result["status"] == "correct":
        st.success(f"回答正确。{last_result['feedback']}")
    elif last_result["status"] == "explained":
        st.warning(f"{last_result['verdict']}。完整解释：{last_result['explanation']}")
