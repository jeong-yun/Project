"""HTML 문자열을 화면 속 작은 창(iframe)에 그리기 — 설치된 Streamlit 에 있는 함수를 골라 쓴다.

Streamlit 1.66 은 `st.components.v1.html` 을 쓸 때마다 서버 기록에 '2026-06-01 뒤에 없어진다'는 경고를 남기고,
같은 일을 하는 `st.iframe` 을 쓰라고 한다. 1.43 에는 `st.iframe` 이 없다. 그래서 있으면 새 함수, 없으면 옛 함수를 쓴다.
두 함수 모두 같은 iframe 을 만든다 (스크립트 실행 · 부모 창 접근 허용 — 1.43.2·1.66.1 화면 코드에서 확인, 2026-11-12).
"""

import streamlit as st


def html_frame(html: str, height: int) -> None:
    """HTML 문자열(스크립트 포함)을 높이 height 픽셀의 iframe 에 그린다. 안쪽 내용이 넘쳐도 스크롤 막대를 기대하지 않는다.

    `st.iframe` 은 높이 0 을 받지 않아 1 로 올린다 (스크립트만 돌리는 보이지 않는 창용).
    앱이 만든 HTML 만 넣는다 — 부모 창에 접근할 수 있는 창이라 사용자·모델이 쓴 글을 그대로 넣으면 안 된다.
    """
    if hasattr(st, "iframe"):
        st.iframe(html, height=max(height, 1))
    else:
        import streamlit.components.v1 as components

        components.html(html, height=height)
