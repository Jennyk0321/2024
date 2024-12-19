import streamlit as st
from datetime import datetime, timedelta

# 오늘 날짜
today = datetime.today()
start_date = datetime(2024, 11, 1)
end_date = start_date + timedelta(days=365 * 10)

# 계절별 아이콘 설정
season_icons = {
    "봄": "🌸",
    "여름": "🌞",
    "가을": "🍂",
    "겨울": "❄️"
}

# 기분 선택 옵션
moods = ["😊 기쁨", "😐 보통", "😔 슬픔", "😠 화남", "😴 피곤함", "😍 설렘"]

# 메모장 초기화
if "daily_notes" not in st.session_state:
    st.session_state["daily_notes"] = {}

# 년도와 월 선택 옵션
selected_year = st.selectbox("연도를 선택하세요", list(range(2024, 2024 + 10)))
months = [
    "1월", "2월", "3월", "4월", "5월", "6월",
    "7월", "8월", "9월", "10월", "11월", "12월"
]
selected_month = st.selectbox("월을 선택하세요", months)
month_index = months.index(selected_month) + 1

# 계절 계산 함수
def get_season(month):
    if month in [3, 4, 5]:
        return "봄"
    elif month in [6, 7, 8]:
        return "여름"
    elif month in [9, 10, 11]:
        return "가을"
    else:
        return "겨울"

# 선택한 연도와 월의 메모장 표시
st.title(f"📅 {selected_year}년 {selected_month} 메모장 📅")
season_icon = season_icons[get_season(month_index)]
st.subheader(f"{season_icon} {selected_year}년 {selected_month}의 메모 {season_icon}")

# 해당 월의 날짜 계산
start_of_month = datetime(selected_year, month_index, 1)
days_in_month = (start_of_month.replace(month=start_of_month.month % 12 + 1, day=1) - timedelta(days=1)).day

# 월별 데일리 메모장
for day in range(1, days_in_month + 1):
    date = datetime(selected_year, month_index, day)
    date_key = date.strftime("%Y-%m-%d")

    # 기분 선택과 메모 입력
    st.write(f"### {date.strftime('%Y-%m-%d')} ({date.strftime('%A')})")
    mood = st.selectbox("오늘의 기분을 선택하세요", moods, key=f"mood_{date_key}")
    note = st.text_area(f"메모 작성 ({date.strftime('%Y-%m-%d')})", st.session_state["daily_notes"].get(date_key, ""), key=f"note_{date_key}")
    
    # 메모 저장
    if note:
        st.session_state["daily_notes"][date_key] = note
        st.write("메모가 저장되었습니다.")
    else:
        st.session_state["daily_notes"][date_key] = ""

st.write("### 월별 메모장 탐색")
st.info("상단에서 월을 선택하여 다른 달의 메모를 탐색할 수 있습니다.")
