import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data.json"


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def fetch_html(url: str) -> BeautifulSoup:
    """Tải trang và trả về BeautifulSoup"""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


# ----------- Trang menuno=89 (APEC chung) ----------

def parse_apec_overview(soup: BeautifulSoup) -> dict:
    """Trích xuất thông tin tổng quan APEC"""
    content_div = soup.find("div", class_="contents")
    if not content_div:
        return {}

    data: dict = {}

    # Tiêu đề H2
    h2 = content_div.find("h2")
    if h2:
        data["title"] = h2.get_text(strip=True)

    # Đoạn giới thiệu (p đầu tiên sau .about_apec_wrap)
    about_wrap = content_div.find("div", class_="about_apec_wrap")
    if about_wrap:
        p = about_wrap.find_next("p")
        if p:
            data["introduction"] = " ".join(p.stripped_strings)

    # Mission, Vision
    slides = content_div.select("div.new_swiper")
    for slide in slides:
        h3 = slide.find("h3")
        para = slide.find("p")
        if not h3 or not para:
            continue
        key = h3.get_text(strip=True).lower()
        data[key] = " ".join(para.stripped_strings)

    # Thành viên
    members = content_div.select("ul.apec_member li strong")
    data["member_economies"] = [m.get_text(strip=True) for m in members]

    return data


# ----------- Trang menuno=90 (APEC 2025 Korea) ----------

def parse_apec2025_overview(soup: BeautifulSoup) -> dict:
    """Trích xuất thông tin APEC 2025 Korea"""
    content_div = soup.find("div", class_="contents")
    if not content_div:
        return {}

    data: dict = {}
    h2 = content_div.find("h2")
    if h2:
        data["title"] = h2.get_text(strip=True)

    # Phần Overview
    overview_section = content_div.find("div", class_="overview_text")
    if overview_section:
        items = overview_section.select("ul.overview_text_inner li")
        for li in items:
            strong = li.find("strong")
            em = li.find("em")
            if strong and em:
                key = strong.get_text(strip=True).rstrip(":").lower()
                data[key] = em.get_text(strip=True)

    # Đoạn Korea and APEC
    korea_section_header = content_div.find("h3", string=re.compile(r"Korea and APEC", re.I))
    if korea_section_header:
        p = korea_section_header.find_next("p")
        if p:
            data["korea_and_apec"] = " ".join(p.stripped_strings)

    return data


# ----------- Trang menuno=93 (Danh sách meetings) ----------

def parse_meetings(soup: BeautifulSoup) -> list:
    """Trả về danh sách các cuộc họp"""
    meetings: list = []
    table = soup.find("table")
    if not table:
        return meetings

    rows = table.find_all("tr")
    for row in rows[1:]:  # bỏ header
        cols = row.find_all("td")
        if len(cols) < 4:
            continue
        no = cols[0].get_text(strip=True)
        title = " ".join(cols[1].stripped_strings)
        date = cols[2].get_text(strip=True)
        venue = cols[3].get_text(strip=True)
        meetings.append({
            "no": no,
            "event_title": title,
            "date": date,
            "venue": venue,
        })
    return meetings


# ----------- Hàm chính ----------

def main():
    # URLs cần crawl
    urls = {
        "apec_info": "https://apec2025.kr/?menuno=89",
        "apec2025": "https://apec2025.kr/?menuno=90",
        "meetings": "https://apec2025.kr/?menuno=93",
    }

    print("Đang tải dữ liệu từ website …")
    soup_info = fetch_html(urls["apec_info"])
    soup_2025 = fetch_html(urls["apec2025"])
    soup_meetings = fetch_html(urls["meetings"])

    print("Đang phân tích …")
    apec_info_data = parse_apec_overview(soup_info)
    apec2025_data = parse_apec2025_overview(soup_2025)
    meetings_data = parse_meetings(soup_meetings)

    # Kết hợp với data.json hiện tại
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}

    existing["apec_general_info"] = apec_info_data
    existing["apec_2025_overview"] = apec2025_data
    existing["apec_2025_meetings"] = meetings_data

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print("Hoàn thành. Đã cập nhật data.json")


if __name__ == "__main__":
    main()
