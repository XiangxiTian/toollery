from __future__ import annotations

import math
import re
from collections import Counter
from hashlib import blake2b
from typing import Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
CJK_RE = re.compile(r"^[\u4e00-\u9fff]+$")


STOPWORDS = {
    "a",
    "an",
    "and",
    "api",
    "by",
    "for",
    "from",
    "get",
    "i",
    "in",
    "is",
    "me",
    "need",
    "of",
    "on",
    "or",
    "please",
    "the",
    "to",
    "tool",
    "use",
    "with",
}

SYNONYMS = {
    "umbrella": ["rain", "weather", "forecast"],
    "rain": ["weather", "forecast"],
    "forecast": ["weather"],
    "hotel": ["stay", "lodging"],
    "hotels": ["stay", "lodging"],
    "stay": ["hotel", "lodging"],
    "flight": ["travel", "origin", "destination"],
    "fly": ["flight", "travel"],
    "meeting": ["calendar", "event", "attendees"],
    "meet": ["calendar", "event"],
    "calendar": ["meeting", "event"],
    "email": ["message", "recipient"],
    "note": ["email", "message"],
    "send": ["email", "message"],
    "usd": ["currency", "money", "amount"],
    "cny": ["currency", "money", "amount"],
    "eur": ["currency", "money", "amount"],
    "currency": ["money", "convert", "amount"],
    "change": ["convert", "currency"],
}

ZH_PHRASES = (
    "光场屏",
    "投影幕布",
    "投影光机",
    "后视镜",
    "中控屏",
    "仪表盘",
    "娱乐屏",
    "隐私玻璃",
    "侧方行驶",
    "蟹行",
    "驾驶模式",
    "运动模式",
    "舒适模式",
    "节能模式",
    "单踏板",
    "自动驻车",
    "自适应巡航",
    "车道辅助",
    "盲区监测",
    "碰撞预警",
    "画面",
    "画幅",
    "屏幕",
    "放大",
    "调大",
    "缩小",
    "调小",
    "取消",
    "关闭",
    "关掉",
    "禁用",
    "退出",
    "打开",
    "开启",
)

ZH_SYNONYMS = {
    "取消": ["关闭", "关掉", "禁用", "退出", "off", "disable"],
    "关闭": ["取消", "关掉", "禁用", "off", "disable"],
    "关掉": ["关闭", "取消", "禁用", "off", "disable"],
    "禁用": ["关闭", "取消", "off", "disable"],
    "退出": ["取消", "关闭", "off", "disable"],
    "放大": ["调大", "变大", "增大", "increase", "enlarge"],
    "调大": ["放大", "变大", "增大", "increase", "enlarge"],
    "缩小": ["调小", "变小", "减小", "decrease"],
    "调小": ["缩小", "变小", "减小", "decrease"],
    "画面": ["画幅", "屏幕", "screen", "size"],
    "画幅": ["画面", "屏幕", "screen", "size"],
    "屏幕": ["画面", "画幅", "screen", "size"],
}


def normalize_text(text: str) -> str:
    spaced = CAMEL_RE.sub(" ", text.replace("_", " ").replace("-", " "))
    return spaced.lower()


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(normalize_text(text))
    expanded: list[str] = []
    for token in tokens:
        if CJK_RE.match(token):
            expanded.extend(_tokenize_cjk(token))
            continue
        if token in STOPWORDS or len(token) <= 1:
            continue
        expanded.append(token)
        expanded.extend(SYNONYMS.get(token, []))
    return expanded


def _tokenize_cjk(text: str) -> list[str]:
    out: list[str] = []
    for phrase in ZH_PHRASES:
        if phrase in text:
            out.append(phrase)
            out.extend(ZH_SYNONYMS.get(phrase, []))
    for n in (2, 3, 4):
        if len(text) >= n:
            out.extend(text[index : index + n] for index in range(len(text) - n + 1))
    return out


def term_overlap(left: str, right: str) -> float:
    left_terms = set(tokenize(left))
    right_terms = set(tokenize(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / math.sqrt(len(left_terms) * len(right_terms))


def top_terms(texts: Iterable[str], limit: int = 8) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return [term for term, _ in counts.most_common(limit)]


def stable_hash(token: str, buckets: int) -> int:
    digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets
