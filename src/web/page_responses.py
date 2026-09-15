"""Branded HTML for server-rendered permission and missing-page states."""
from __future__ import annotations

from html import escape

from fastapi.responses import HTMLResponse


def message_page(
    title: str,
    message: str,
    *,
    status_code: int,
) -> HTMLResponse:
    """Use the same accessible shell as the static UI without changing access rules."""
    safe_title = escape(title)
    safe_message = escape(message)
    return HTMLResponse(
        content=f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#edf5ef">
    <title>{safe_title} · 心语陪伴</title>
    <link rel="icon" href="data:,">
    <link rel="stylesheet" href="/static/vendor/fontawesome/css/all.min.css">
    <link rel="stylesheet" href="/static/companion_ui.css?v=20260914-unified-1">
</head>
<body class="companion-page message-page">
    <a class="ui-skip-link" href="#message-main">跳到页面内容</a>
    <a class="brand" href="/" aria-label="心语陪伴首页">
        <span class="brand-mark"><i class="fas fa-heart-pulse" aria-hidden="true"></i></span>
        <span class="brand-title">心语陪伴</span>
    </a>
    <main class="message-main" id="message-main" tabindex="-1">
        <section class="message-shell" aria-labelledby="message-title">
            <span class="state-badge">{status_code}</span>
            <h1 id="message-title">{safe_title}</h1>
            <p>{safe_message}</p>
            <a class="btn primary" href="/">
                <i class="fas fa-arrow-left" aria-hidden="true"></i>返回陪伴
            </a>
        </section>
    </main>
</body>
</html>''',
        status_code=status_code,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
