"""Permission and fallback HTML stay branded without weakening authorization."""
import asyncio
from types import SimpleNamespace

import pytest

from src.web import ApplicationHttpController, ApplicationLogBroker, AuthController, AuthService
from src.web.page_responses import message_page


class _Repository:
    def get_user_by_username(self, username):
        if username in {"reader", "administrator"}:
            return {
                "username": username,
                "display_name": username,
                "role": "admin" if username == "administrator" else "user",
            }
        return None


@pytest.fixture
def controllers(tmp_path):
    repository = _Repository()
    auth = AuthService(
        repository=repository,
        data_dir=tmp_path / "data",
        logger=lambda _message: None,
    )
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    controller = AuthController(auth, static_dir=static_dir, repository=repository)
    application = ApplicationHttpController(
        auth=auth,
        logs=ApplicationLogBroker(),
        static_dir=static_dir,
        voice_calls_dir=tmp_path / "voice_calls",
        get_agent=lambda: None,
        repository=repository,
        environ={},
        logger=lambda _message: None,
    )
    return auth, controller, application, static_dir


def _request(auth, username=None):
    return SimpleNamespace(cookies=(
        {auth.cookie_name: auth.sign_session_token(username)} if username else {}
    ))


def _assert_branded(response, status_code):
    assert response.status_code == status_code
    html = response.body.decode("utf-8")
    assert '<html lang="zh-CN">' in html
    assert 'name="viewport"' in html
    assert 'name="theme-color" content="#edf5ef"' in html
    assert '/static/companion_ui.css?v=20260914-unified-1' in html
    assert 'class="companion-page message-page"' in html
    assert 'href="#message-main"' in html and 'id="message-main"' in html
    assert 'aria-labelledby="message-title"' in html and 'id="message-title"' in html
    assert '心语陪伴' in html and '返回陪伴' in html
    assert '返回评估首页' not in html
    assert 'no-store' in response.headers['cache-control']
    return html


@pytest.mark.parametrize("status_code", [403, 404])
def test_message_page_is_accessible_branded_and_escapes_copy(status_code):
    html = _assert_branded(
        message_page('页面 <test>', '说明 <script>alert("x")</script> & 提示', status_code=status_code),
        status_code,
    )
    assert '&lt;test&gt;' in html
    assert '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; 提示' in html
    assert '<script>' not in html


def test_unauthenticated_admin_page_still_redirects_to_login(controllers):
    auth, controller, _, _ = controllers
    response = asyncio.run(controller.admin_page(_request(auth)))
    assert response.status_code == 307
    assert response.headers['location'] == '/login?next=/admin'


def test_non_admin_gets_branded_forbidden_page_without_admin_content(controllers):
    auth, controller, _, static_dir = controllers
    (static_dir / 'admin.html').write_text('private-admin-content', encoding='utf-8')
    response = asyncio.run(controller.admin_page(_request(auth, 'reader')))
    html = _assert_branded(response, 403)
    assert '需要管理员权限' in html
    assert 'private-admin-content' not in html


def test_admin_still_receives_admin_page(controllers):
    auth, controller, _, static_dir = controllers
    (static_dir / 'admin.html').write_text('private-admin-content', encoding='utf-8')
    response = asyncio.run(controller.admin_page(_request(auth, 'administrator')))
    assert response.status_code == 200
    assert response.body.decode('utf-8') == 'private-admin-content'


def test_missing_login_page_is_branded_not_an_unstyled_heading(controllers):
    auth, controller, _, _ = controllers
    response = asyncio.run(controller.login_page(_request(auth)))
    assert '登录页面未找到' in _assert_branded(response, 404)


@pytest.mark.parametrize('controller_index', [1, 2])
def test_missing_static_pages_use_the_shared_fallback(controllers, controller_index):
    controller = controllers[controller_index]
    response = controller._static_html('missing.html', '页面未找到')
    assert '页面未找到' in _assert_branded(response, 404)
