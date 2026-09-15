/* Shared, dependency-free interactions for 心语陪伴. */
(() => {
    'use strict';

    let activeDialog = null;

    function showDialog({ title, message, confirmText = '确定', cancelText = '取消', destructive = false, alertOnly = false } = {}) {
        // Repeated clicks must never silently confirm a destructive action.
        if (activeDialog) return Promise.resolve(false);
        const returnFocus = document.activeElement;
        const dialog = document.createElement('dialog');
        dialog.className = 'companion-dialog';
        dialog.setAttribute('aria-labelledby', 'companion-dialog-title');
        dialog.setAttribute('aria-describedby', 'companion-dialog-message');

        const heading = document.createElement('h2');
        heading.id = 'companion-dialog-title';
        heading.className = 'ui-dialog-title';
        heading.textContent = title || (alertOnly ? '提示' : '请确认');
        const text = document.createElement('p');
        text.id = 'companion-dialog-message';
        text.className = 'ui-dialog-message';
        text.textContent = message || '';
        const actions = document.createElement('form');
        actions.className = 'ui-dialog-actions';
        actions.method = 'dialog';
        const cancel = document.createElement('button');
        cancel.type = 'submit';
        cancel.value = 'cancel';
        cancel.className = 'ui-button';
        cancel.textContent = cancelText;
        if (!alertOnly) actions.append(cancel);
        const accept = document.createElement('button');
        accept.type = 'submit';
        accept.value = 'confirm';
        accept.className = `ui-button ${destructive ? 'ui-button-danger' : 'ui-button-primary'}`;
        accept.textContent = confirmText;
        actions.append(accept);
        dialog.append(heading, text, actions);
        document.body.append(dialog);
        activeDialog = dialog;

        return new Promise(resolve => {
            let finished = false;
            const finish = confirmed => {
                if (finished) return;
                finished = true;
                activeDialog = null;
                dialog.remove();
                if (returnFocus && returnFocus.isConnected && !returnFocus.disabled && !returnFocus.closest('[hidden], [inert]') && returnFocus.getClientRects().length) {
                    returnFocus.focus({ preventScroll: true });
                }
                resolve(confirmed);
            };
            dialog.addEventListener('close', () => finish(dialog.returnValue === 'confirm'));
            dialog.addEventListener('cancel', event => {
                event.preventDefault();
                dialog.close('cancel');
            });
            dialog.addEventListener('click', event => {
                if (event.target !== dialog) return;
                const rect = dialog.getBoundingClientRect();
                if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) {
                    dialog.close('cancel');
                }
            });
            try {
                dialog.showModal();
                // The non-destructive option receives focus for a confirmation.
                (alertOnly ? accept : cancel).focus();
            } catch (_) {
                // An unsupported dialog must not fall through to deletion/revocation.
                finish(false);
            }
        });
    }

    function handleTabKeydown(event) {
        if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
        const tab = event.target.closest?.('[role="tab"]');
        const group = tab && tab.closest('[data-keyboard-tabs]');
        if (!group || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        const tabs = [...group.querySelectorAll('[role="tab"]')].filter(item =>
            !item.disabled && item.getAttribute('aria-disabled') !== 'true' &&
            !item.closest('[hidden], [inert]') && item.closest('[data-keyboard-tabs]') === group);
        const current = tabs.indexOf(tab);
        if (current < 0 || !tabs.length) return;
        let next = current;
        if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = tabs.length - 1;
        else next = (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
        event.preventDefault();
        tabs[next].click();
        tabs[next].focus();
    }

    document.addEventListener('keydown', handleTabKeydown);
    window.CompanionUI = Object.freeze({
        confirm: options => showDialog(options),
        alert: options => showDialog({ ...options, alertOnly: true }),
    });
})();
