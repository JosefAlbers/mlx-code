from __future__ import annotations
import asyncio
import copy
import json
import os
import re
import sys
import tempfile
import logging
import datetime
import threading
from typing import Any, Callable, Literal
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.requests import Request
import uvicorn
from .repl import Agent, collect_skills, _make_agent_env
from .gits import create_worktree, commit_worktree, resume_worktree, cleanup_worktree, git_new_branch, git_new_branch_at, git_switch_branch, GitError, get_diff_between_refs, get_branch_base_sha, find_rev_commit
logger = logging.getLogger(__name__)
_WEB_HTML = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">\n<meta name="color-scheme" content="dark light">\n<title>MLX Code</title>\n<style>\n:root { color-scheme: dark; }\n*{margin:0;padding:0;box-sizing:border-box}\nhtml, body { height: 100%; }\nbody{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;height:100vh;height:100dvh;display:flex;flex-direction:column;overscroll-behavior:none;-webkit-tap-highlight-color:transparent}\n#hdr{padding:8px 16px;padding-top:max(8px, env(safe-area-inset-top));background:#161b22;border-bottom:1px solid #30363d;display:flex;justify-content:space-between;align-items:center;flex-shrink:0;gap:8px}\n#hdr h1{font-size:14px;font-weight:600;white-space:nowrap}\n.filters{display:flex;gap:8px;align-items:center;font-size:11px;color:#8b949e;background:#0d1117;padding:4px 8px;border:1px solid #30363d;border-radius:6px}\n.filters label{display:flex;align-items:center;gap:4px;cursor:pointer;user-select:none}\n.filters input{cursor:pointer;accent-color:#58a6ff;width:14px;height:14px}\n#tabbar{display:flex;align-items:center;background:#161b22;border-bottom:1px solid #30363d;padding:6px 16px;gap:6px;overflow-x:auto;flex-shrink:0;-webkit-overflow-scrolling:touch}\n#tabbar::-webkit-scrollbar{height:4px}\n#tabbar::-webkit-scrollbar-thumb{background:#30363d;border-radius:2px}\n.tab{padding:6px 10px;border-radius:6px;cursor:pointer;white-space:nowrap;font-size:13px;color:#8b949e;display:flex;align-items:center;gap:6px;border:1px solid transparent;flex-shrink:0;min-height:32px}\n.tab:hover{background:#21262d}\n.tab.active{background:rgba(56,139,253,0.15);color:#58a6ff;border-color:rgba(56,139,253,0.3)}\n.tab-marker{color:#3fb950;font-size:10px}\n.tab.running .tab-marker{color:#d29922;animation:pulse 1.5s infinite}\n@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}\n.tab-num{color:#484f58;font-size:11px}\n.close-btn{color:#484f58;margin-left:2px;font-size:14px;line-height:1;padding:2px 4px;cursor:pointer}\n.close-btn:hover{color:#f85149}\n#newTabBtn{padding:6px 12px;border-radius:6px;cursor:pointer;color:#8b949e;border:1px solid #30363d;background:transparent;font-size:13px;flex-shrink:0;min-height:32px}\n#newTabBtn:hover{background:#21262d;color:#c9d1d9}\n#chat{flex:1;overflow-y:auto;padding:clamp(8px, 3vw, 16px);padding-bottom:8px;-webkit-overflow-scrolling:touch}\n.chat-inner{max-width:920px;margin:0 auto}\n.msg{margin-bottom:14px}\n.msg-role{font-size:12px;color:#8b949e;margin-bottom:3px}\n.msg-body{padding:10px 14px;border-radius:8px;line-height:1.6;white-space:pre-wrap;word-break:break-word;overflow-wrap:anywhere;font-size:14px}\n.msg-user .msg-body{background:#1c2128;border:1px solid #30363d}\n.msg-assistant .msg-body{background:#161b22;border:1px solid #30363d}\n.msg-thinking .msg-body{color:#6e7681;font-style:italic;background:rgba(136,144,150,0.05);border-left:2px solid #30363d;font-size:13px}\n.msg-tool .msg-body{background:rgba(210,153,34,0.08);border-left:2px solid #d29922;font-family:ui-monospace,SFMono-Regular,monospace;font-size:13px;overflow-x:auto}\n.msg-tool-result .msg-body{background:rgba(35,134,54,0.08);border-left:2px solid #238636;font-family:ui-monospace,SFMono-Regular,monospace;font-size:13px;overflow-x:auto}\n.msg-commit .msg-body{background:rgba(56,139,253,0.08);border-left:2px solid #388bfd;color:#8b949e;font-size:13px;overflow-x:auto}\n.msg-error .msg-body{color:#f85149;background:rgba(248,81,73,0.05)}\n.cursor{display:inline-block;width:7px;height:15px;background:#58a6ff;animation:blink 1s steps(2) infinite;vertical-align:text-bottom;margin-left:2px;border-radius:1px}\n@keyframes blink{50%{opacity:0}}\n#input-area{padding:12px 16px;padding-bottom:max(12px, env(safe-area-inset-bottom));background:#161b22;border-top:1px solid #30363d;flex-shrink:0}\n.input-inner{max-width:920px;margin:0 auto;display:flex;gap:8px}\n#input{flex:1;background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:8px;padding:10px 14px;font-family:inherit;font-size:16px;resize:none;height:46px;max-height:200px;line-height:1.5}\n#input:focus{outline:none;border-color:#58a6ff}\n#send{background:#238636;color:#fff;border:none;border-radius:8px;padding:0 20px;cursor:pointer;font-size:14px;font-weight:500;white-space:nowrap;min-width:80px;height:46px}\n#send:hover{background:#2ea043}\n#send.abort{background:#da3633}\n#send.abort:hover{background:#f85149}\n.hide-thinking .msg-thinking, .hide-tools .msg-tool, .hide-results .msg-tool-result, .hide-commits .msg-commit { display: none; }\n\n/* Mobile compactness */\n@media (max-width: 600px) {\n  .filters label .hide-mobile { display: none; }\n  #hdr h1 { font-size: 12px; }\n  #hdr { padding: 6px 10px; padding-top:max(6px, env(safe-area-inset-top)); }\n  #tabbar { padding: 6px 10px; }\n  #input-area { padding: 8px 10px; padding-bottom:max(8px, env(safe-area-inset-bottom)); }\n  .msg-body { font-size: 15px; }\n}\n</style>\n</head>\n<body>\n<div id="hdr">\n  <h1>⚡ MLX Code</h1>\n  <div class="filters">\n    <label><input type="checkbox" id="f-thinking" checked><span class="hide-mobile">Thinking</span></label>\n    <label><input type="checkbox" id="f-tools" checked><span class="hide-mobile">Tools</span></label>\n    <label><input type="checkbox" id="f-results" checked><span class="hide-mobile">Results</span></label>\n    <label><input type="checkbox" id="f-commits" checked><span class="hide-mobile">Commits</span></label>\n    <label><input type="checkbox" id="f-autoscroll" checked><span>Auto-scroll</span></label>\n  </div>\n</div>\n<div id="tabbar">\n    <button id="newTabBtn" title="Branch from current tab">+ Branch</button>\n</div>\n<div id="chat">\n    <div class="chat-inner" id="chatInner"></div>\n    <div class="chat-inner" id="streamInner"></div>\n</div>\n<div id="input-area">\n  <div class="input-inner">\n    <textarea id="input" placeholder="Send a message..." rows="1"></textarea>\n    <button id="send">Send</button>\n  </div>\n</div>\n<script>\nconst chatEl=document.getElementById(\'chat\');\nconst innerEl=document.getElementById(\'chatInner\');\nconst streamInnerEl=document.getElementById(\'streamInner\');\nconst inputEl=document.getElementById(\'input\');\nconst sendBtn=document.getElementById(\'send\');\nconst tabbar=document.getElementById(\'tabbar\');\nconst newTabBtn=document.getElementById(\'newTabBtn\');\nconst autoScrollChk = document.getElementById(\'f-autoscroll\');\n\nlet activeTab = 0;\nlet tabs = [];\nconst tabState = {};\nlet historyFetchInProgress = false;\nlet activeStreamNodes = [];\nlet pendingForceScroll = false; // FIX: Track if we need to force scroll on the next render\n\n[\'thinking\',\'tools\',\'results\',\'commits\'].forEach(t => {\n  const el = document.getElementById(\'f-\'+t);\n  el.addEventListener(\'change\', () => chatEl.classList.toggle(\'hide-\'+t, !el.checked));\n});\n\nfunction getTabState(tabId) {\n    if (!tabState[tabId]) {\n        tabState[tabId] = { streamBlocks: [], status: \'idle\', toolCallBuf: \'\' };\n    }\n    return tabState[tabId];\n}\n\nfunction connect() {\n    const evtSource = new EventSource(\'/events\');\n    evtSource.onmessage = (e) => { handleEvent(JSON.parse(e.data)); };\n    evtSource.onerror = () => { evtSource.close(); setTimeout(connect, 1000); };\n}\n\nlet scrollPending = false;\nlet userScrolledUp = false;\n\nfunction isNearBottom() {\n    return chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight < 150;\n}\n\nchatEl.addEventListener(\'scroll\', () => {\n    if (!isNearBottom()) {\n        userScrolledUp = true;\n    } else {\n        userScrolledUp = false;\n    }\n});\n\nfunction scrollBottom(force = false){\n    if (scrollPending) return;\n    if (!force && userScrolledUp) return;\n\n    scrollPending = true;\n    requestAnimationFrame(() => {\n        chatEl.scrollTop = chatEl.scrollHeight;\n        scrollPending = false;\n        if (force) userScrolledUp = false; // Reset state if we forced it\n    });\n}\n\nfunction addMsg(role,label,parentEl){\n  const d=document.createElement(\'div\');d.className=\'msg msg-\'+role;\n  const r=document.createElement(\'div\');r.className=\'msg-role\';r.textContent=label;\n  const b=document.createElement(\'div\');b.className=\'msg-body\';\n  d.appendChild(r);d.appendChild(b);parentEl.appendChild(d);return b;\n}\n\nfunction stripToolXml(text){\n  text=text.replace(/<tool_call>[\\s\\S]*?<\\/tool_call>/g,\'\');\n  const idx=text.lastIndexOf(\'<tool_call>\');\n  if(idx!==-1&&text.indexOf(\'</tool_call>\',idx)===-1)return text.substring(0,idx);\n  const tag=\'<tool_call>\';\n  for(let i=tag.length-1;i>0;i--){if(text.endsWith(tag.substring(0,i)))return text.substring(0,text.length-i);}\n  return text;\n}\n\nfunction cleanDisplay(text){ return text.replace(/^\\n+/, \'\').replace(/\\n+$/, \'\'); }\n\nfunction handleEvent(data){\n    const type = data.type;\n    const payload = data.payload || {};\n    const tabId = data.tab_id;\n\n    if (type === \'tab_list\') {\n        const prevActive = activeTab;\n        tabs = payload.tabs || [];\n        activeTab = payload.active_id;\n        if (activeTab !== prevActive) {\n            const state = getTabState(activeTab);\n            renderStream(state);\n            updateStatus(state);\n            refreshHistory(true);\n        }\n        renderTabs();\n        return;\n    }\n\n    if (type === \'history\') {\n        if (tabId === activeTab) {\n            renderHistory(payload.messages || [], true);\n        }\n        return;\n    }\n\n    const state = getTabState(tabId);\n\n    switch (type) {\n        case \'agent_start\':\n            state.status = \'running\';\n            state.streamBlocks = [];\n            state.toolCallBuf = \'\';\n            break;\n        case \'turn_start\':\n            state.streamBlocks = [];\n            state.toolCallBuf = \'\';\n            break;\n        case \'text_delta\':\n            state.toolCallBuf += payload.delta || \'\';\n            var cleaned = state.toolCallBuf.replace(/<tool_call>[\\s\\S]*?<\\/tool_call>/g, \'\');\n            var emit;\n            var idx = cleaned.indexOf(\'<tool_call>\');\n            if (idx !== -1) {\n                emit = cleaned.substring(0, idx);\n                state.toolCallBuf = cleaned.substring(idx);\n            } else {\n                emit = cleaned;\n                state.toolCallBuf = \'\';\n            }\n            if (emit.trim()) {\n                var last = state.streamBlocks[state.streamBlocks.length - 1];\n                if (last && last.type === \'text\' && !last.isError) {\n                    last.text += emit;\n                } else {\n                    state.streamBlocks.push({ type: \'text\', text: emit });\n                }\n            }\n            break;\n        case \'thinking_delta\':\n            var tDelta = payload.delta || \'\';\n            if (tDelta) {\n                var last = state.streamBlocks[state.streamBlocks.length - 1];\n                if (last && last.type === \'thinking\') {\n                    last.text += tDelta;\n                } else {\n                    state.streamBlocks.push({ type: \'thinking\', text: tDelta });\n                }\n            }\n            break;\n        case \'tool_start\':\n            state.streamBlocks.push({\n                type: \'toolCall\',\n                name: payload.name || \'tool\',\n                arguments: payload.args || {}\n            });\n            break;\n        case \'tool_end\':\n            var result = payload.result || {};\n            var content = result.content || [];\n            var outText = \'\';\n            if (typeof content === \'string\') {\n                outText = content;\n            } else if (Array.isArray(content)) {\n                outText = content.filter(b => b.type === \'text\').map(b => b.text || \'\').join(\'\\n\').trim();\n            }\n            if (payload.is_error) {\n                if (!outText) outText = (payload.name || \'tool\') + \' failed\';\n                state.streamBlocks.push({ type: \'text\', text: outText, isError: true });\n            } else if (outText) {\n                state.streamBlocks.push({ type: \'toolResult\', text: outText });\n            }\n            break;\n        case \'commit\':\n            state.streamBlocks.push({\n                type: \'commit\',\n                sha: payload.sha || \'\',\n                diff: payload.diff_stat || \'\'\n            });\n            if (tabId === activeTab) refreshHistory(false);\n            break;\n        case \'error\':\n            var errObj = payload.error || payload;\n            var errMsg = (errObj && (errObj.error_message || errObj.message)) || String(errObj);\n            state.streamBlocks.push({ type: \'text\', text: errMsg, isError: true });\n            break;\n        case \'turn_end\':\n            state.streamBlocks = [];\n            if (tabId === activeTab) refreshHistory(false);\n            break;\n        case \'tool_results_ready\':\n            if (tabId === activeTab) refreshHistory(false);\n            break;\n        case \'agent_end\':\n            state.status = \'idle\';\n            state.streamBlocks = [];\n            // FIX: Only force scroll if the Auto-scroll checkbox is checked\n            if (tabId === activeTab) refreshHistory(autoScrollChk.checked); \n            break;\n        case \'command_output\':\n            state.streamBlocks.push({\n                type: \'command\',\n                command: payload.command || \'\',\n                output: payload.output || \'\'\n            });\n            break;\n        case \'shell_output\':\n            state.streamBlocks.push({\n                type: \'shell\',\n                command: payload.command || \'\',\n                output: payload.output || \'\'\n            });\n            break;\n    }\n\n    if (tabId === activeTab) {\n        renderStream(state);\n        updateStatus(state);\n    }\n\n    if (type === \'agent_start\' || type === \'agent_end\') {\n        renderTabs();\n    }\n}\n\nfunction renderTabs() {\n    var existing = tabbar.querySelectorAll(\'.tab\');\n    existing.forEach(function(el) { el.remove(); });\n\n    tabs.forEach(function(t, i) {\n        var el = document.createElement(\'div\');\n        el.className = \'tab\' + (t.id === activeTab ? \' active\' : \'\') + (t.is_running ? \' running\' : \'\');\n        var marker = \'\\u25CF\';\n        el.innerHTML = \'<span class="tab-marker">\' + marker + \'</span>\' +\n                      \'<span class="tab-title">\' + (t.title) + \'</span>\' +\n                      \'<span class="tab-num">\' + (i + 1) + \'</span>\';\n        if (!t.is_main) {\n            var closeBtn = document.createElement(\'span\');\n            closeBtn.className = \'close-btn\';\n            closeBtn.textContent = \'\\u00D7\';\n            closeBtn.onclick = function(e) {\n                e.stopPropagation();\n                closeTab(t.id);\n            };\n            el.appendChild(closeBtn);\n        }\n        el.onclick = function() { switchTab(t.id); };\n        tabbar.insertBefore(el, newTabBtn);\n    });\n}\n\nfunction renderHistory(messages, force = false) {\n    innerEl.innerHTML = \'\';\n    for (const msg of messages) {\n        const role = msg.role;\n        const content = msg.content;\n        const isError = msg.is_error || false;\n\n        if (role === \'commit\') {\n            addMsg(\'commit\', \'◇ Commit\', innerEl).textContent = cleanDisplay(\'◇ [\' + (msg.sha || \'\') + \'] committed\');\n        } else if (typeof content === \'string\') {\n            if (role === \'user\') addMsg(\'user\', \'≫ You\', innerEl).textContent = cleanDisplay(content);\n            else if (role === \'system\') addMsg(\'assistant\', \'· System\', innerEl).textContent = cleanDisplay(content);\n        } else if (Array.isArray(content)) {\n            if (role === \'toolResult\') {\n                let t = content.map(b => b.text || \'\').filter(Boolean).join(\'\\n\');\n                addMsg(isError ? \'error\' : \'tool-result\', isError ? \'✗ Error\' : \'→ Result\', innerEl).textContent = cleanDisplay((isError ? \'✗ \' : \'→ \') + (t || \'(no output)\'));\n            } else {\n                for (const block of content) {\n                    if (block.type === \'thinking\') addMsg(\'thinking\', \'◌ Thinking\', innerEl).textContent = cleanDisplay(block.thinking || \'\');\n                    else if (block.type === \'text\') {\n                        const txt = cleanDisplay(stripToolXml(block.text || \'\'));\n                        if (txt.trim()) {\n                            addMsg(\'assistant\', \'○ Assistant\', innerEl).textContent = txt;\n                        }\n                    }\n                    else if (block.type === \'toolCall\') {\n                        if (getTabState(activeTab).status === \'running\' && msg === messages[messages.length - 1]) {\n                            continue;\n                        }\n                        const b = addMsg(\'tool\', \'⚙ \' + (block.name || \'\'), innerEl);\n                        b.textContent = cleanDisplay(\'⚙ \' + (block.name || \'\') + \'\\n\' + JSON.stringify(block.arguments || {}, null, 2));\n                    }\n                }\n            }\n        }\n    }\n    scrollBottom(force);\n}\n\nfunction renderStream(state) {\n    if (state.streamBlocks.length === 0) {\n        streamInnerEl.innerHTML = \'\';\n        activeStreamNodes = [];\n        return;\n    }\n\n    while (activeStreamNodes.length < state.streamBlocks.length) {\n        const idx = activeStreamNodes.length;\n        const block = state.streamBlocks[idx];\n        const bodyEl = createStreamBlock(block);\n        activeStreamNodes.push(bodyEl);\n    }\n\n    const lastIdx = state.streamBlocks.length - 1;\n    const lastBlock = state.streamBlocks[lastIdx];\n    const lastNode = activeStreamNodes[lastIdx];\n\n    if (lastNode) {\n        if (lastBlock.type === \'text\' || lastBlock.type === \'thinking\') {\n            const txt = cleanDisplay(stripToolXml(lastBlock.text || \'\'));\n            lastNode.textContent = txt;\n            if (lastBlock.type === \'text\') {\n                lastNode.parentElement.style.display = txt.trim() ? \'\' : \'none\';\n            }\n        } else if (lastBlock.type === \'toolResult\') {\n            lastNode.textContent = cleanDisplay(\'→ \' + (lastBlock.text || \'\'));\n        }\n    }\n    scrollBottom(false);\n}\n\nfunction createStreamBlock(block) {\n    let label = \'\';\n    let role = \'\';\n\n    if (block.type === \'text\') {\n        role = \'assistant\'; label = \'○ Assistant\';\n    } else if (block.type === \'thinking\') {\n        role = \'thinking\'; label = \'◌ Thinking\';\n    } else if (block.type === \'toolCall\') {\n        role = \'tool\'; label = \'⚙ \' + block.name;\n    } else if (block.type === \'toolResult\') {\n        role = \'tool-result\'; label = \'→ Result\';\n    } else if (block.type === \'commit\') {\n        role = \'commit\'; label = \'◇ Commit\';\n    } else if (block.type === \'command\') {\n        role = \'tool\'; label = \'✓ \' + block.command;\n    } else if (block.type === \'shell\') {\n        role = \'tool\'; label = \'! \' + block.command;\n    }\n\n    const bodyEl = addMsg(role, label, streamInnerEl);\n\n    if (block.type === \'text\' || block.type === \'thinking\') {\n        const txt = cleanDisplay(stripToolXml(block.text || \'\'));\n        bodyEl.textContent = txt;\n        if (block.type === \'text\' && !txt.trim()) {\n            bodyEl.parentElement.style.display = \'none\';\n        }\n    } else if (block.type === \'toolCall\') {\n        bodyEl.textContent = cleanDisplay(\'⚙ \' + block.name + \'\\n\' + JSON.stringify(block.arguments, null, 2));\n    } else if (block.type === \'toolResult\') {\n        bodyEl.textContent = cleanDisplay(\'→ \' + (block.text || \'\'));\n    } else if (block.type === \'commit\') {\n        bodyEl.textContent = cleanDisplay(\'◇ [\' + (block.sha || \'\') + \'] committed\');\n    } else if (block.type === \'command\' || block.type === \'shell\') {\n        bodyEl.textContent = cleanDisplay(block.output || \'\');\n    }\n    return bodyEl;\n}\n\nfunction updateStatus(state) {\n    if (state.status === \'running\') {\n        sendBtn.textContent = \'Abort\';\n        sendBtn.classList.add(\'abort\');\n    } else {\n        sendBtn.textContent = \'Send\';\n        sendBtn.classList.remove(\'abort\');\n    }\n}\n\n// FIX: Robust auto-resize without jitter\nfunction autoResizeInput() {\n    inputEl.style.height = \'auto\'; // collapse for measurement\n    let h = inputEl.scrollHeight + 2; // +2px to account for top and bottom borders\n    if (h < 46) h = 46; // Enforce the base CSS height\n    inputEl.style.height = h + \'px\';\n}\n\nasync function send(){\n    const text=inputEl.value.trim();if(!text)return;\n    var state = getTabState(activeTab);\n    if (state.status === \'running\') return;\n\n    inputEl.value=\'\';\n    inputEl.style.height=\'46px\'; // Reset exactly to original CSS height\n    userScrolledUp = false;\n\n    if(!text.startsWith(\'/\') && !text.startsWith(\'!\')) {\n        addMsg(\'user\', \'≫ You\', innerEl).textContent = text;\n        scrollBottom(true);\n    }\n\n    try {\n        await fetch(\'/send\', {\n            method: \'POST\',\n            headers: {\'Content-Type\': \'application/json\'},\n            body: JSON.stringify({ text: text, tab_id: activeTab })\n        });\n    } catch (e) {\n        addMsg(\'error\', \'✗ Error\', streamInnerEl).textContent = cleanDisplay(\'✗ \' + e.message);\n    }\n}\n\nfunction abortAgent() {\n    fetch(\'/abort\', {\n        method: \'POST\',\n        headers: {\'Content-Type\': \'application/json\'},\n        body: JSON.stringify({ tab_id: activeTab })\n    });\n}\n\nfunction switchTab(tabId) {\n    if (tabId === activeTab) return;\n    fetch(\'/switch_tab\', {\n        method: \'POST\',\n        headers: {\'Content-Type\': \'application/json\'},\n        body: JSON.stringify({ tab_id: tabId })\n    }).then(r => r.json()).then(data => {\n        if (data.ok) {\n            activeTab = tabId;\n            var state = getTabState(activeTab);\n            userScrolledUp = false;\n            inputEl.value = \'\'; // Clear input on tab switch\n            inputEl.style.height = \'46px\'; // Reset height on tab switch\n            renderStream(state);\n            updateStatus(state);\n            renderHistory(data.messages || [], true);\n            renderTabs();\n        }\n    });\n}\n\nfunction closeTab(tabId) {\n    fetch(\'/close_tab\', {\n        method: \'POST\',\n        headers: {\'Content-Type\': \'application/json\'},\n        body: JSON.stringify({ tab_id: tabId })\n    });\n}\n\n// FIX: Carry over the force flag if a fetch is already in progress\nfunction refreshHistory(force = false) {\n    if (force) pendingForceScroll = true;\n    if (historyFetchInProgress) return;\n    historyFetchInProgress = true;\n    fetch(\'/history/\' + activeTab)\n        .then(r => r.json())\n        .then(data => { \n            const shouldForce = pendingForceScroll;\n            pendingForceScroll = false;\n            renderHistory(data.messages || [], shouldForce); \n        })\n        .catch(e => console.error(\'History fetch error:\', e))\n        .finally(() => { historyFetchInProgress = false; });\n}\n\ninputEl.addEventListener(\'keydown\',e=>{\n    if(e.key===\'Enter\'&&!e.shiftKey) {\n        e.preventDefault();\n        var state = getTabState(activeTab);\n        if (state.status === \'running\') {\n            abortAgent();\n        } else {\n            send();\n        }\n    }\n});\ninputEl.addEventListener(\'input\', autoResizeInput); // Use the robust resize function\n\nsendBtn.addEventListener(\'click\', () => {\n    var state = getTabState(activeTab);\n    if (state.status === \'running\') {\n        abortAgent();\n    } else {\n        send();\n    }\n});\n\nnewTabBtn.addEventListener(\'click\', () => {\n    fetch(\'/branch\', {\n        method: \'POST\',\n        headers: {\'Content-Type\': \'application/json\'},\n        body: JSON.stringify({ tab_id: activeTab, prompt: \'\' })\n    });\n});\n\ndocument.addEventListener(\'keydown\', e => {\n    if (e.altKey && e.key >= \'1\' && e.key <= \'9\') {\n        e.preventDefault();\n        var idx = parseInt(e.key) - 1;\n        if (idx < tabs.length) switchTab(tabs[idx].id);\n    }\n});\n\ninputEl.focus();\nconnect();\n</script>\n</body>\n</html>'

def _branch_index_title(parent_path: tuple[int, ...], existing_tabs: list['WebTab']) -> tuple[tuple[int, ...], str]:
    depth = len(parent_path) + 1
    child_count = sum((1 for t in existing_tabs if len(t.index_path) == depth and t.index_path[:-1] == parent_path))
    index_path = parent_path + (child_count,)
    title = 'branch-' + '-'.join((str(i) for i in index_path))
    return (index_path, title)

class WebTab:

    def __init__(self, tab_id: int, agent: Agent, title: str, index_path: tuple=(), owns_worktree: bool=False, is_main: bool=False):
        self.id = tab_id
        self.agent = agent
        self.title = title
        self.index_path = index_path
        self.owns_worktree = owns_worktree
        self.is_main = is_main
        self.errors: list[tuple[str, str]] = []
        self.running_task: asyncio.Task | None = None
        self.status: str = 'idle'

    @property
    def is_running(self) -> bool:
        return self.running_task is not None and (not self.running_task.done())

class WebRepl:

    def __init__(self, agent: Agent, init_prompt: str | None=None):
        self._next_id = 0
        self.tabs: list[WebTab] = []
        self.active_id: int = 0
        self._unsubscribers: dict[int, Callable] = {}
        self._subscribers: set[asyncio.Queue] = set()
        self.init_prompt = init_prompt
        main_tab = WebTab(self._next_id, agent, 'main', is_main=True)
        self._next_id += 1
        self.tabs.append(main_tab)
        self._attach_agent(main_tab)

    @property
    def active_tab(self) -> WebTab:
        return next((t for t in self.tabs if t.id == self.active_id))

    def _tab_by_id(self, tab_id: int) -> WebTab | None:
        return next((t for t in self.tabs if t.id == tab_id), None)

    def _attach_agent(self, tab: WebTab):
        key = id(tab.agent)
        if key in self._unsubscribers:
            return

        async def on_event(event: dict):
            event_with_tab = {**event, 'tab_id': tab.id}
            for q in list(self._subscribers):
                q.put_nowait(event_with_tab)
        self._unsubscribers[key] = tab.agent.subscribe(on_event)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers.discard(q)

    def _emit_to_tab(self, tab: WebTab, event: dict):
        event_with_tab = {**event, 'tab_id': tab.id}
        for q in list(self._subscribers):
            q.put_nowait(event_with_tab)

    def _broadcast_tab_list(self):
        tab_list = [{'id': t.id, 'title': t.title, 'is_running': t.is_running, 'status': t.status, 'is_main': t.is_main} for t in self.tabs]
        event = {'type': 'tab_list', 'payload': {'tabs': tab_list, 'active_id': self.active_id}, 'tab_id': -1}
        for q in list(self._subscribers):
            q.put_nowait(event)

    async def run_prompt(self, tab_id: int, text: str):
        tab = self._tab_by_id(tab_id)
        if not tab or tab.is_running:
            return

        async def _run():
            try:
                tab.status = 'running'
                self._broadcast_tab_list()
                if text.startswith('/'):
                    await self._handle_command(tab, text)
                elif text.startswith('!'):
                    await self._run_shell(tab, text[1:].strip())
                else:
                    await tab.agent.run(text)
            except Exception as e:
                logger.exception(f'run_prompt error: {e}')
                self._emit_to_tab(tab, {'type': 'error', 'payload': {'error': {'error_message': str(e)}}})
            finally:
                tab.status = 'idle'
                self._broadcast_tab_list()
        tab.running_task = asyncio.create_task(_run())

    async def _run_shell(self, tab: WebTab, command: str):
        if not command:
            return
        gwt = tab.agent.ctx.get('gwt')
        cwd = gwt.worktree if gwt and hasattr(gwt, 'worktree') else tab.agent.ctx.get('cwd', os.getcwd())
        env = tab.agent.ctx.get('env')
        proc = await asyncio.create_subprocess_shell(command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
        stdout, stderr = await proc.communicate()
        out = stdout.decode(errors='replace').rstrip('\n')
        err = stderr.decode(errors='replace').rstrip('\n')
        result = out
        if err:
            result += f'\n[stderr]\n{err}' if result else f'[stderr]\n{err}'
        if proc.returncode:
            result += f'\n[exit {proc.returncode}]'
        self._emit_to_tab(tab, {'type': 'shell_output', 'payload': {'command': command, 'output': result}})

    async def _handle_command(self, tab: WebTab, text: str):
        cmd, _, arg = text.partition(' ')
        cmd = cmd.lower().strip()
        arg = arg.strip()
        if cmd == '/help':
            help_text = 'Commands:\n/help               show this message\n/clear              clear conversation\n/history            show conversation history\n/diff [--all]       show git diff\n/errors             show error log\n/tools              list active tools\n/branch [--rev N] [--no-worktree] [prompt]\n                    open a branch tab; optional prompt runs immediately\n/abort              abort the running agent\n/export [path]      export session to JSON\n/exit /quit         close branch tab\n!command            run shell command in worktree'
            self._emit_to_tab(tab, {'type': 'command_output', 'payload': {'command': text, 'output': help_text}})
        elif cmd == '/clear':
            tab.agent.messages.clear()
            self._emit_to_tab(tab, {'type': 'history', 'payload': {'messages': []}})
            self._emit_to_tab(tab, {'type': 'command_output', 'payload': {'command': text, 'output': 'Conversation cleared.'}})
        elif cmd == '/branch':
            await self._cmd_branch(tab, arg)
        elif cmd == '/tab':
            if arg and arg.isdigit():
                n = int(arg) - 1
                if 0 <= n < len(self.tabs):
                    await self._switch_tab(self.tabs[n].id)
        elif cmd == '/branches':
            lines = []
            for i, t in enumerate(self.tabs):
                marker = '▶' if t.id == self.active_id else ' '
                lines.append(f'{marker} {i + 1}. {t.title}')
            self._emit_to_tab(tab, {'type': 'command_output', 'payload': {'command': text, 'output': '\n'.join(lines)}})
        elif cmd == '/abort':
            tab.agent.abort()
            self._emit_to_tab(tab, {'type': 'command_output', 'payload': {'command': text, 'output': 'Abort requested.'}})
        elif cmd == '/tools':
            tools = tab.agent.tools
            output = '\n'.join((f'{t.name}  {t.description}' for t in tools)) if tools else 'No tools enabled.'
            self._emit_to_tab(tab, {'type': 'command_output', 'payload': {'command': text, 'output': output}})
        elif cmd == '/diff':
            gwt = tab.agent.ctx.get('gwt')
            if not gwt or not hasattr(gwt, 'worktree'):
                output = 'No git worktree available.'
            else:
                ref1, ref2 = ('HEAD~1', 'HEAD')
                if '--all' in arg:
                    base = get_branch_base_sha(gwt.worktree)
                    if base:
                        ref1 = base
                try:
                    diff_text = get_diff_between_refs(gwt.worktree, ref1, ref2)
                    output = diff_text if diff_text.strip() else 'No differences.'
                except GitError as e:
                    output = f'Git diff failed: {e}'
            self._emit_to_tab(tab, {'type': 'command_output', 'payload': {'command': text, 'output': output}})
        elif cmd == '/history':
            user_msgs = [m for m in tab.agent.messages if m.get('role') == 'user']
            if not user_msgs:
                output = 'No prompts yet.'
            else:
                lines = []
                for i, m in enumerate(user_msgs, 1):
                    content = m.get('content', '')
                    if isinstance(content, list):
                        content = ' '.join((b.get('text', '') for b in content if isinstance(b, dict) and b.get('type') == 'text'))
                    content = re.sub('\\s+', ' ', content).strip()
                    if len(content) > 100:
                        content = content[:100] + '…'
                    lines.append(f'{i}. {content}')
                output = '\n'.join(lines)
            self._emit_to_tab(tab, {'type': 'command_output', 'payload': {'command': text, 'output': output}})
        elif cmd == '/errors':
            if not tab.errors:
                output = 'No errors recorded.'
            else:
                output = '\n'.join((f'{ts}  {msg}' for ts, msg in tab.errors[-30:]))
            self._emit_to_tab(tab, {'type': 'command_output', 'payload': {'command': text, 'output': output}})
        elif cmd == '/export':
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            user_cwd = tab.agent.ctx.get('user_cwd', os.getcwd())
            path = arg if arg and os.path.isabs(arg) else os.path.join(user_cwd, arg or f'session_{ts}.json')
            data = {'version': 1, 'exported_at': ts, 'system': tab.agent.system, 'messages': tab.agent.messages}
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                output = f'Exported → {path}'
            except OSError as e:
                output = f'Export failed: {e}'
            self._emit_to_tab(tab, {'type': 'command_output', 'payload': {'command': text, 'output': output}})
        elif cmd in ('/exit', '/quit'):
            if len(self.tabs) > 1 and (not tab.is_main):
                await self._close_tab(tab.id)
            else:
                self._emit_to_tab(tab, {'type': 'command_output', 'payload': {'command': text, 'output': 'Cannot close the main tab.'}})
        else:
            self._emit_to_tab(tab, {'type': 'command_output', 'payload': {'command': text, 'output': f'Unknown command: {cmd!r} — try /help'}})

    async def _cmd_branch(self, parent: WebTab, arg: str):
        as_worktree = True
        rev_n: int | None = None
        prompt = arg
        if '--no-worktree' in prompt:
            as_worktree = False
            prompt = prompt.replace('--no-worktree', '').strip()
        rev_match = re.search('--rev\\s+(\\d+)', prompt)
        if rev_match:
            rev_n = int(rev_match.group(1))
            prompt = (prompt[:rev_match.start()] + prompt[rev_match.end():]).strip()
        all_msgs = parent.agent.messages
        user_indices = [i for i, m in enumerate(all_msgs) if m.get('role') == 'user']
        if rev_n is not None:
            if rev_n < 1 or rev_n > len(user_indices):
                self._emit_to_tab(parent, {'type': 'error', 'payload': {'error': {'error_message': f'--rev {rev_n}: must be between 1 and {len(user_indices)}'}}})
                return
            cut_at = user_indices[rev_n - 1]
            sliced_messages = copy.deepcopy(all_msgs[:cut_at])
        else:
            sliced_messages = copy.deepcopy(all_msgs)
        child = parent.agent.branch()
        child.messages = sliced_messages
        index_path, title = _branch_index_title(parent.index_path, self.tabs)
        owns_worktree = False
        gwt = child.ctx.get('gwt')
        if as_worktree:
            repo_dir = gwt.worktree if gwt else child.ctx.get('cwd', os.getcwd())
            ref = 'HEAD'
            if rev_n is not None and gwt:
                target_sha = find_rev_commit(gwt.worktree, rev_n - 1)
                if target_sha:
                    ref = target_sha
            new_gwt = create_worktree(repo_dir, prefix=title, ref=ref)
            if new_gwt is None:
                self._emit_to_tab(parent, {'type': 'error', 'payload': {'error': {'error_message': f'git worktree creation failed for {title!r}'}}})
                return
            child.ctx['gwt'] = new_gwt
            child.ctx['cwd'] = new_gwt.worktree
            if 'env' in child.ctx:
                child.ctx['env']['PWD'] = new_gwt.worktree
            owns_worktree = True
        elif gwt:
            try:
                if rev_n is not None:
                    target_sha = find_rev_commit(gwt.worktree, rev_n - 1)
                    if target_sha:
                        new_gwt = git_new_branch_at(gwt.worktree, title, target_sha)
                    else:
                        new_gwt = git_new_branch(gwt.worktree, title)
                else:
                    new_gwt = git_new_branch(gwt.worktree, title)
                child.ctx['gwt'] = new_gwt
            except GitError as exc:
                logger.warning(f'git_new_branch failed: {exc}')
        new_tab = WebTab(self._next_id, child, title, index_path=index_path, owns_worktree=owns_worktree)
        self._next_id += 1
        self.tabs.append(new_tab)
        self._attach_agent(new_tab)
        self.active_id = new_tab.id
        self._broadcast_tab_list()
        if prompt:
            await self.run_prompt(new_tab.id, prompt)

    async def _switch_tab(self, tab_id: int):
        tab = self._tab_by_id(tab_id)
        if not tab:
            return
        prev = self.active_tab
        if prev is not tab:
            prev_gwt = prev.agent.ctx.get('gwt')
            next_gwt = tab.agent.ctx.get('gwt')
            if prev_gwt and next_gwt and (prev_gwt.worktree == next_gwt.worktree) and (prev_gwt.branch != next_gwt.branch):
                try:
                    updated = git_switch_branch(next_gwt.worktree, next_gwt.branch)
                    tab.agent.ctx['gwt'] = updated
                except GitError as exc:
                    logger.warning(f'git switch failed: {exc}')
        self.active_id = tab_id
        self._broadcast_tab_list()

    async def _close_tab(self, tab_id: int):
        tab = self._tab_by_id(tab_id)
        if not tab or tab.is_main:
            return
        if tab.is_running:
            tab.agent.abort()
            if tab.running_task:
                tab.running_task.cancel()
        key = id(tab.agent)
        if key in self._unsubscribers:
            self._unsubscribers[key]()
            del self._unsubscribers[key]
        self.tabs.remove(tab)
        if self.active_id == tab_id:
            self.active_id = self.tabs[0].id
        self._broadcast_tab_list()

async def homepage(request: Request):
    return HTMLResponse(_WEB_HTML)

async def event_stream(request: Request):
    web_repl: WebRepl = request.app.state.web_repl

    async def generator():
        q = web_repl.subscribe()
        try:
            initial = {'type': 'tab_list', 'payload': {'tabs': [{'id': t.id, 'title': t.title, 'is_running': t.is_running, 'status': t.status, 'is_main': t.is_main} for t in web_repl.tabs], 'active_id': web_repl.active_id}, 'tab_id': -1}
            yield f'data: {json.dumps(initial)}\n\n'
            active = web_repl.active_tab
            hist = {'type': 'history', 'payload': {'messages': active.agent.messages}, 'tab_id': active.id}
            yield f'data: {json.dumps(hist, default=str)}\n\n'
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f'data: {json.dumps(event, default=str)}\n\n'
                except asyncio.TimeoutError:
                    yield ': keepalive\n\n'
        except asyncio.CancelledError:
            pass
        finally:
            web_repl.unsubscribe(q)
    return StreamingResponse(generator(), media_type='text/event-stream')

async def send_message(request: Request):
    web_repl: WebRepl = request.app.state.web_repl
    data = await request.json()
    text = data.get('text', '').strip()
    tab_id = data.get('tab_id', web_repl.active_id)
    tab = web_repl._tab_by_id(tab_id)
    if not tab:
        return JSONResponse({'ok': False, 'error': 'Tab not found'}, status_code=404)
    if tab.is_running:
        return JSONResponse({'ok': False, 'error': 'Agent is running. Use /abort to stop.'}, status_code=400)
    await web_repl.run_prompt(tab_id, text)
    return JSONResponse({'ok': True})

async def switch_tab(request: Request):
    web_repl: WebRepl = request.app.state.web_repl
    data = await request.json()
    tab_id = data.get('tab_id')
    tab = web_repl._tab_by_id(tab_id)
    if not tab:
        return JSONResponse({'ok': False, 'error': 'Tab not found'}, status_code=404)
    await web_repl._switch_tab(tab_id)
    return JSONResponse({'ok': True, 'tab_id': tab_id, 'messages': tab.agent.messages})

async def branch(request: Request):
    web_repl: WebRepl = request.app.state.web_repl
    data = await request.json()
    prompt = data.get('prompt', '')
    tab_id = data.get('tab_id', web_repl.active_id)
    tab = web_repl._tab_by_id(tab_id)
    if not tab:
        return JSONResponse({'ok': False, 'error': 'Tab not found'}, status_code=404)
    await web_repl._cmd_branch(tab, prompt)
    return JSONResponse({'ok': True, 'active_id': web_repl.active_id})

async def abort_handler(request: Request):
    web_repl: WebRepl = request.app.state.web_repl
    data = await request.json()
    tab_id = data.get('tab_id', web_repl.active_id)
    tab = web_repl._tab_by_id(tab_id)
    if tab:
        tab.agent.abort()
    return JSONResponse({'ok': True})

async def history(request: Request):
    web_repl: WebRepl = request.app.state.web_repl
    tab_id = int(request.path_params['tab_id'])
    tab = web_repl._tab_by_id(tab_id)
    if not tab:
        return JSONResponse({'error': 'Tab not found'}, status_code=404)
    return JSONResponse({'messages': tab.agent.messages})

async def close_tab(request: Request):
    web_repl: WebRepl = request.app.state.web_repl
    data = await request.json()
    tab_id = data.get('tab_id')
    await web_repl._close_tab(tab_id)
    return JSONResponse({'ok': True, 'active_id': web_repl.active_id})

def run_web(*, base_url=None, model=None, api: Literal['claude', 'codex', 'gemini', 'deepseek', 'noapi']='noapi', system='', sdir=None, skills=None, env=None, tool_names=None, extra_tool_classes=None, api_key=None, gwt=None, ctx=None, init_prompt=None, resume_messages=None, repo=None, resume=None, stream=None, host='127.0.0.1', port=8080):
    repo = os.path.abspath(repo or os.getcwd())
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as _home:
        if gwt is None:
            if resume:
                result = resume_worktree(repo, resume, worktree_dir=os.path.join(_home, 'workspace'))
                if result is None or result[0] is None:
                    print(f'[error] Could not resume from commit {resume!r}. Aborting.')
                    return
                gwt, resume_messages = result
                print(f'[resumed worktree at {gwt.worktree} from commit {resume}]')
            else:
                gwt = create_worktree(repo, worktree_dir=os.path.join(_home, 'workspace'))
        cwd = gwt.worktree if gwt else repo
        if env is None:
            env = os.environ.copy()
        env.setdefault('SHELL', '/bin/bash')
        agent_env = _make_agent_env(env)
        agent_env['HOME'] = _home
        agent_env['PWD'] = cwd
        user_cwd = os.path.abspath(os.getcwd())
        sdir = os.path.abspath(sdir or cwd)
        skills, skill_prompt = collect_skills(sdir, skills)
        system = '\n\n'.join(filter(None, [system, skill_prompt]))
        merged_ctx = {'cwd': cwd, 'user_cwd': user_cwd, 'skills': skills, 'gwt': gwt, 'env': agent_env, **(ctx or {})}
        agent = Agent(system=system, api=api, model=model, tool_names=tool_names, extra_tool_classes=extra_tool_classes, api_key=api_key, base_url=base_url, ctx=merged_ctx)
        log_fp = None
        if stream is not None:
            from .stream_log import StreamLogger
            log_fp = open(stream, 'w', buffering=1)
            agent.ctx['_stream_log_fp'] = log_fp
            agent.ctx['_stream_log_depth'] = 0
            StreamLogger(agent, log_fp, depth=0, name='base')
            print(f'[streaming log: tail -f {stream}]')
        if resume_messages:
            agent.messages = list(resume_messages)
            print(f'[resumed {len(resume_messages)} messages from checkpoint]')
        web_repl = WebRepl(agent, init_prompt=init_prompt)

        @asynccontextmanager
        async def lifespan(app):
            if init_prompt:
                asyncio.create_task(web_repl.run_prompt(0, init_prompt))
            yield
            for t in web_repl.tabs:
                if t.is_running:
                    t.agent.abort()
                    if t.running_task:
                        t.running_task.cancel()
        app = Starlette(routes=[Route('/', homepage, methods=['GET']), Route('/events', event_stream, methods=['GET']), Route('/send', send_message, methods=['POST']), Route('/switch_tab', switch_tab, methods=['POST']), Route('/branch', branch, methods=['POST']), Route('/abort', abort_handler, methods=['POST']), Route('/history/{tab_id}', history, methods=['GET']), Route('/close_tab', close_tab, methods=['POST'])], lifespan=lifespan)
        app.state.web_repl = web_repl
        print(f'[web UI: http://{host}:{port}]')
        config = uvicorn.Config(app, host=host, port=port, log_level='warning')
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        try:
            while server_thread.is_alive():
                server_thread.join(0.1)
        except KeyboardInterrupt:
            print('\nShutting down server...')
        finally:
            if log_fp:
                log_fp.close()
            cleaned = set()
            for t in web_repl.tabs:
                gwt = t.agent.ctx.get('gwt')
                if gwt and hasattr(gwt, 'worktree') and (gwt.worktree not in cleaned):
                    cleaned.add(gwt.worktree)
                    try:
                        cleanup_worktree(gwt)
                    except Exception:
                        pass
            os._exit(0)