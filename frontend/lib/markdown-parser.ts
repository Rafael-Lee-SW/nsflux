/**
 * 마크다운 텍스트를 HTML로 변환하는 함수
 * @param text 변환할 마크다운 텍스트
 * @returns HTML 문자열
 */
export function parseMarkdown(text: string): string {
    if (!text) return '';

    // 특수 테이블 포맷 처리
    text = text.replace(/<<<TABLE>>>([\s\S]*?)<<<END_TABLE>>>/g, (match, tableContent) => {
        return convertMarkdownTable(tableContent);
    });

    // 일반 텍스트 테이블 처리
    text = parsePlainTextTables(text);

    // 각주 정리
    const footnoteRegex = /^\[\^(\d+)\]:\s*(.+)$/gm;
    const footnotesDict: Record<string, string> = {};
    let match;
    while ((match = footnoteRegex.exec(text)) !== null) {
        const num = match[1];
        const content = match[2];
        footnotesDict[num] = content;
    }
    text = text.replace(footnoteRegex, '').trim();

    // 이미 HTML인 경우 그대로 반환
    if (text.trim().startsWith('<')) {
        return text;
    }

    // 코드 블록 처리
    text = text.replace(/\`\`\`([a-z]*)\n([\s\S]*?)\`\`\`/g, (m, lang, code) => {
        return `<pre><code class="language-${lang}">${escapeHtml(code.trim())}</code></pre>`;
    });

    // 인라인 서식 변환
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

    // 헤더 처리
    text = text.replace(/^###### (.*)$/gm, '<h6>$1</h6>');
    text = text.replace(/^##### (.*)$/gm, '<h5>$1</h5>');
    text = text.replace(/^#### (.*)$/gm, '<h4>$1</h4>');
    text = text.replace(/^### (.*)$/gm, '<h3>$1</h3>');
    text = text.replace(/^## (.*)$/gm, '<h2>$1</h2>');
    text = text.replace(/^# (.*)$/gm, '<h1>$1</h1>');

    // 목록 처리
    text = text.replace(/^\s*\* (.*)$/gm, '<li>$1</li>');
    text = text.replace(/^\s*\d+\. (.*)$/gm, '<li>$1</li>');
    text = text.replace(/(<li>[\s\S]*?<\/li>)/gm, (m) => {
        if (m.trim().startsWith('<ul>')) return m;
        return '<ul>' + m + '</ul>';
    });

    // 인용문
    text = text.replace(/^> (.*)$/gm, '<blockquote>$1</blockquote>');

    // 문단 처리
    const blocks = text.split(/\n\s*\n/);
    const blockLevelTags = ['<h1>', '<h2>', '<h3>', '<ul>', '<ol>', '<blockquote>', '<pre>'];
    const htmlBlocks = blocks.map(block => {
        const trimmed = block.trim();
        for (let tag of blockLevelTags) {
            if (trimmed.startsWith(tag)) {
                return trimmed;
            }
        }
        const replaced = trimmed.replace(/\n/g, '<br>');
        return `<p>${replaced}</p>`;
    });
    let html = htmlBlocks.join('');

    // 마크다운 테이블 처리
    html = parseMarkdownTables(html);

    // 각주 처리
    const footnotesKeys = Object.keys(footnotesDict);
    if (footnotesKeys.length > 0) {
        let footnotesHTML = '<div class="footnotes"><hr><ol>';
        footnotesKeys.sort((a, b) => Number(a) - Number(b)).forEach(num => {
            footnotesHTML += `<li id="fn${num}">${footnotesDict[num]} <a href="#ref${num}" title="Back to content">↩</a></li>`;
        });
        footnotesHTML += '</ol></div>';
        html += footnotesHTML;
    }

    return html;
}

/**
 * HTML 특수 문자를 이스케이프하는 함수
 */
function escapeHtml(unsafe: string): string {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/**
 * 마크다운 테이블을 HTML 테이블로 변환하는 함수
 */
function parseMarkdownTables(text: string): string {
    const tableRegex = /((?:^\|.*\|(?:\r?\n|$))+)/gm;
    return text.replace(tableRegex, function (match) {
        return convertMarkdownTable(match);
    });
}

/**
 * 마크다운 테이블을 HTML 테이블로 변환하는 유틸리티 함수
 */
function convertMarkdownTable(tableContent: string): string {
    const lines = tableContent.trim().split(/\r?\n/);
    if (lines.length < 2) return tableContent;

    const headers = lines[0].split('|').map(s => s.trim()).filter(s => s);
    const dataRows = lines.slice(2).map(line => line.split('|').map(s => s.trim()).filter(s => s));

    let html = '<table border="1" cellpadding="5" cellspacing="0"><thead><tr>';
    headers.forEach(h => { html += `<th>${h}</th>`; });
    html += '</tr></thead><tbody>';

    dataRows.forEach(row => {
        html += '<tr>';
        row.forEach(cell => { html += `<td>${cell}</td>`; });
        html += '</tr>';
    });

    html += '</tbody></table>';
    return html;
}

/**
 * 일반 텍스트 테이블(스페이스로 구분된)을 HTML 테이블로 변환하는 함수
 */
function parsePlainTextTables(text: string): string {
    text = text.replace(/\t/g, "    ");
    const lines = text.split(/\r?\n/);
    let outputLines = [];
    let i = 0;

    while (i < lines.length) {
        if (lines[i].trim() && lines[i].match(/\S+(?:\s{2,}\S+)+/)) {
            const headerLine = lines[i];
            if (i + 1 < lines.length && lines[i + 1].trim().match(/^[-\s]+$/)) {
                let tableLines = [];
                tableLines.push(headerLine);
                tableLines.push(lines[i + 1]);
                i += 2;

                while (i < lines.length && lines[i].trim() && lines[i].match(/\S+(?:\s{2,}\S+)+/)) {
                    tableLines.push(lines[i]);
                    i++;
                }

                outputLines.push(convertPlainTextTableToHTML(tableLines));
                continue;
            }
        }
        outputLines.push(lines[i]);
        i++;
    }

    return outputLines.join("\n");
}

/**
 * 일반 텍스트 테이블을 HTML 테이블로 변환하는 유틸리티 함수
 */
function convertPlainTextTableToHTML(lines: string[]): string {
    const headerLine = lines[0];
    const headers = headerLine.trim().split(/\s{2,}/);

    let html = '<table border="1" cellpadding="5" cellspacing="0"><thead><tr>';
    headers.forEach(h => { html += `<th>${h}</th>`; });
    html += '</tr></thead><tbody>';

    for (let j = 2; j < lines.length; j++) {
        const row = lines[j].trim().split(/\s{2,}/);
        html += '<tr>';
        row.forEach(cell => {
            html += `<td>${cell}</td>`;
        });
        html += '</tr>';
    }

    html += '</tbody></table>';
    return html;
}