#!/usr/bin/env python3
"""Create a standalone, report-style revised Chapter 6 DOCX."""
from __future__ import annotations

from pathlib import Path
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path('/home/luyang/.codex/attachments/5534f3e6-324d-44e2-b592-87a3caa9cd49/技术方案报告0914.docx')
OUTPUT = ROOT / 'output' / '第6章_实验与验证_修订版.docx'


def set_run_font(run, size=12, bold=False, italic=False, color='000000'):
    run.font.name = '宋体'
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts')
        rpr.insert(0, rfonts)
    rfonts.set(qn('w:ascii'), 'Times New Roman')
    rfonts.set(qn('w:hAnsi'), 'Times New Roman')
    rfonts.set(qn('w:eastAsia'), '宋体')


def set_para(paragraph, text='', *, style='Body Text', align=WD_ALIGN_PARAGRAPH.JUSTIFY,
             first_indent=True, before=0, after=0, line=1.25, bold=False, size=12):
    paragraph.style = style
    paragraph.alignment = align
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line
    fmt.first_line_indent = Inches(0.3) if first_indent else Inches(0)
    if text:
        run = paragraph.add_run(text)
        set_run_font(run, size=size, bold=bold)
    return paragraph


def heading(doc, text, level):
    p = doc.add_paragraph(style='Heading 1' if level == 1 else 'Heading 2')
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT
    fmt = p.paragraph_format
    fmt.space_before = Pt(14 if level == 1 else 6)
    fmt.space_after = Pt(7 if level == 1 else 5)
    fmt.line_spacing = 1.25
    fmt.first_line_indent = Inches(0)
    r = p.add_run(text)
    set_run_font(r, size=16 if level == 1 else 14, bold=True)
    return p


def add_label_paragraph(doc, label, text):
    p = doc.add_paragraph(style='Body Text')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt = p.paragraph_format
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(3)
    fmt.line_spacing = 1.25
    fmt.first_line_indent = Inches(0.3)
    r = p.add_run(label)
    set_run_font(r, size=12, bold=True)
    r = p.add_run(text)
    set_run_font(r, size=12)
    return p


def set_cell_shading(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = tcpr.find(qn('w:shd'))
    if shd is None:
        shd = OxmlElement('w:shd')
        tcpr.append(shd)
    shd.set(qn('w:fill'), fill)
    shd.set(qn('w:val'), 'clear')


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110):
    tcpr = cell._tc.get_or_add_tcPr()
    tcmt = tcpr.first_child_found_in('w:tcMar')
    if tcmt is None:
        tcmt = OxmlElement('w:tcMar')
        tcpr.append(tcmt)
    for m, v in [('top', top), ('start', start), ('bottom', bottom), ('end', end)]:
        node = tcmt.find(qn(f'w:{m}'))
        if node is None:
            node = OxmlElement(f'w:{m}')
            tcmt.append(node)
        node.set(qn('w:w'), str(v))
        node.set(qn('w:type'), 'dxa')


def set_cell_borders(cell, color='D9D9D9', size='6'):
    tcpr = cell._tc.get_or_add_tcPr()
    borders = tcpr.first_child_found_in('w:tcBorders')
    if borders is None:
        borders = OxmlElement('w:tcBorders')
        tcpr.append(borders)
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        tag = qn(f'w:{edge}')
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f'w:{edge}')
            borders.append(element)
        element.set(qn('w:val'), 'single')
        element.set(qn('w:sz'), size)
        element.set(qn('w:space'), '0')
        element.set(qn('w:color'), color)


def repeat_header(row):
    trpr = row._tr.get_or_add_trPr()
    header = OxmlElement('w:tblHeader')
    header.set(qn('w:val'), 'true')
    trpr.append(header)


def write_table(doc, headers, rows, widths, caption):
    cap = doc.add_paragraph(style='Body Text')
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(4)
    cap.paragraph_format.first_line_indent = Inches(0)
    cap.paragraph_format.line_spacing = 1.0
    cap.paragraph_format.keep_with_next = True
    set_run_font(cap.add_run(caption), size=10.5, bold=True)

    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.allow_autofit = False
    for i, width in enumerate(widths):
        table.columns[i].width = Inches(width)
    head = table.rows[0]
    repeat_header(head)
    for i, value in enumerate(headers):
        cell = head.cells[i]
        cell.width = Inches(widths[i])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, '2F5597')
        set_cell_margins(cell)
        set_cell_borders(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Inches(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.clear()
        set_run_font(p.add_run(value), size=10.5, bold=True, color='FFFFFF')
    for ri, row_values in enumerate(rows):
        row = table.add_row()
        for i, value in enumerate(row_values):
            cell = row.cells[i]
            cell.width = Inches(widths[i])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_shading(cell, 'F2F6FB' if ri % 2 else 'FFFFFF')
            set_cell_margins(cell)
            set_cell_borders(cell)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.clear()
            set_run_font(p.add_run(str(value)), size=10.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def main():
    doc = Document(str(REFERENCE))
    body = doc._element.body
    for child in list(body):
        if child.tag != qn('w:sectPr'):
            body.remove(child)
    section = doc.sections[0]
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.25)
    section.right_margin = Inches(1.25)

    heading(doc, '第6章 实验与验证', 1)

    heading(doc, '6.1 原型系统设计与实验方案', 2)
    heading(doc, '6.1.1 实验目标', 2)
    set_para(doc.add_paragraph(), '本章围绕语音交互链路的实时性开展验证，重点测量ASR、Agent、TTS以及端到端总延迟，并通过长期记忆和情绪分析模块的开关对比，观察模块配置对上述延迟指标的影响。回答质量评估不在本节提前下结论，单独为后续112条高辨别力样本的盲评结果预留位置。')

    heading(doc, '6.1.2 实验环境与数据', 2)
    set_para(doc.add_paragraph(), '实验运行在与生产服务隔离的测试服务和独立SQLite数据库中，使用项目当前实际配置的流式语音识别、Agent回复生成、语音合成和情绪分析链路。测试使用虚构用户资料与人工设计的合成语音场景，不接入真实用户录音，不保存真实个人交互数据。每次试验均创建新的会话，长期记忆写入关闭，配对试验使用相同输入波形和相同用户资料。')
    set_para(doc.add_paragraph(), '首轮正式实验包含12个语音场景，每个场景在四种模块配置下重复两次，共计划96条正式试验；另设置4条预热试验。因8条试验未形成完整的可用事件链路，最终纳入本章统计的正式试验为88条。正在进行的112条高辨别力样本实验不提前计入本章结果。')

    heading(doc, '6.1.3 实验分组与指标定义', 2)
    set_para(doc.add_paragraph(), '实验采用2×2消融设计：M1E1表示记忆开、情绪开；M0E1表示记忆关、情绪开；M1E0表示记忆开、情绪关；M0E0表示记忆关、情绪关。四种配置在每个场景内随机排列，避免固定顺序造成影响。')
    add_label_paragraph(doc, 'ASR实时首段延迟：', '从有效语音开始到首次非空流式识别结果。该指标用于表示用户开始说话后系统首次识别出内容的时间。')
    add_label_paragraph(doc, 'ASR最终转写延迟：', '从语音段结束后的最终识别阶段到系统发布稳定最终文本的时间。')
    add_label_paragraph(doc, 'Agent延迟：', '从最终ASR文本发布到Agent产生首段非空文字的时间。')
    add_label_paragraph(doc, 'TTS延迟：', '从Agent首段文字产生到TTS输出首个非空音频块的时间。')
    add_label_paragraph(doc, '总延迟：', '从用户最后一段有效语音结束到系统输出首个TTS音频块的时间，表示用户感知到的端到端首响延迟。')

    heading(doc, '6.2 实验结果与分析', 2)
    heading(doc, '6.2.1 88条有效试验的总体延迟', 2)
    write_table(doc,
                ['阶段', '时间区间', '中位数（ms）', '平均值（ms）', 'P95（ms）'],
                [
                    ['ASR', '开始说话→首次流式识别', '862.7', '930.0', '1254.8'],
                    ['ASR', 'ASR结束判定→最终转写', '109.6', '138.2', '142.5'],
                    ['Agent', '最终转写→Agent首段文字', '1664.9', '1701.9', '2169.8'],
                    ['TTS', 'Agent首段文字→TTS首音频', '295.3', '305.9', '333.9'],
                    ['总延迟', '最后语音→TTS首音频', '3418.4', '3438.5', '3911.9'],
                ],
                [0.85, 2.65, 0.85, 0.85, 0.8],
                '表6.1 88条有效试验的总体延迟结果')
    set_para(doc.add_paragraph(), '从总体结果看，ASR最终转写阶段的延迟约为110 ms，Agent首段文字生成是首响链路中耗时最长的阶段之一，TTS从收到首段文字到产生首个音频块约需295 ms。端到端总延迟的中位数为3418.4 ms，表示用户停止说话后平均约3.42秒能够收到第一段语音回复。')

    heading(doc, '6.2.2 四种模块配置下的延迟对比', 2)
    write_table(doc,
                ['配置', 'ASR首段（ms）', 'ASR最终（ms）', 'Agent（ms）', 'TTS（ms）', '总延迟（ms）'],
                [
                    ['M1E1 记忆开·情绪开', '868.2', '109.2', '1739.6', '292.2', '3510.3'],
                    ['M0E1 记忆关·情绪开', '851.3', '111.6', '1664.9', '299.7', '3393.0'],
                    ['M1E0 记忆开·情绪关', '878.8', '106.4', '1686.7', '291.2', '3383.2'],
                    ['M0E0 记忆关·情绪关', '854.5', '112.1', '1635.1', '297.1', '3350.4'],
                ],
                [1.65, 0.82, 0.82, 0.82, 0.82, 0.82],
                '表6.2 四种模块配置下的延迟中位数')
    set_para(doc.add_paragraph(), '四种配置下的ASR首段延迟、ASR最终转写延迟和TTS首音频延迟总体接近，当前数据未显示记忆或情绪模块对这些阶段产生稳定的固定延迟。Agent阶段和总延迟存在一定波动，但该差异同时受到场景内容、模型生成长度和网络抖动影响，现阶段仅作描述性观察，不据此作显著性结论。')

    heading(doc, '6.2.3 回答质量评估（112条高辨别力样本，待补充）', 2)
    set_para(doc.add_paragraph(), '为进一步放大长期记忆和情绪模块之间的质量差异，后续将使用14个高辨别力场景，在四种模块配置下各重复两次，共112条系统回答。该部分采用匿名盲评，不在本版本中填入未完成或未经复核的结果。')
    write_table(doc,
                ['评估项目', '设计内容', '结果'],
                [
                    ['样本规模', '14个场景×4种配置×2次重复，共112条回答', '待补充'],
                    ['记忆相关场景', '记忆依赖、事实更新、记忆陷阱和无关控制', '待补充'],
                    ['情绪相关场景', '相同文本、不同语气的成对样本', '待补充'],
                    ['盲评维度', '记忆相关性、事实一致性、情绪匹配、共情、边界遵守、帮助性和安全性', '待补充'],
                    ['统计方式', '四条件配对差值、置信区间、偏好和硬错误标记', '待补充'],
                ],
                [1.15, 3.55, 1.05],
                '表6.3 112条高辨别力样本回答质量评估预留表')

    heading(doc, '6.3 实验结论与限制', 2)
    set_para(doc.add_paragraph(), '本轮实验完成了对ASR、Agent、TTS和端到端总延迟的统一测量。结果显示，当前链路的总体首响中位数为3418.4 ms，其中ASR最终转写约109.6 ms，Agent首段文字生成约1664.9 ms，TTS首音频启动约295.3 ms。后续降低总延迟时，应优先优化Agent首段生成和端到端等待策略。')
    set_para(doc.add_paragraph(), '本章结果来自合成语音和环回事件链路，不包含真实麦克风、扬声器、浏览器渲染和物理播放延迟。8条未形成完整事件链路的试验未纳入统计，但原始记录已保留。112条高辨别力样本完成后，应将回答质量结果补入表6.3，并根据盲评结果更新本章结论。')

    # Keep the report's existing footer/header package parts; add a compact footer note.
    footer = section.footer
    if footer.paragraphs:
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fp.clear()
        set_run_font(fp.add_run('第6章 实验与验证'), size=9, color='666666')

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUTPUT))
    print(OUTPUT)


if __name__ == '__main__':
    main()
