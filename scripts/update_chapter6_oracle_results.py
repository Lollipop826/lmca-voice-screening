from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


SOURCE = Path('/data/luyang/lmca-share/output/第6章_实验与验证_LY_实验结果更新版.docx')
OUTPUT = Path('/data/luyang/lmca-share/output/第6章_实验与验证_LY_正确标签实验版_含112条场景表.docx')
BLUE, PALE, BORDER = '2F5B9C', 'EAF0F8', '8EA9C1'
BLACK, WHITE = RGBColor(0, 0, 0), RGBColor(255, 255, 255)


def font(run, size=9.5, bold=False, color=BLACK):
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.highlight_color = None
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts')
        rpr.insert(0, rfonts)
    for key in ('ascii', 'hAnsi'):
        rfonts.set(qn('w:' + key), 'Times New Roman')
    rfonts.set(qn('w:eastAsia'), '宋体')


def replace_para(p, text, style=None, size=12, bold=False):
    p.clear()
    if style:
        p.style = style
    r = p.add_run(text)
    font(r, size=size, bold=bold)
    p.paragraph_format.first_line_indent = Pt(24) if style != 'Heading 2' else Pt(0)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.35
    return p


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    node = tcpr.find(qn('w:shd'))
    if node is None:
        node = OxmlElement('w:shd')
        tcpr.append(node)
    node.set(qn('w:fill'), fill)


def margins(cell):
    tcpr = cell._tc.get_or_add_tcPr()
    mar = tcpr.first_child_found_in('w:tcMar')
    if mar is None:
        mar = OxmlElement('w:tcMar')
        tcpr.append(mar)
    for tag, value in (('top', 70), ('start', 85), ('bottom', 70), ('end', 85)):
        node = mar.find(qn('w:' + tag))
        if node is None:
            node = OxmlElement('w:' + tag)
            mar.append(node)
        node.set(qn('w:w'), str(value))
        node.set(qn('w:type'), 'dxa')


def style_table(table, widths, font_size=9.5, left_cols=None):
    left_cols = set(left_cols or ())
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tblpr = table._tbl.tblPr
    borders = tblpr.find(qn('w:tblBorders'))
    if borders is None:
        borders = OxmlElement('w:tblBorders')
        tblpr.append(borders)
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        e = borders.find(qn('w:' + edge))
        if e is None:
            e = OxmlElement('w:' + edge)
            borders.append(e)
        e.set(qn('w:val'), 'single')
        e.set(qn('w:sz'), '6')
        e.set(qn('w:color'), BORDER)
    trpr = table.rows[0]._tr.get_or_add_trPr()
    if trpr.find(qn('w:tblHeader')) is None:
        trpr.append(OxmlElement('w:tblHeader'))
    for ridx, row in enumerate(table.rows):
        if row._tr.get_or_add_trPr().find(qn('w:cantSplit')) is None:
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        for cidx, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            margins(cell)
            if cidx < len(widths):
                cell.width = Inches(widths[cidx])
            shade(cell, BLUE if ridx == 0 else (PALE if ridx % 2 == 0 else 'FFFFFF'))
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ridx and cidx in left_cols else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.05
                for r in p.runs:
                    font(r, size=font_size, bold=(ridx == 0), color=WHITE if ridx == 0 else BLACK)


def replace_table(table, rows, widths):
    if len(table.rows) != len(rows):
        raise ValueError('unexpected table row count')
    for ridx, values in enumerate(rows):
        if len(values) != len(table.columns):
            raise ValueError('unexpected table column count')
        for cidx, value in enumerate(values):
            table.cell(ridx, cidx).text = str(value)
    style_table(table, widths)


def insert_paragraph_after(paragraph, text, style=None, size=10.5, bold=False):
    new_p = OxmlElement('w:p')
    paragraph._p.addnext(new_p)
    inserted = Paragraph(new_p, paragraph._parent)
    if style:
        inserted.style = style
    r = inserted.add_run(text)
    font(r, size=size, bold=bold)
    inserted.paragraph_format.first_line_indent = Pt(24) if style != 'Heading 2' else Pt(0)
    inserted.paragraph_format.space_after = Pt(5)
    inserted.paragraph_format.line_spacing = 1.35
    return inserted


def main():
    doc = Document(SOURCE)
    p = doc.paragraphs
    replace_para(p[27], '6.2.3 回答质量评估（112条高辨别力样本与正确标签复验）', 'Heading 2', 16, True)
    replace_para(p[28], '回答质量实验首先完成112条高辨别力样本的四配置盲评；随后，为隔离上游语音情绪识别误差，使用同一批14个场景和同一2×2设计进行正确情绪标签注入复验。两轮均包含14个场景、4种模块配置和2次重复，分别形成28个配对单元和84次盲评任务。')
    replace_para(p[29], '评估时隐藏模块配置，将同一场景的四条回答匿名映射为A、B、C、D，使用qwen-flash作为模型评审器，对每个配对单元重复评分3次。评分维度包括记忆相关性、连续性、事实一致性、情绪匹配、共情、校准性、边界遵守、帮助性与安全性。下述记忆结果来自首轮112条实验；情绪结果来自正确标签注入复验。表中“平均分差”均为同一配对单元内开启模块减去关闭模块，正值表示开启后得分更高。')
    replace_para(p[30], '表6.4 回答质量两轮实验与盲评完成情况', 'Body Text', 10.5, True).alignment = WD_ALIGN_PARAGRAPH.CENTER
    replace_table(doc.tables[5], [
        ['项目', '计划量', '完成量', '说明'],
        ['首轮系统回答', '112条', '112条', '14场景×4配置×2次'],
        ['正确标签注入复验', '112条', '112条', '情绪开启组直接注入预设正确标签'],
        ['配对单元', '56组', '56组', '两轮各28组，同场景条件内配对'],
        ['盲评任务', '168次', '168次', '两轮各84次，每组重复3次'],
        ['失败记录', '—', '0次', '无API或格式解析失败'],
    ], [1.25, 0.95, 0.95, 2.85])
    replace_para(p[31], '在首轮112条实验的记忆依赖场景中，开启长期记忆后，记忆相关性、对话连续性、共情和帮助性均呈稳定正向变化；事实一致性为正向趋势。具体结果见表6.5。')
    replace_para(p[33], '在首轮实验的记忆陷阱和中性控制场景中，各配置差异接近0，说明长期记忆的增益主要集中在确实需要调用历史信息的场景。')
    replace_para(p[34], '正确标签注入复验只改变情绪开启组的内部标签，不改变用户文本、记忆条件、场景顺序或评分流程。非中性情绪场景（喜悦、低落、生气、紧张）的8个配对单元中，正确标签使情绪匹配、共情、校准性和帮助性均出现正向提升；边界遵守基本不变。该结果支持下游Agent在获得正确标签后能够改善回应，但不代表Emotion2Vec本身的识别准确率。结果见表6.6。')
    replace_para(p[32], '表6.5 记忆依赖场景中长期记忆模块的配对评分结果', 'Body Text', 10.5, True).alignment = WD_ALIGN_PARAGRAPH.CENTER
    replace_table(doc.tables[7], [
        ['评价维度', '平均分差', '95%置信区间', '胜/平/负', '结果解释'],
        ['情绪匹配', '+0.5417', '[0.1667, 0.9583]', '4/4/0', '稳定正向'],
        ['共情', '+0.6042', '[0.2083, 1.0000]', '5/2/1', '稳定正向'],
        ['校准性', '+0.4375', '[0.0417, 0.8958]', '5/1/2', '稳定正向'],
        ['边界遵守', '+0.0625', '[-0.0625, 0.2500]', '1/6/1', '无稳定差异'],
        ['帮助性', '+0.5000', '[0.1042, 0.8958]', '5/2/1', '稳定正向'],
    ], [1.15, 1.05, 1.55, 0.95, 1.30])
    replace_para(p[35], '表6.6 正确情绪标签注入复验的非中性情绪配对结果', 'Body Text', 10.5, True).alignment = WD_ALIGN_PARAGRAPH.CENTER
    replace_para(p[36], '在正确标签注入复验的84次盲评任务中，四配置总体首选以并列为主（67次，79.8%）；M1E0、M0E1、M1E1和M0E0分别获得9、4、2和2次单独首选。四选一偏好同时混合了记忆、中性控制和情绪场景，因此情绪模块的主要判断以表6.6的成对效应为准。')
    replace_para(p[37], '表6.7 正确标签注入复验的四配置总体首选分布（84次盲评任务）', 'Body Text', 10.5, True).alignment = WD_ALIGN_PARAGRAPH.CENTER
    replace_table(doc.tables[8], [
        ['首选结果', '次数', '占比', '说明'],
        ['并列', '67', '79.8%', '未形成唯一首选'],
        ['M1E0', '9', '10.7%', '记忆开、情绪关'],
        ['M0E1', '4', '4.8%', '记忆关、情绪开'],
        ['M1E1', '2', '2.4%', '记忆开、情绪开'],
        ['M0E0', '2', '2.4%', '记忆关、情绪关'],
    ], [1.3, 0.85, 0.85, 3.0])
    replace_para(p[40], '回答质量方面，首轮112条实验显示长期记忆在记忆依赖场景中具有正向作用；正确标签注入复验显示，在非中性情绪场景中，情绪匹配、共情、校准性和帮助性分别提升0.5417、0.6042、0.4375和0.5000分。该复验结果验证了下游Agent利用正确情绪标签的潜在收益，但不能替代真实语音情绪识别准确率测试。')
    replace_para(p[41], '本章延迟结果来自首轮合成语音和环回事件链路，不包含真实麦克风、扬声器、浏览器渲染和物理播放延迟；8条未形成完整事件链路的首轮试验未纳入延迟统计，但原始记录已保留。回答质量包含首轮112条实验和正确标签注入复验各112条回答，共168次盲评任务；评分由同一模型评审器重复3次完成，不等同于3名独立人工评审。正确标签来自场景预设而非人工听音标注，复验只能说明下游Agent在理想标签输入下的效果，不能据此宣称Emotion2Vec识别准确。')

    # Add the new 112-sample composition table after the quality-design paragraph.
    intro = insert_paragraph_after(
        p[28],
        '本轮新增的112条并不是112个完全不同的场景，而是14个高辨别力语音场景在四种模块配置（M1E1、M1E0、M0E1、M0E0）下各重复两次；因此每个场景形成8条回答，14×8=112。其中记忆依赖4个场景、32条回答，记忆陷阱2个场景、16条回答，情绪成对6个场景、48条回答，中性控制2个场景、16条回答。样本既检验模块开启后的收益，也检验不应调用记忆或情绪时的克制能力。下表列出每个场景的测试内容。',
    )
    caption = insert_paragraph_after(
        intro,
        '表6.3 112条高辨别力样本的场景构成',
        'Body Text',
        10.5,
        True,
    )
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    note = insert_paragraph_after(
        caption,
        '注：预设标签仅用于正确标签注入复验；标签来自场景设计，不是人工听音标注。因此该表用于解释112条样本的实验构成和辨别目标，不用于证明上游语音情绪识别准确率。',
        size=9.2,
    )
    sample_rows = [
        ['序号', '类别', '测试语句', '预设标签', '辨别目标', '每场景'],
        ['1', '记忆依赖·家庭', '我这两天还是因为女儿不回消息觉得难受。', '悲伤', '衔接家庭沟通背景，不编造女儿想法。', '8条'],
        ['2', '记忆依赖·睡眠', '最近又睡不好，晚上总忍不住看工作消息。', '平静', '调用睡眠与手机习惯，不武断诊断。', '8条'],
        ['3', '记忆依赖·指代', '上次说的睡前那个办法我试了两天，还是拿不准下一步。', '困惑', '解析“那个办法”指代，保持方法连续性。', '8条'],
        ['4', '记忆依赖·更新', '我和女儿已经把误会说开了，今天想分享她工作上的好消息。', '喜悦', '以新事实覆盖旧冲突，不沿用过时记忆。', '8条'],
        ['5', '记忆陷阱·纠正', '散步现在改到周五下午了，周三要上课，别再按旧时间记了。', '平静', '接受事实更新，不继续强调旧时间。', '8条'],
        ['6', '记忆陷阱·无关', '今天买菜回来发现钥匙忘在家里了，在门口等了半天。', '平静', '回应眼前经历，不强行引用家庭、睡眠或散步历史。', '8条'],
        ['7', '情绪成对·完成积极', '这件事终于完成了，我现在想和你说说我的感受。', '喜悦', '匹配积极语气，但不过度夸大。', '8条'],
        ['8', '情绪成对·完成低落', '这件事终于完成了，我现在想和你说说我的感受。', '低落', '识别低落语气，不把“完成”自动解释为开心。', '8条'],
        ['9', '情绪成对·边界生气', '我希望这次能按照我说的方式来处理。', '生气', '识别边界和生气语气，不激化冲突。', '8条'],
        ['10', '情绪成对·边界平静', '我希望这次能按照我说的方式来处理。', '平静', '保持平实回应，不凭空添加愤怒或委屈。', '8条'],
        ['11', '情绪成对·办事紧张', '明天我需要一个人去办理这件事。', '紧张', '提供轻量、可执行支持，不过度诊断。', '8条'],
        ['12', '情绪成对·办事平静', '明天我需要一个人去办理这件事。', '平静', '保持平实，不把普通计划误读为焦虑。', '8条'],
        ['13', '中性控制·实用', '冰箱里还有两个鸡蛋和一根黄瓜，能做点什么简单的？', '平静', '检验直接、可执行的实用帮助，不做心理分析。', '8条'],
        ['14', '中性控制·日常', '今天下午有空的话，我想整理一下书桌。', '平静', '检验自然、轻量回应，不强行引用历史或情绪。', '8条'],
    ]
    table = doc.add_table(rows=len(sample_rows), cols=len(sample_rows[0]))
    for ridx, values in enumerate(sample_rows):
        for cidx, value in enumerate(values):
            table.cell(ridx, cidx).text = value
    style_table(table, [0.38, 1.22, 2.03, 0.70, 1.67, 0.46], font_size=8.2, left_cols={1, 2, 4})
    note._p.addprevious(table._tbl)
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            run.font.highlight_color = None
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == '__main__':
    main()
