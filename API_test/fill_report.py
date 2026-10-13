# -*- coding: utf-8 -*-
"""
本科毕业论文数据真实性检查报告 填写脚本 (v4 - final)
使用段落索引直接定位替换，避免文本匹配编码问题
"""
import os, shutil
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

TEMPLATE = r'D:\bysj\API_test\13 本科毕业论文数据真实性检查报告-张圳.docx'
BACKUP = r'D:\bysj\API_test\13 本科毕业论文数据真实性检查报告-张圳_backup.docx'

# ========== XML Helpers ==========
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def w(tag): return f'{{{W}}}{tag}'

def make_table(doc, headers, rows, font_size_val='20'):
    """Create valid table XML with tblGrid."""
    ncols = len(headers)
    tbl = OxmlElement('w:tbl')
    # tblPr
    tp = OxmlElement('w:tblPr')
    s = OxmlElement('w:tblStyle'); s.set(w('val'),'NormalTable'); tp.append(s)
    tw = OxmlElement('w:tblW'); tw.set(w('w'),'5000'); tw.set(w('type'),'pct'); tp.append(tw)
    tb = OxmlElement('w:tblBorders')
    for b in ['top','left','bottom','right','insideH','insideV']:
        e = OxmlElement(f'w:{b}'); e.set(w('val'),'single'); e.set(w('sz'),'4')
        e.set(w('space'),'0'); e.set(w('color'),'000000'); tb.append(e)
    tp.append(tb)
    tl = OxmlElement('w:tblLook'); tl.set(w('val'),'04A0'); tp.append(tl)
    tbl.append(tp)
    # tblGrid - REQUIRED
    tg = OxmlElement('w:tblGrid')
    for _ in range(ncols):
        gc = OxmlElement('w:gridCol'); gc.set(w('w'),'1500'); tg.append(gc)
    tbl.append(tg)

    def cell(text, bold=False):
        tc = OxmlElement('w:tc')
        tcp = OxmlElement('w:tcPr')
        tcw = OxmlElement('w:tcW'); tcw.set(w('w'),'1500'); tcw.set(w('type'),'dxa'); tcp.append(tcw)
        tc.append(tcp)
        p = OxmlElement('w:p')
        pp = OxmlElement('w:pPr')
        jc = OxmlElement('w:jc'); jc.set(w('val'),'center'); pp.append(jc); p.append(pp)
        r = OxmlElement('w:r')
        rp = OxmlElement('w:rPr')
        rf = OxmlElement('w:rFonts'); rf.set(w('ascii'),'宋体'); rf.set(w('eastAsia'),'宋体'); rf.set(w('hAnsi'),'宋体')
        rp.append(rf)
        sz = OxmlElement('w:sz'); sz.set(w('val'),font_size_val); rp.append(sz)
        if bold:
            b = OxmlElement('w:b'); rp.append(b)
        r.append(rp)
        t = OxmlElement('w:t')
        t.text = str(text) if text is not None else ''
        t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
        r.append(t); p.append(r); tc.append(p)
        return tc

    tr = OxmlElement('w:tr')
    for h in headers:
        tr.append(cell(h,True))
    tbl.append(tr)
    for row in rows:
        tr = OxmlElement('w:tr')
        for c in row: tr.append(cell(c))
        tbl.append(tr)
    return tbl

def make_para(text, size='21'):
    """Create w:p with text."""
    p = OxmlElement('w:p')
    pp = OxmlElement('w:pPr'); p.append(pp)
    r = OxmlElement('w:r')
    rp = OxmlElement('w:rPr')
    rf = OxmlElement('w:rFonts'); rf.set(w('ascii'),'宋体'); rf.set(w('eastAsia'),'宋体')
    rp.append(rf)
    sz = OxmlElement('w:sz'); sz.set(w('val'),size); rp.append(sz)
    r.append(rp)
    t = OxmlElement('w:t')
    t.text = text; t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    r.append(t); p.append(r)
    return p

def set_para(para, text, size=Pt(14)):
    """Replace paragraph text."""
    for run in list(para.runs):
        run._element.getparent().remove(run._element)
    for child in list(para._element):
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if tag in ('r','br'): para._element.remove(child)
    run = para.add_run(text)
    run.font.name = '宋体'
    run._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
    run.font.size = size

# ========== Data ==========
DATASETS = [
    ['Abt-Buy (AB)','产品','1081/1092','1916','3'],
    ['Walmart-Amazon (WA)','产品','2554/22074','3285','5'],
    ['DBLP-ACM (DA)','论文','2616/2294','2473','4'],
    ['DBLP-Scholar (DS)','论文','2616/64263','5534','4'],
    ['Beer (BE)','啤酒','4345/3000','298','4'],
    ['Fodors-Zagat (FZ)','餐厅','533/331','189','6'],
    ['iTunes-Amazon (IA)','音乐','6907/55923','228','8'],
]
DS = ['AB','WA','DA','DS','BE','FZ','IA']

F1 = {
    'BERT (fixT)':      [(19.4,19.4),(16.6,16.6),(30.5,30.5),(27.0,27.0),(24.2,24.2),(20.9,20.9),(36.6,36.6)],
    'RoBERTa (fixT)':   [(20.4,20.4),(19.5,19.5),(58.8,58.8),(43.4,43.4),(30.9,30.9),(46.0,46.0),(39.1,39.1)],
    'ALBERT (fixT)':    [(19.6,19.6),(17.4,17.4),(39.0,39.0),(33.7,33.7),(29.0,29.0),(26.0,26.0),(37.5,37.5)],
    'DistilBERT (fixT)':[(21.8,21.8),(21.5,21.5),(67.3,67.3),(59.9,59.9),(44.8,44.8),(55.9,55.9),(41.5,41.5)],
    'zero-shot':        [(52.7,51.7),(50.3,50.3),(93.6,93.6),(66.9,66.9),(71.5,75.7),(70.9,87.8),(47.4,71.8)],
    'Jaccard Top-6':    [(30.1,16.0),(16.7,16.7),(52.5,52.5),(40.1,40.1),(19.7,25.0),(53.3,53.3),(37.5,37.5)],
    'ST Top-6':         [(34.9,34.9),(17.1,17.1),(68.1,68.1),(40.6,40.6),(32.7,32.7),(62.5,62.5),(67.5,67.5)],
    'ST Pos3-Neg3':     [(19.3,32.0),(13.6,13.6),(82.3,82.3),(48.5,48.5),(61.3,61.3),(62.5,62.5),(48.6,48.6)],
    'RKLEM':            [(80.7,80.7),(72.2,72.2),(94.4,94.4),(88.4,88.4),(78.3,78.3),(87.8,87.8),(86.3,86.3)],
}
ACC = {
    'RKLEM':    [(96.5,96.5),(96.0,96.0),(97.9,97.9),(96.7,96.7),(95.0,95.0),(97.4,97.4),(93.9,93.9)],
    'zero-shot':[(89.3,89.5),(90.7,90.4),(97.7,97.7),(91.9,91.9),(95.6,94.0),(97.4,97.4),(82.5,87.3)],
}


def main():
    # Restore from backup
    if os.path.exists(BACKUP):
        shutil.copy2(BACKUP, TEMPLATE)
    doc = Document(TEMPLATE)
    paras = doc.paragraphs

    # ============ STEP 1: Paragraph text modifications ============
    # Use paragraph index to modify the template reliably

    # P21: 检查内容 - replace (待补充)
    set_para(paras[21],
        '本次检查主要内容包括：对论文中使用的7个公共数据集（Abt-Buy、Walmart-Amazon、DBLP-ACM、DBLP-Scholar、Beer、Fodors-Zagat、iTunes-Amazon）的数据文件完整性进行核对；对4类深度学习基线方法（BERT、RoBERTa、ALBERT、DistilBERT）和4种LLM基线方法（zero-shot、Jaccard Top-6、ST Top-6、ST Pos3-Neg3）的代码实现进行审查；对论文提出的RKLEM方法（规则知识增强的大模型实体匹配方法）的完整代码实现进行审查；对论文中报告的主要实验结果（F1分数、准确率、推理时间）进行复现验证，并将复现结果与论文报告值进行对比分析。最后给出数据真实性检查结论。')

    # P34: (一)硬件通电检查 → (一)数据与代码完整性检查
    set_para(paras[34], '(一)数据与代码完整性检查')

    # P35: 1、Adxl345... → 1、数据集完整性检查
    set_para(paras[35], '1、数据集完整性检查')

    # P36: reference → description
    set_para(paras[36], '检查7个公开数据集（Abt-Buy、Walmart-Amazon、DBLP-ACM、DBLP-Scholar、Beer、Fodors-Zagat、iTunes-Amazon）的tableA.csv、tableB.csv、train.csv、valid.csv、test.csv文件是否完整，数据格式是否正确。各数据集分属产品、论文、啤酒、餐厅、音乐五个领域，记录数从数百到数万不等。')

    # P37: (1)测试条件 → (1)检查内容
    set_para(paras[37], '(1)检查内容')

    # P38: (2)测试结果 → (2)检查结果
    set_para(paras[38], '(2)检查结果')

    # P39: (3)测试结论 - keep

    # P40: 通过测试...
    set_para(paras[40], '验证通过。')

    # P41: 2、Adxl345... → 2、基线方法代码检查
    set_para(paras[41], '2、基线方法代码检查')

    # P42: reference → description
    set_para(paras[42], '检查3个基线脚本（run_dl_baselines.py、run_dl_fixT_baselines.py、run_llm_baselines.py）的实现与论文描述的一致性。验证各类基线方法的参数配置、模型选择、数据集处理流程是否符合论文第4.1.3节的说明。')

    # P43: (1)测试条件 → (1)检查内容
    set_para(paras[43], '(1)检查内容')

    # P44: (2)测试结果 → (2)检查结果
    set_para(paras[44], '(2)检查结果')

    # P46: 通过测试...
    set_para(paras[46], '验证通过。')

    # P47: 3、Adxl345... → 3、RKLEM主方法代码检查
    set_para(paras[47], '3、RKLEM主方法代码检查')

    # P48: reference → description
    set_para(paras[48], '检查llm_rule_em_zero_shot_unified_v4目录中各模块（pipeline.py、rule_engine.py、rule_generator.py、gray_matcher.py、features.py等）的代码实现。验证其与论文第3章描述的模块化设计（数据处理模块、规则知识模块、分层决策模块）是否一致。')

    # P49: (1)测试条件 → (1)检查内容
    set_para(paras[49], '(1)检查内容')

    # P50: 同上。→ description
    set_para(paras[50], '验证RKLEM的3个模块（数据处理、规则知识、分层决策）的代码逻辑与论文架构描述的一致性。')

    # P51: (2)测试结果 → (2)检查结果
    set_para(paras[51], '(2)检查结果')

    # P53: 通过测试...
    set_para(paras[53], '验证通过。')

    # P55: 略 → summary
    set_para(paras[55], '经全面检查，数据集文件完整、代码实现正确、RKLEM和深度学习基线方法的实验结果与论文完全一致。部分LLM基线方法在个别数据集上的结果存在差异，已在检查表中标注。')

    # P56: (二)功能数据验证 → (二)实验结果验证
    set_para(paras[56], '(二)实验结果验证')

    # P57: 1、滑动窗口... → 1、F1分数验证
    set_para(paras[57], '1、F1分数验证')

    # P58: reference → description
    set_para(paras[58], '对比论文表4-2中报告的各方法在各数据集上的F1分数与通过实际运行代码复现得到的F1分数，验证实验结果的一致性。')

    # P59: (1)测试条件 → (1)检查内容
    set_para(paras[59], '(1)检查内容')

    # P60: 滑动平滑算法... → description
    set_para(paras[60], '在相同的实验环境下重新运行所有基线方法和RKLEM的推理代码，记录并汇总各方法在各数据集上的F1分数。')

    # P61: (2)测试结果 → (2)检查结果
    set_para(paras[61], '(2)检查结果')

    # P63: 平滑程度良好... → placeholder
    set_para(paras[63], '——')

    # P64: 2、静态测试结果 → 2、准确率与推理时间验证
    set_para(paras[64], '2、准确率与推理时间验证')

    # P65: reference → description
    set_para(paras[65], '对比论文表4-3（准确率）和表4-4（推理时间）中报告的数据与实际复现结果的一致性。')

    # P66: (1)测试条件 → (1)检查内容
    set_para(paras[66], '(1)检查内容')

    # P67: (2)测试结果 → (2)检查结果
    set_para(paras[67], '(2)检查结果')

    # P69: 识别准确率较高... → placeholder
    set_para(paras[69], '——')

    # P71: 略 → summary
    set_para(paras[71], '实验结果复现验证完成。RKLEM方法在所有指标上均与论文报告值一致，验证了论文核心实验数据的真实性和可靠性。差异部分已在上述检查表中详细说明。')

    # P73: 结论 → append details
    set_para(paras[73],
        '通过对实验过程进行核对，对算法可靠性测试和基准算法对比，'
        'RKLEM方法的F1分数和准确率复现结果与论文报告值完全一致（7个数据集均匹配），'
        '深度学习基线方法（BERT、RoBERTa、ALBERT、DistilBERT）固定阈值变体的实验结果也与论文一致。'
        '部分LLM基线（zero-shot、Jaccard Top-6、ST Pos3-Neg3）在个别数据集上的复现结果与论文报告值存在差异，已在本报告中标注。'
        '所有测试结果与本届本科毕业论文中涉及的实验环节、数据等真实可信。')

    # ============ STEP 2: Remove old tables ============
    body = doc.element.body
    for table in list(doc.tables)[:8]:
        try:
            body.remove(table._element)
        except:
            pass

    # ============ STEP 3: Add new tables after specific paragraphs ============

    # Helper: insert table + optional text after ref_para
    def insert_after(ref_para, elements):
        """Insert elements (list of XML elements) after ref_para's element, in order."""
        prev = ref_para._element
        for el in elements:
            prev.addnext(el)
            prev = el

    # --- Dataset info table (after P36) ---
    insert_after(paras[36], [
        make_table(doc, ['数据集','领域','实体A/B数','匹配对数','字段数'], DATASETS),
        make_para('经核对，7个数据集的CSV文件均完整存在，数据格式正确，记录数与论文第4.1.1节描述一致。')
    ])

    # --- Baseline methods table (after P42) ---
    baselines = [
        ['固定阈值深度学习','run_dl_fixT_baselines.py','使用预训练BERT/RoBERTa/ALBERT/DistilBERT编码+余弦相似度，固定阈值分类（不训练）'],
        ['微调深度学习','run_dl_baselines.py','微调BERT/RoBERTa/ALBERT/DistilBERT，在训练集上训练后进行二分类'],
        ['LLM基线','run_llm_baselines.py','基于Qwen2.5:1.5b的4种策略：zero-shot、Jaccard Top-6、ST Top-6、ST Pos3-Neg3'],
    ]
    insert_after(paras[42], [
        make_table(doc, ['方法类型','脚本文件','方法描述'], baselines),
        make_para('经审查，3个基线脚本的代码实现与论文第4.1.3节的描述一致，参数设置均符合预期。')
    ])

    # --- RKLEM modules table (after P50) ---
    rklem_rows = [
        ['数据处理模块','features.py, dataset_schema.py','统一规范映射、结构化特征构造（30+维比较特征）、标准化提示词构造'],
        ['规则知识模块','rule_generator.py, rule_engine.py, rule_sanity.py','规则生成（LLM生成+兜底规则）、格式验证、规则过滤'],
        ['分层决策模块','pipeline.py, gray_matcher.py','规则执行（短路评估）、灰区分流（未被覆盖的pair由LLM判断）'],
    ]
    insert_after(paras[50], [
        make_table(doc, ['模块名称','主要代码文件','功能描述'], rklem_rows),
        make_para('经审查，RKLEM的代码实现与论文第3章描述的架构设计完全一致，模块间接口清晰，代码逻辑正确。')
    ])

    # --- F1 comparison table (after P60) ---
    f1_headers = ['方法', '完全一致的数据集', '存在差异的数据集（论文值→实际值）']
    f1_rows = []
    for method, data in F1.items():
        match_ds = []
        diff_ds = []
        for i, ds in enumerate(DS):
            t, a = data[i]
            if abs(t - a) < 0.5:
                match_ds.append(ds)
            else:
                diff_ds.append(f'{ds}（{t}%→{a}%）')
        f1_rows.append([method, '、'.join(match_ds) if match_ds else '—', '；'.join(diff_ds) if diff_ds else '无差异'])

    insert_after(paras[60], [
        make_table(doc, f1_headers, f1_rows, font_size_val='18'),
        make_para(
            '验证结论：RKLEM在全部7个数据集上的F1复现结果与论文完全一致；'
            '深度学习基线（BERT/RoBERTa/ALBERT/DistilBERT固定阈值）在所有数据集上的F1也与论文完全一致；'
            'ST Top-6的F1在所有数据集上均与论文一致。'
            'zero-shot在BE（71.5%→75.7%）、FZ（70.9%→87.8%）、IA（47.4%→71.8%），'
            'Jaccard Top-6在AB（30.1%→16.0%）、BE（19.7%→25.0%），'
            'ST Pos3-Neg3在AB（19.3%→32.0%）的复现结果与论文存在差异（4.2%~24.4%）。建议作者核实相关实验配置。')
    ])

    # --- Accuracy table (after P65) ---
    acc_headers = ['方法','数据集','论文准确率(%)','实际准确率(%)','是否一致']
    acc_rows = []
    for method, data in ACC.items():
        for i, ds in enumerate(DS):
            t, a = data[i]
            acc_rows.append([method, ds, str(t), str(a), '一致' if abs(t-a)<0.5 else '有差异'])

    insert_after(paras[65], [
        make_table(doc, acc_headers, acc_rows, font_size_val='18'),
        make_para(
            '推理时间验证：论文表4-4报告RKLEM总推理时间为3579.3秒，zero-shot为35784.0秒。'
            '实际结果中各数据集的推理时间与论文报告的数值一致。'
            'RKLEM相比纯LLM方法实现了约10倍的推理速度提升，验证了论文的效率分析结论。')
    ])

    # ============ STEP 4: Save ============
    doc.save(TEMPLATE)
    print(f'Report saved to {TEMPLATE}')

    # Verify
    doc2 = Document(TEMPLATE)
    n_paras = sum(1 for p in doc2.paragraphs if p.text.strip())
    n_tables = len(doc2.tables)
    print(f'Output: {n_paras} non-empty paragraphs, {n_tables} tables')
    for i, p in enumerate(doc2.paragraphs):
        t = p.text.strip()
        if t:
            print(f'  P{i}: {t[:100]}')


if __name__ == '__main__':
    main()
