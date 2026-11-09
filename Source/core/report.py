"""Xuat bao cao moi ca du doan: Word (.docx) va PDF."""
import io
from datetime import datetime
from pathlib import Path
import matplotlib
from .io import CLASSES, CLASS_FULL
from . import viz

DISCLAIMER = ("Công cụ phục vụ nghiên cứu/học thuật trên bộ dữ liệu OASIS-1. Kết quả KHÔNG phải chẩn đoán y khoa "
              "và không được dùng để quyết định điều trị. Nguồn xác suất và cấu hình mô hình được ghi trong phiếu; "
              "xác suất phân loại chưa được xác nhận là nguy cơ mắc bệnh trong tương lai.")


def _rows(detail):
    rows = []
    for name, m in detail["models"].items():
        p = m["mean_prob"] * 100
        votes = ", ".join(f"seed {s}: {CLASSES[v]}" for s, v in m["seed_preds"].items())
        rows.append([name, CLASSES[m["pred"]], "Đúng" if m["pred"] == detail["y_true"] else "Sai",
                     f"{p[0]:.1f}", f"{p[1]:.1f}", f"{p[2]:.1f}", votes])
    return rows


def _summary_line(detail):
    t = CLASSES[detail["y_true"]]
    preds = {n: CLASSES[m["pred"]] for n, m in detail["models"].items()}
    if detail["agree"]:
        return f"Các mô hình đồng thuận dự đoán {list(preds.values())[0]}; nhãn tham chiếu là {t}."
    return "Các mô hình bất đồng: " + "; ".join(f"{n} → {p}" for n, p in preds.items()) + f". Nhãn tham chiếu là {t}."


def build_docx(detail, notes="", cams=None, include_chart=True):
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    doc = Document()
    for s in doc.sections: s.left_margin = s.right_margin = Cm(2); s.top_margin = s.bottom_margin = Cm(1.8)
    st = doc.styles["Normal"]; st.font.name = "Arial"; st.font.size = Pt(10.5)
    h = doc.add_heading("BÁO CÁO DỰ ĐOÁN — TriFuse-AD Demo", 0); h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f"Mã ca: {detail['subject_id']}    |    Ngày xuất: {datetime.now():%d/%m/%Y %H:%M}")
    t = CLASSES[detail["y_true"]]
    doc.add_paragraph(f"Nhãn tham chiếu (theo CDR): {t} — {CLASS_FULL[t]}")
    doc.add_heading("1. Kết quả các mô hình", 1)
    tb = doc.add_table(rows=1, cols=7); tb.style = "Light Grid Accent 1"
    for c, txt in zip(tb.rows[0].cells, ["Mô hình", "Dự đoán", "Đúng/Sai", "CN %", "VMD %", "AD %", "Theo seed"]):
        c.text = txt
    for r in _rows(detail):
        cells = tb.add_row().cells
        for c, txt in zip(cells, r): c.text = str(txt)
    doc.add_paragraph()
    doc.add_paragraph(_summary_line(detail))
    if include_chart:
        doc.add_picture(io.BytesIO(viz.fig_bytes(viz.case_prob_figure(detail, detail["y_true"]))), width=Cm(13))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if cams:
        doc.add_heading("2. Giải thích mô hình (Grad-CAM)", 1)
        for title, png in cams:
            doc.add_paragraph(title).runs[0].bold = True
            doc.add_picture(io.BytesIO(png), width=Cm(16.5))
        doc.add_paragraph("Vùng nóng (đỏ/vàng) là vùng ảnh hưởng nhiều nhất tới điểm của lớp được chọn. Đây là công cụ gợi ý, "
                          "không chứng minh quan hệ nhân quả.")
    if notes.strip():
        doc.add_heading("Ghi chú", 1); doc.add_paragraph(notes.strip())
    p = doc.add_paragraph(); r = p.add_run("Lưu ý: " + DISCLAIMER); r.italic = True; r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    buf = io.BytesIO(); doc.save(buf)
    return buf.getvalue()


def _register_fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DJ", str(base / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DJB", str(base / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DJI", str(base / "DejaVuSans-Oblique.ttf")))


def build_pdf(detail, notes="", cams=None, include_chart=True):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    _register_fonts()
    S = lambda n, **k: ParagraphStyle(n, fontName=k.pop("fontName", "DJ"), **k)
    title, h1, body, small = S("t", fontName="DJB", fontSize=16, leading=20, alignment=1, textColor=colors.HexColor("#1F3A5F")), \
        S("h", fontName="DJB", fontSize=12, leading=16, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#1F3A5F")), \
        S("b", fontSize=9.5, leading=13), S("s", fontName="DJI", fontSize=8, leading=11, textColor=colors.HexColor("#595959"))
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm)
    t = CLASSES[detail["y_true"]]
    el = [Paragraph("BÁO CÁO DỰ ĐOÁN — TriFuse-AD Demo", title), Spacer(1, 6),
          Paragraph(f"Mã ca: <b>{detail['subject_id']}</b> &nbsp;|&nbsp; Ngày xuất: {datetime.now():%d/%m/%Y %H:%M}", body),
          Paragraph(f"Nhãn tham chiếu (theo CDR): <b>{t}</b> — {CLASS_FULL[t]}", body), Paragraph("1. Kết quả các mô hình", h1)]
    cell = S("c", fontSize=8, leading=10)
    data = [[Paragraph(f"<b>{x}</b>", cell) for x in ["Mô hình", "Dự đoán", "Đúng/Sai", "CN %", "VMD %", "AD %", "Theo seed"]]] + \
           [[Paragraph(str(x), cell) for x in r] for r in _rows(detail)]
    tb = Table(data, colWidths=[3.6 * cm, 1.6 * cm, 1.6 * cm, 1.3 * cm, 1.4 * cm, 1.3 * cm, 6.2 * cm], repeatRows=1)
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F2")), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#BFBFBF")),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    el += [tb, Spacer(1, 6), Paragraph(_summary_line(detail), body)]
    if include_chart:
        el += [Spacer(1, 6), Image(io.BytesIO(viz.fig_bytes(viz.case_prob_figure(detail, detail["y_true"]))), width=12.5 * cm, height=6.4 * cm)]
    if cams:
        el += [Paragraph("2. Giải thích mô hình (Grad-CAM)", h1)]
        for title, png in cams:
            el += [Paragraph(f"<b>{title}</b>", body), Image(io.BytesIO(png), width=17 * cm, height=6.5 * cm, kind="proportional")]
        el += [Paragraph("Vùng nóng (đỏ/vàng) là vùng ảnh hưởng nhiều nhất tới điểm của lớp được chọn; chỉ mang tính gợi ý.", small)]
    if notes.strip():
        el += [Paragraph("Ghi chú", h1), Paragraph(notes.strip().replace("\n", "<br/>"), body)]
    el += [Spacer(1, 10), Paragraph("Lưu ý: " + DISCLAIMER, small)]
    doc.build(el)
    return buf.getvalue()


def build_custom_prediction_docx(patient_id, clinical_info, model_name, pred_class, probs,
                                  risk_info, gate_val=None, notes="", cams=None,
                                  overview_chart_bytes=None, triplanar_bytes=None):
    """Xuất báo cáo kết quả dự đoán ca mới tải lên dưới định dạng Word (.docx)."""
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2)
        s.top_margin = s.bottom_margin = Cm(1.8)
    st = doc.styles["Normal"]
    st.font.name = "Arial"
    st.font.size = Pt(10.5)

    h = doc.add_heading("PHIẾU KẾT QUẢ ĐÁNH GIÁ NGUY CƠ ALZHEIMER", 0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f"Mã ca / Bệnh nhân: {patient_id}    |    Ngày đánh giá: {datetime.now():%d/%m/%Y %H:%M}")
    doc.add_paragraph(f"Mô hình AI sử dụng: {model_name}")

    # Phần 1: Thông tin lâm sàng
    doc.add_heading("1. Thông tin lâm sàng đầu vào", 1)
    t_clin = doc.add_table(rows=1, cols=4)
    t_clin.style = "Light Grid Accent 1"
    for c, txt in zip(t_clin.rows[0].cells, ["Chỉ số", "Giá trị", "Chỉ số", "Giá trị"]):
        c.text = txt
    
    clin_items = [
        ("Tuổi", f"{clinical_info.get('age', 'N/A')} tuổi"),
        ("Giới tính", "Nam" if clinical_info.get('sex') == 1 else "Nữ"),
        ("Trình độ học vấn", f"Bậc {clinical_info.get('education', 'N/A')} / 5"),
        ("Chỉ số SES", f"{clinical_info.get('ses', 'N/A')} / 5"),
        ("Thể tích nội sọ (eTIV)", f"{clinical_info.get('etiv', 'N/A')} mm³"),
        ("Thể tích não chuẩn hóa (nWBV)", f"{clinical_info.get('nwbv', 'N/A')}"),
        ("Hệ số tỷ lệ Atlas (ASF)", f"{clinical_info.get('asf', 'N/A')}"),
        ("Trọng số Gate (Lâm sàng)", f"{gate_val:.3f}" if gate_val is not None else "N/A"),
    ]
    for i in range(0, len(clin_items), 2):
        row = t_clin.add_row().cells
        row[0].text = clin_items[i][0]
        row[1].text = str(clin_items[i][1])
        if i + 1 < len(clin_items):
            row[2].text = clin_items[i+1][0]
            row[3].text = str(clin_items[i+1][1])

    # Phần 2: Kết quả dự đoán & Nguy cơ
    doc.add_heading("2. Kết quả phân loại & Chỉ số nguy cơ", 1)
    p_pred = doc.add_paragraph()
    r1 = p_pred.add_run(f"Dự đoán giai đoạn: {pred_class} — {CLASS_FULL.get(pred_class, pred_class)}\n")
    r1.bold = True
    r1.font.size = Pt(12)
    
    r_risk = p_pred.add_run(f"Chỉ số nguy cơ tổng thể: {risk_info['score']:.1f}% ({risk_info['level']})\n")
    r_risk.bold = True
    
    p_pred.add_run(f"Phân bố xác suất: CN (Bình thường): {probs[0]*100:.1f}% | VMD (Suy giảm nhẹ): {probs[1]*100:.1f}% | AD (Alzheimer): {probs[2]*100:.1f}%\n")
    p_pred.add_run(f"Khuyến nghị chuyên môn: {risk_info['recommendation']}")

    if overview_chart_bytes:
        doc.add_picture(io.BytesIO(overview_chart_bytes), width=Cm(15.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Phần 3: Hình ảnh MRI và Giải thích Grad-CAM
    if triplanar_bytes:
        doc.add_heading("3. Các lát cắt MRI não trung tâm (Tri-planar)", 1)
        doc.add_picture(io.BytesIO(triplanar_bytes), width=Cm(15.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    if cams:
        doc.add_heading("4. Bản đồ kích hoạt vùng não tổn thương (Grad-CAM)", 1)
        for title, png in cams:
            doc.add_paragraph(title).runs[0].bold = True
            doc.add_picture(io.BytesIO(png), width=Cm(16.5))
        doc.add_paragraph("Vùng tập trung nhiệt (vàng/đỏ) phản ánh các cấu trúc giải phẫu não (như hồi hải mã, não thất, vỏ não thái dương) đóng vai trò quyết định trong phân loại của mô hình AI.")

    if notes.strip():
        doc.add_heading("5. Ghi chú của bác sĩ / người đánh giá", 1)
        doc.add_paragraph(notes.strip())

    p = doc.add_paragraph()
    r = p.add_run("Lưu ý: " + DISCLAIMER)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def build_custom_prediction_pdf(patient_id, clinical_info, model_name, pred_class, probs,
                                 risk_info, gate_val=None, notes="", cams=None,
                                 overview_chart_bytes=None, triplanar_bytes=None):
    """Xuất báo cáo kết quả dự đoán ca mới tải lên dưới định dạng PDF."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    _register_fonts()
    S = lambda n, **k: ParagraphStyle(n, fontName=k.pop("fontName", "DJ"), **k)
    title = S("t", fontName="DJB", fontSize=15, leading=19, alignment=1, textColor=colors.HexColor("#1F3A5F"))
    h1 = S("h", fontName="DJB", fontSize=11, leading=15, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1F3A5F"))
    body = S("b", fontSize=9, leading=12)
    small = S("s", fontName="DJI", fontSize=8, leading=11, textColor=colors.HexColor("#595959"))
    cell = S("c", fontSize=8, leading=10)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)

    el = [
        Paragraph("PHIẾU KẾT QUẢ ĐÁNH GIÁ NGUY CƠ ALZHEIMER", title),
        Spacer(1, 4),
        Paragraph(f"Mã ca / Bệnh nhân: <b>{patient_id}</b> &nbsp;|&nbsp; Ngày đánh giá: {datetime.now():%d/%m/%Y %H:%M}", body),
        Paragraph(f"Mô hình AI: <b>{model_name}</b>", body),
        Paragraph("1. Thông tin lâm sàng đầu vào", h1),
    ]

    sex_str = "Nam" if clinical_info.get("sex") == 1 else "Nữ"
    gate_str = f"{gate_val:.3f}" if gate_val is not None else "N/A"
    c_data = [
        [Paragraph("<b>Chỉ số</b>", cell), Paragraph("<b>Giá trị</b>", cell), Paragraph("<b>Chỉ số</b>", cell), Paragraph("<b>Giá trị</b>", cell)],
        [Paragraph("Tuổi", cell), Paragraph(f"{clinical_info.get('age', 'N/A')} tuổi", cell), Paragraph("Giới tính", cell), Paragraph(sex_str, cell)],
        [Paragraph("Trình độ học vấn", cell), Paragraph(f"Bậc {clinical_info.get('education', 'N/A')} / 5", cell), Paragraph("Chỉ số SES", cell), Paragraph(f"{clinical_info.get('ses', 'N/A')}", cell)],
        [Paragraph("Thể tích eTIV", cell), Paragraph(f"{clinical_info.get('etiv', 'N/A')} mm³", cell), Paragraph("Thể tích não nWBV", cell), Paragraph(f"{clinical_info.get('nwbv', 'N/A')}", cell)],
        [Paragraph("Hệ số Atlas (ASF)", cell), Paragraph(f"{clinical_info.get('asf', 'N/A')}", cell), Paragraph("Trọng số Gate", cell), Paragraph(gate_str, cell)],
    ]
    tb = Table(c_data, colWidths=[4.2 * cm, 4.2 * cm, 4.2 * cm, 4.2 * cm])
    tb.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F2")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFBFBF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    el += [tb, Spacer(1, 6)]

    el += [
        Paragraph("2. Kết quả phân loại & Chỉ số nguy cơ", h1),
        Paragraph(f"Dự đoán giai đoạn: <b>{pred_class} — {CLASS_FULL.get(pred_class, pred_class)}</b>", body),
        Paragraph(f"Chỉ số nguy cơ tổng thể: <b>{risk_info['score']:.1f}%</b> ({risk_info['level']})", body),
        Paragraph(f"Phân bố xác suất: CN: {probs[0]*100:.1f}% &nbsp;|&nbsp; VMD: {probs[1]*100:.1f}% &nbsp;|&nbsp; AD: {probs[2]*100:.1f}%", body),
        Paragraph(f"Khuyến nghị chuyên môn: {risk_info['recommendation']}", body),
    ]

    if overview_chart_bytes:
        el += [Spacer(1, 4), Image(io.BytesIO(overview_chart_bytes), width=16 * cm, height=5.2 * cm, kind="proportional")]

    if triplanar_bytes:
        el += [Paragraph("3. Hình ảnh MRI não trung tâm (Tri-planar)", h1),
               Image(io.BytesIO(triplanar_bytes), width=16 * cm, height=5.5 * cm, kind="proportional")]

    if cams:
        el += [Paragraph("4. Bản đồ giải thích vùng não kích hoạt (Grad-CAM)", h1)]
        for title_str, png in cams:
            el += [Paragraph(f"<b>{title_str}</b>", body), Image(io.BytesIO(png), width=17 * cm, height=6.0 * cm, kind="proportional")]
        el += [Paragraph("Vùng màu nóng (đỏ/vàng) là vùng vỏ não và nhu mô não có trọng số kích hoạt cao nhất; chỉ mang tính gợi ý học thuật.", small)]

    if notes.strip():
        el += [Paragraph("5. Ghi chú", h1), Paragraph(notes.strip().replace("\n", "<br/>"), body)]

    el += [Spacer(1, 8), Paragraph("Lưu ý: " + DISCLAIMER, small)]
    doc.build(el)
    return buf.getvalue()
