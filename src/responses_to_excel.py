import json
import re
from pathlib import Path
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.chart import BarChart, Reference

# ==============================================================================
# CONFIGURACIÓ DE FITXERS D'ENTRADA I SORTIDA
# ==============================================================================
FITXER_MARKDOWN = "data/questionnaire_bike.md"
FITXERS_MODELS = [
    ("outputs/tmp/augmented_personas_bike_poligpt.jsonl", "PoliGPT"),
    ("outputs/tmp/augmented_personas_bike_qwen3.jsonl", "Qwen3"),
]
FITXER_EXCEL_SORTIDA = "respostes_questionari_bicicleta_valencia.xlsx"


def parsejar_questionari(ruta_md: str):
    """Llegeix el fitxer Markdown i extrau seccions, preguntes i tipus d'escala."""
    md_text = Path(ruta_md).read_text(encoding="utf-8")

    sections = []
    current_sec = None
    questions = []

    for line in md_text.splitlines():
        m_sec = re.match(r"^## (\d+\.\s+.+)$", line.strip())
        if m_sec:
            current_sec = m_sec.group(1).strip()
            sections.append(current_sec)

        m_q = re.match(r"^### Question (\d+):\s*(.+)$", line.strip())
        if m_q:
            q_num = int(m_q.group(1))
            q_text = m_q.group(2).strip()
            questions.append({
                "q_num": q_num,
                "q_code": f"Q{q_num}",
                "section": current_sec,
                "question": q_text,
            })

    # Extreure metadades de tipus i escala de cada bloc de pregunta
    q_blocks = re.split(r"(?=### Question \d+:)", md_text)
    q_meta = {}
    for blk in q_blocks:
        m_q = re.match(r"^### Question (\d+):\s*(.+)", blk.strip())
        if m_q:
            qn = int(m_q.group(1))
            m_type = re.search(r"\*\*Type:\*\*\s*(.+)", blk)
            m_scale = re.search(r"\*\*Scale:\*\*\s*(.+)", blk)
            q_meta[qn] = {
                "type": m_type.group(1).strip() if m_type else "",
                "scale": m_scale.group(1).strip() if m_scale else "",
            }

    for q in questions:
        q["type"] = q_meta.get(q["q_num"], {}).get("type", "")
        q["scale"] = q_meta.get(q["q_num"], {}).get("scale", "")

    return sections, questions


def carregar_respostes(fitxers_models: list, questions: list):
    """Llegeix els fitxers JSONL i aplana l'estructura de facet_responses."""
    records = []
    personas_info = {}

    for fpath, model_name in fitxers_models:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                p_id = obj.get("id")
                orig = obj.get("original_persona", {})
                facets = obj.get("facet_responses", {})

                if p_id not in personas_info:
                    personas_info[p_id] = orig

                # Aplanar totes les seccions de facet_responses en un únic diccionari Q1..Q45
                flat_q = {}
                for _, q_dict in facets.items():
                    if isinstance(q_dict, dict):
                        for q_key, val in q_dict.items():
                            flat_q[q_key] = val

                row = {
                    "Persona_ID": p_id,
                    "Name": orig.get("name", ""),
                    "Model": model_name,
                    "Age": orig.get("age", ""),
                    "Gender": orig.get("gender", ""),
                    "Occupation": orig.get("occupation", ""),
                    "Education": orig.get("education", ""),
                    "Location": orig.get("location", ""),
                }

                for q in questions:
                    qc = q["q_code"]
                    val = flat_q.get(qc, "")
                    # Convertir valors d'escala numèrica (1-5) a enter per permetre càlculs
                    if isinstance(val, str) and val.strip().isdigit() and "Opinion Scale" in q["type"]:
                        val = int(val.strip())
                    row[qc] = val

                records.append(row)

    df = pd.DataFrame(records).sort_values(["Persona_ID", "Model"]).reset_index(drop=True)
    return df, personas_info


def generar_excel(sections: list, questions: list, df: pd.DataFrame, personas_info: dict, ruta_sortida: str):
    """Crea i formata el llibre d'Excel amb 4 pestanyes i gràfic comparatiu."""
    wb = openpyxl.Workbook()

    # Estils visuals
    HDR_FILL = PatternFill("solid", fgColor="1E3A4C")
    SUBHDR_FILL = PatternFill("solid", fgColor="2C556E")
    SEC_FILL = PatternFill("solid", fgColor="D9E6EE")
    ZEBRA_FILL = PatternFill("solid", fgColor="F5F8FA")
    WHITE_FILL = PatternFill("solid", fgColor="FFFFFF")

    HDR_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    SUBHDR_FONT = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    BOLD_FONT = Font(name="Calibri", size=10, bold=True, color="1E293B")
    REG_FONT = Font(name="Calibri", size=10, bold=False, color="1E293B")
    TITLE_FONT = Font(name="Calibri", size=15, bold=True, color="1E3A4C")
    SUBTITLE_FONT = Font(name="Calibri", size=10, italic=True, color="475569")

    THIN_SIDE = Side(border_style="thin", color="CBD5E1")
    MED_NAVY = Side(border_style="medium", color="1E3A4C")
    CELL_BORDER = Border(left=THIN_SIDE, right=THIN_SIDE, top=THIN_SIDE, bottom=THIN_SIDE)
    HDR_BORDER = Border(left=THIN_SIDE, right=THIN_SIDE, top=MED_NAVY, bottom=MED_NAVY)

    # --------------------------------------------------------------------------
    # PESTANYA 1: Respostes_Per_Persona_Model
    # --------------------------------------------------------------------------
    ws1 = wb.active
    ws1.title = "Respostes_Per_Persona_Model"

    ws1["A1"] = "Qüestionari SCOPE – Mobilitat Ciclista a València: Respostes per Persona i Model"
    ws1["A1"].font = TITLE_FONT
    ws1["A2"] = "Comparativa completa de les 45 preguntes (8 seccions) entre models per a cada persona avaluada."
    ws1["A2"].font = SUBTITLE_FONT

    meta_cols = ["Persona_ID", "Name", "Model", "Age", "Gender", "Occupation", "Location"]
    meta_labels = ["ID Persona", "Nom", "Model IA", "Edat", "Gènere", "Ocupació", "Ubicació"]

    ws1.merge_cells(start_row=4, start_column=1, end_row=4, end_column=len(meta_cols))
    c_meta = ws1.cell(row=4, column=1, value="DADES DE LA PERSONA I MODEL")
    c_meta.font = HDR_FONT
    c_meta.alignment = Alignment(horizontal="center", vertical="center")

    for col_idx in range(1, len(meta_cols) + 1):
        ws1.cell(row=4, column=col_idx).fill = HDR_FILL
        ws1.cell(row=4, column=col_idx).border = HDR_BORDER
        ch = ws1.cell(row=5, column=col_idx, value=meta_labels[col_idx - 1])
        ch.font = SUBHDR_FONT
        ch.fill = SUBHDR_FILL
        ch.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ch.border = HDR_BORDER

    col_cursor = len(meta_cols) + 1
    sec_colors = ["23485E", "2E5B75"]

    for s_idx, sec_name in enumerate(sections):
        sec_qs = [q for q in questions if q["section"] == sec_name]
        start_c = col_cursor
        end_c = col_cursor + len(sec_qs) - 1
        ws1.merge_cells(start_row=4, start_column=start_c, end_row=4, end_column=end_c)
        s_fill = PatternFill("solid", fgColor=sec_colors[s_idx % len(sec_colors)])
        sc = ws1.cell(row=4, column=start_c, value=sec_name)
        sc.font = HDR_FONT
        sc.alignment = Alignment(horizontal="center", vertical="center")

        for c_i in range(start_c, end_c + 1):
            ws1.cell(row=4, column=c_i).fill = s_fill
            ws1.cell(row=4, column=c_i).border = HDR_BORDER

        for q in sec_qs:
            qh = ws1.cell(row=5, column=col_cursor, value=f"{q['q_code']}: {q['question']}")
            qh.font = SUBHDR_FONT
            qh.fill = SUBHDR_FILL
            qh.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            qh.border = HDR_BORDER
            col_cursor += 1

    ws1.row_dimensions[4].height = 24
    ws1.row_dimensions[5].height = 48

    num_rows = len(df)
    for r_idx, row_data in df.iterrows():
        excel_row = 6 + r_idx
        row_fill = ZEBRA_FILL if (r_idx % 2 == 1) else WHITE_FILL

        for c_i, col_key in enumerate(meta_cols, start=1):
            val = row_data[col_key]
            cell = ws1.cell(row=excel_row, column=c_i, value=val)
            cell.font = BOLD_FONT if col_key in ["Persona_ID", "Name", "Model"] else REG_FONT
            cell.fill = row_fill
            cell.border = CELL_BORDER
            cell.alignment = Alignment(
                horizontal="center" if col_key in ["Persona_ID", "Model", "Age"] else "left",
                vertical="top",
                wrap_text=True,
            )

        for q_i, q in enumerate(questions):
            c_i = len(meta_cols) + 1 + q_i
            val = row_data[q["q_code"]]
            cell = ws1.cell(row=excel_row, column=c_i, value=val)
            cell.font = REG_FONT
            cell.fill = row_fill
            cell.border = CELL_BORDER
            if isinstance(val, int):
                cell.alignment = Alignment(horizontal="center", vertical="top")
                cell.number_format = "0"
            else:
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

        ws1.row_dimensions[excel_row].height = 78

    meta_widths = [13, 16, 11, 7, 13, 24, 16]
    for idx, w in enumerate(meta_widths, start=1):
        ws1.column_dimensions[get_column_letter(idx)].width = w

    for q_i, q in enumerate(questions):
        c_letter = get_column_letter(len(meta_cols) + 1 + q_i)
        if "Opinion Scale" in q["type"]:
            ws1.column_dimensions[c_letter].width = 16
        elif q["q_code"] in ["Q37", "Q38", "Q40", "Q43"]:
            ws1.column_dimensions[c_letter].width = 48
        else:
            ws1.column_dimensions[c_letter].width = 28

    ws1.freeze_panes = "D6"
    ws1.auto_filter.ref = f"A5:{get_column_letter(len(meta_cols) + len(questions))}{5 + num_rows}"

    # --------------------------------------------------------------------------
    # PESTANYA 2: Comparativa_Vertical
    # --------------------------------------------------------------------------
    ws2 = wb.create_sheet(title="Comparativa_Vertical")
    ws2["A1"] = "Vista Vertical per Pregunta i Secció (Comparativa Directa entre Models)"
    ws2["A1"].font = TITLE_FONT
    ws2["A2"] = "Cada fila correspon a una pregunta del qüestionari; les columnes mostren les respostes per persona i model."
    ws2["A2"].font = SUBTITLE_FONT

    fixed_headers = ["Secció", "Codi", "Pregunta", "Tipus de Pregunta"]
    persona_model_pairs = [(r["Persona_ID"], r["Name"], r["Model"]) for _, r in df.iterrows()]

    for c_i, h in enumerate(fixed_headers, start=1):
        cell = ws2.cell(row=4, column=c_i, value=h)
        cell.font = HDR_FONT
        cell.fill = HDR_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = HDR_BORDER

    for p_i, (pid, pname, mname) in enumerate(persona_model_pairs):
        c_i = len(fixed_headers) + 1 + p_i
        cell = ws2.cell(row=4, column=c_i, value=f"{pname}\n({pid} – {mname})")
        cell.font = HDR_FONT
        cell.fill = SUBHDR_FILL if p_i % 2 == 1 else HDR_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = HDR_BORDER

    ws2.row_dimensions[4].height = 36

    for q_i, q in enumerate(questions):
        r_num = 5 + q_i
        r_fill = ZEBRA_FILL if (q_i % 2 == 1) else WHITE_FILL

        c1 = ws2.cell(row=r_num, column=1, value=q["section"])
        c2 = ws2.cell(row=r_num, column=2, value=q["q_code"])
        c3 = ws2.cell(row=r_num, column=3, value=q["question"])
        c4 = ws2.cell(row=r_num, column=4, value=q["type"])

        for c, align_h, is_b in [(c1, "left", False), (c2, "center", True), (c3, "left", True), (c4, "left", False)]:
            c.font = BOLD_FONT if is_b else REG_FONT
            c.fill = r_fill
            c.border = CELL_BORDER
            c.alignment = Alignment(horizontal=align_h, vertical="top", wrap_text=True)

        is_long_text = q["q_code"] in ["Q37", "Q38", "Q40", "Q43"]

        for p_i, (pid, pname, mname) in enumerate(persona_model_pairs):
            c_i = len(fixed_headers) + 1 + p_i
            val = df.loc[(df["Persona_ID"] == pid) & (df["Model"] == mname), q["q_code"]].values[0]
            cell = ws2.cell(row=r_num, column=c_i, value=val)
            cell.font = REG_FONT
            cell.fill = r_fill
            cell.border = CELL_BORDER
            if isinstance(val, int):
                cell.alignment = Alignment(horizontal="center", vertical="top")
                cell.number_format = "0"
            else:
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

        ws2.row_dimensions[r_num].height = 95 if is_long_text else 38

    ws2.column_dimensions["A"].width = 28
    ws2.column_dimensions["B"].width = 8
    ws2.column_dimensions["C"].width = 42
    ws2.column_dimensions["D"].width = 24
    for p_i in range(len(persona_model_pairs)):
        ws2.column_dimensions[get_column_letter(len(fixed_headers) + 1 + p_i)].width = 32

    ws2.freeze_panes = "E5"
    ws2.auto_filter.ref = f"A4:{get_column_letter(len(fixed_headers) + len(persona_model_pairs))}{4 + len(questions)}"

    # --------------------------------------------------------------------------
    # PESTANYA 3: Resum_Escales_i_Comparativa
    # --------------------------------------------------------------------------
    ws3 = wb.create_sheet(title="Resum_Escales_i_Comparativa")
    ws3["A1"] = "Resum Quantitatiu de les Escales d'Opinió (1–5) per Model i Pregunta"
    ws3["A1"].font = TITLE_FONT
    ws3["A2"] = "Mitjanes calculades dinàmicament mitjançant fórmules AVERAGEIFS sobre la pestanya 'Respostes_Per_Persona_Model'."
    ws3["A2"].font = SUBTITLE_FONT

    scale_qs = [q for q in questions if "Opinion Scale" in q["type"]]
    models_list = df["Model"].unique().tolist()
    m1, m2 = models_list[0], models_list[1]

    headers_s3 = ["Secció", "Codi", "Pregunta (Escala 1–5)", f"Mitjana {m1}", f"Mitjana {m2}", f"Diferència ({m2} - {m1})", "Mitjana Global"]
    for c_i, h in enumerate(headers_s3, start=1):
        cell = ws3.cell(row=4, column=c_i, value=h)
        cell.font = HDR_FONT
        cell.fill = HDR_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = HDR_BORDER
    ws3.row_dimensions[4].height = 30

    last_data_row_s1 = 5 + num_rows
    for idx, q in enumerate(scale_qs):
        r_num = 5 + idx
        r_fill = ZEBRA_FILL if (idx % 2 == 1) else WHITE_FILL

        q_idx_in_all = [i for i, item in enumerate(questions) if item["q_code"] == q["q_code"]][0]
        s1_col_letter = get_column_letter(len(meta_cols) + 1 + q_idx_in_all)

        c1 = ws3.cell(row=r_num, column=1, value=q["section"])
        c2 = ws3.cell(row=r_num, column=2, value=q["q_code"])
        c3 = ws3.cell(row=r_num, column=3, value=q["question"])
        c4 = ws3.cell(row=r_num, column=4, value=f'=AVERAGEIFS(Respostes_Per_Persona_Model!${s1_col_letter}$6:${s1_col_letter}${last_data_row_s1}, Respostes_Per_Persona_Model!$C$6:$C${last_data_row_s1}, "{m1}")')
        c5 = ws3.cell(row=r_num, column=5, value=f'=AVERAGEIFS(Respostes_Per_Persona_Model!${s1_col_letter}$6:${s1_col_letter}${last_data_row_s1}, Respostes_Per_Persona_Model!$C$6:$C${last_data_row_s1}, "{m2}")')
        c6 = ws3.cell(row=r_num, column=6, value=f"=E{r_num}-D{r_num}")
        c7 = ws3.cell(row=r_num, column=7, value=f"=AVERAGE(D{r_num}:E{r_num})")

        for c, align_h, is_b in [(c1, "left", False), (c2, "center", True), (c3, "left", False)]:
            c.font = BOLD_FONT if is_b else REG_FONT
            c.fill = r_fill
            c.border = CELL_BORDER
            c.alignment = Alignment(horizontal=align_h, vertical="center", wrap_text=True)

        for c, num_fmt in [(c4, "0.00"), (c5, "0.00"), (c6, "+0.00;-0.00;0.00"), (c7, "0.00")]:
            c.font = BOLD_FONT if c == c7 else REG_FONT
            c.fill = r_fill
            c.border = CELL_BORDER
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.number_format = num_fmt

        ws3.row_dimensions[r_num].height = 24

    last_scale_row = 4 + len(scale_qs)
    total_row = last_scale_row + 1

    ws3.cell(row=total_row, column=1, value="MITJANA GENERAL (TOTES LES ESCALES)").font = BOLD_FONT
    ws3.merge_cells(start_row=total_row, start_column=1, end_row=total_row, end_column=3)
    for col_i in range(1, 8):
        cell = ws3.cell(row=total_row, column=col_i)
        cell.fill = SEC_FILL
        cell.border = HDR_BORDER
        if col_i == 1:
            cell.alignment = Alignment(horizontal="right", vertical="center")
        elif col_i in [4, 5, 7]:
            col_let = get_column_letter(col_i)
            cell.value = f"=AVERAGE({col_let}5:{col_let}{last_scale_row})"
            cell.font = BOLD_FONT
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.number_format = "0.00"
        elif col_i == 6:
            cell.value = f"=E{total_row}-D{total_row}"
            cell.font = BOLD_FONT
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.number_format = "+0.00;-0.00;0.00"
    ws3.row_dimensions[total_row].height = 26

    ws3.conditional_formatting.add(
        f"F5:F{last_scale_row}",
        ColorScaleRule(
            start_type="num", start_value=-1.5, start_color="F87171",
            mid_type="num", mid_value=0, mid_color="FFFFFF",
            end_type="num", end_value=1.5, end_color="60A5FA",
        ),
    )

    chart = BarChart()
    chart.type = "col"
    chart.style = 10
    chart.grouping = "clustered"
    chart.overlap = -15
    chart.title = f"Comparativa de Mitjanes per Pregunta d'Escala ({m1} vs {m2})"
    chart.y_axis.title = "Puntuació Mitjana (1–5)"
    chart.x_axis.title = "Codi de Pregunta"
    chart.y_axis.scaling.min = 0
    chart.y_axis.scaling.max = 5.5

    data_ref = Reference(ws3, min_col=4, min_row=4, max_col=5, max_row=last_scale_row)
    cats_ref = Reference(ws3, min_col=2, min_row=5, max_row=last_scale_row)
    chart.add_data(data_ref, titles_from_data=True)
    chart.set_categories(cats_ref)
    chart.width = 26
    chart.height = 13
    ws3.add_chart(chart, "I4")

    ws3.column_dimensions["A"].width = 30
    ws3.column_dimensions["B"].width = 8
    ws3.column_dimensions["C"].width = 48
    ws3.column_dimensions["D"].width = 16
    ws3.column_dimensions["E"].width = 16
    ws3.column_dimensions["F"].width = 22
    ws3.column_dimensions["G"].width = 16
    ws3.freeze_panes = "D5"

    # --------------------------------------------------------------------------
    # PESTANYA 4: Perfils_Persones
    # --------------------------------------------------------------------------
    ws4 = wb.create_sheet(title="Perfils_Persones")
    ws4["A1"] = "Perfils de les Persones Avaluades (Original Personas)"
    ws4["A1"].font = TITLE_FONT
    ws4["A2"] = "Informació biogràfica i sociodemogràfica proporcionada als models com a context."
    ws4["A2"].font = SUBTITLE_FONT

    p_headers = ["ID Persona", "Nom", "Edat", "Gènere", "Ocupació", "Educació", "Ubicació", "Context / Background"]
    for c_i, h in enumerate(p_headers, start=1):
        cell = ws4.cell(row=4, column=c_i, value=h)
        cell.font = HDR_FONT
        cell.fill = HDR_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = HDR_BORDER
    ws4.row_dimensions[4].height = 28

    for idx, pid in enumerate(sorted(personas_info.keys())):
        r_num = 5 + idx
        p = personas_info[pid]
        r_fill = ZEBRA_FILL if (idx % 2 == 1) else WHITE_FILL
        vals = [
            pid,
            p.get("name", ""),
            p.get("age", ""),
            p.get("gender", ""),
            p.get("occupation", ""),
            p.get("education", ""),
            p.get("location", ""),
            p.get("background", ""),
        ]
        for c_i, v in enumerate(vals, start=1):
            cell = ws4.cell(row=r_num, column=c_i, value=v)
            cell.font = BOLD_FONT if c_i in [1, 2] else REG_FONT
            cell.fill = r_fill
            cell.border = CELL_BORDER
            cell.alignment = Alignment(
                horizontal="center" if c_i in [1, 3] else "left",
                vertical="top",
                wrap_text=True,
            )
        ws4.row_dimensions[r_num].height = 75

    p_widths = [14, 18, 8, 14, 24, 24, 18, 70]
    for idx, w in enumerate(p_widths, start=1):
        ws4.column_dimensions[get_column_letter(idx)].width = w

    wb.save(ruta_sortida)
    print(f"Fitxer Excel generat correctament: {ruta_sortida}")


if __name__ == "__main__":
    seccions, preguntes = parsejar_questionari(FITXER_MARKDOWN)
    df_respostes, perfils = carregar_respostes(FITXERS_MODELS, preguntes)
    generar_excel(seccions, preguntes, df_respostes, perfils, FITXER_EXCEL_SORTIDA)