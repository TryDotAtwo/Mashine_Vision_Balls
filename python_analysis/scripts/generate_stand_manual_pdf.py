from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PDF = ROOT / "output" / "pdf" / "ivan_stand_manual_ru.pdf"
PRIMARY = colors.HexColor("#17324d")
ACCENT = colors.HexColor("#4f7cac")
LIGHT = colors.HexColor("#eef3f8")
TEXT = colors.HexColor("#1d2125")
MUTED = colors.HexColor("#6b7682")
WARN = colors.HexColor("#fff2cc")


def register_fonts() -> None:
    fonts = {
        "Body": r"C:\Windows\Fonts\times.ttf",
        "Body-Bold": r"C:\Windows\Fonts\timesbd.ttf",
        "Body-Italic": r"C:\Windows\Fonts\timesi.ttf",
        "Sans": r"C:\Windows\Fonts\arial.ttf",
        "Sans-Bold": r"C:\Windows\Fonts\arialbd.ttf",
    }
    for name, path in fonts.items():
        pdfmetrics.registerFont(TTFont(name, path))


def styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(name="BodyRu", fontName="Body", fontSize=11.2, leading=15.2, alignment=TA_JUSTIFY, textColor=TEXT, spaceAfter=3))
    ss.add(ParagraphStyle(name="BodyCenterRu", parent=ss["BodyRu"], alignment=TA_CENTER))
    ss.add(ParagraphStyle(name="H1Ru", fontName="Sans-Bold", fontSize=18, leading=22, textColor=PRIMARY, spaceBefore=8, spaceAfter=8))
    ss.add(ParagraphStyle(name="H2Ru", fontName="Sans-Bold", fontSize=14.2, leading=18.2, textColor=PRIMARY, spaceBefore=7, spaceAfter=7))
    ss.add(ParagraphStyle(name="H3Ru", fontName="Sans-Bold", fontSize=12.0, leading=15, textColor=ACCENT, spaceBefore=5, spaceAfter=5))
    ss.add(ParagraphStyle(name="TitleRu", fontName="Sans-Bold", fontSize=24, leading=28, alignment=TA_CENTER, textColor=PRIMARY))
    ss.add(ParagraphStyle(name="SubtitleRu", fontName="Sans", fontSize=12.5, leading=17, alignment=TA_CENTER, textColor=MUTED))
    ss.add(ParagraphStyle(name="SmallRu", fontName="Body-Italic", fontSize=8.8, leading=11, alignment=TA_CENTER, textColor=MUTED))
    return ss


def footer(canvas, doc):
    if canvas.getPageNumber() == 1:
        return
    canvas.saveState()
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 14 * mm, A4[0] - doc.rightMargin, 14 * mm)
    canvas.setFont("Sans", 8.3)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 9.5 * mm, "Иван Литвак. Руководство по стенду и CV-эксперименту.")
    canvas.drawRightString(A4[0] - doc.rightMargin, 9.5 * mm, f"Стр. {canvas.getPageNumber()}")
    canvas.restoreState()


def p(text: str, style):
    return Paragraph(text, style)


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md_inline_to_reportlab(text: str) -> str:
    """Convert markdown **bold** to ReportLab <b>...</b>; escape other XML-sensitive chars."""
    if not text:
        return text
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("**", i):
            j = text.find("**", i + 2)
            if j == -1:
                out.append(_xml_escape(text[i:]))
                break
            inner = text[i + 2 : j]
            out.append("<b>" + _xml_escape(inner) + "</b>")
            i = j + 2
        else:
            nxt = text.find("**", i)
            if nxt == -1:
                out.append(_xml_escape(text[i:]))
                break
            out.append(_xml_escape(text[i:nxt]))
            i = nxt
    return "".join(out)


def mk_table(data, col_widths):
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Sans-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                ("FONTNAME", (0, 1), (-1, -1), "Body"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.4),
                ("LEADING", (0, 0), (-1, -1), 12.6),
                ("TEXTCOLOR", (0, 0), (-1, -1), TEXT),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8d3df")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def fig(path: Path, width_mm: float, caption: str, ss):
    img = Image(str(path))
    target_width = width_mm * mm
    target_height = img.imageHeight * target_width / img.imageWidth
    max_height = 120 * mm
    if target_height > max_height:
        scale = max_height / target_height
        target_width *= scale
        target_height *= scale
    img.drawWidth = target_width
    img.drawHeight = target_height
    return [img, Spacer(1, 2 * mm), p(caption, ss["SmallRu"]), Spacer(1, 4 * mm)]


def callout(text: str, ss, title: str = "Важно"):
    box = Table([[p(f"<b>{title}</b>", ss["BodyRu"])], [p(text, ss["BodyRu"])]], colWidths=[170 * mm], hAlign="CENTER")
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), WARN),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#fffaf0")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d9b44a")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return [box, Spacer(1, 4 * mm)]


def append_markdown(story, path: Path, ss, pagebreak_h2: bool = False, h2_pagebreak_every: int = 1):
    lines = path.read_text(encoding="utf-8").splitlines()
    paragraph_buffer: list[str] = []
    bullet_buffer: list[str] = []
    code_buffer: list[str] = []
    table_buffer: list[str] = []
    in_code = False
    h2_counter = 0

    def flush_paragraph():
        nonlocal paragraph_buffer
        if paragraph_buffer:
            joined = " ".join(part.strip() for part in paragraph_buffer)
            story.append(p(md_inline_to_reportlab(joined), ss["BodyRu"]))
            paragraph_buffer = []

    def flush_bullets():
        nonlocal bullet_buffer
        if bullet_buffer:
            for item in bullet_buffer:
                story.append(p("• " + md_inline_to_reportlab(item), ss["BodyRu"]))
            bullet_buffer = []

    def flush_code():
        nonlocal code_buffer
        if code_buffer:
            story.append(
                Preformatted(
                    "\n".join(code_buffer),
                    style=ParagraphStyle(
                        "CodeRu",
                        fontName="Courier",
                        fontSize=8.8,
                        leading=10.5,
                        textColor=TEXT,
                        backColor=colors.HexColor("#f7f9fb"),
                        borderColor=colors.HexColor("#d9e2ec"),
                        borderWidth=0.5,
                        borderPadding=6,
                    ),
                )
            )
            story.append(Spacer(1, 3 * mm))
            code_buffer = []

    def flush_table():
        nonlocal table_buffer
        if table_buffer:
            rows = []
            for raw in table_buffer:
                if raw.strip().startswith("| ---"):
                    continue
                cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
                rows.append(cells)
            if rows:
                widths = [180 * mm / len(rows[0])] * len(rows[0])
                story.append(mk_table(rows, widths))
                story.append(Spacer(1, 3 * mm))
            table_buffer = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            flush_bullets()
            flush_table()
            in_code = not in_code
            if not in_code:
                flush_code()
            continue
        if in_code:
            code_buffer.append(line)
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            flush_bullets()
            table_buffer.append(stripped)
            continue
        flush_table()
        if not stripped:
            flush_paragraph()
            flush_bullets()
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            flush_bullets()
            story.append(p(md_inline_to_reportlab(stripped[2:]), ss["H1Ru"]))
            story.append(HRFlowable(width="100%", thickness=1.0, color=ACCENT, spaceAfter=4))
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            flush_bullets()
            h2_counter += 1
            should_break = pagebreak_h2 and h2_pagebreak_every > 0 and (h2_counter % h2_pagebreak_every == 0)
            if should_break and story and not isinstance(story[-1], PageBreak):
                story.append(PageBreak())
            story.append(p(md_inline_to_reportlab(stripped[3:]), ss["H2Ru"]))
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            flush_bullets()
            story.append(p(md_inline_to_reportlab(stripped[4:]), ss["H3Ru"]))
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            bullet_buffer.append(stripped[2:])
            continue
        if stripped[0].isdigit() and ". " in stripped[:4]:
            flush_paragraph()
            bullet_buffer.append(stripped.split(". ", 1)[1])
            continue
        paragraph_buffer.append(stripped)

    flush_paragraph()
    flush_bullets()
    flush_table()
    flush_code()


def build_story():
    ss = styles()
    story = []

    story.append(Spacer(1, 12 * mm))
    story.append(p("Руководство по эксперименту с шариком в жидком азоте", ss["TitleRu"]))
    story.append(p("Стенд, подключение, CV-измерение, коррекция по уровню LN2, бюджет погрешностей и порядок работы", ss["SubtitleRu"]))
    story.append(Spacer(1, 6 * mm))
    story.extend(fig(ROOT / "data" / "photos" / "photo_2025-07-28_13-21-07 (2).jpg", 150, "Рисунок 1. Общий вид текущего стенда.", ss))
    story.append(p("Автор: Иван Литвак<br/>Дата: 02.04.2026", ss["BodyCenterRu"]))
    story.append(Spacer(1, 8 * mm))
    story.extend(callout("Я пишу этот документ как рабочую инструкцию от первого лица: как я собираю стенд, как я снимаю серии и как я считаю погрешности. Документ должен быть полезен и без устных пояснений.", ss, title="Назначение"))
    story.append(PageBreak())

    story.append(p("Содержание", ss["H1Ru"]))
    for item in [
        "1. Основное руководство по стенду и эксперименту",
        "2. Отдельная методика по погрешностям и коррекции масштаба",
        "3. Приложения: характеристики оборудования, распиновка, требования к ноутбуку",
    ]:
        story.append(p(f"• {item}", ss["BodyRu"]))
    story.append(PageBreak())

    equip = [
        ["Узел", "Модель / характеристика", "Роль"],
        ["Камера", "Daheng MER2-231-41GM", "Основной видеосенсор стенда"],
        ["Сенсор", "Sony IMX249, 5.86 мкм, 1920x1200, 41 fps", "Базовая дискретизация и поток данных"],
        ["Объектив", "VT-LEM1214CB-H1, 12.5 мм, F1.4, 1''", "Формирование изображения на IMX249"],
        ["Кольца", "VT-C2CS, 0.5/1/5/10/20 мм", "Добор рабочего фокуса и макрорежим"],
        ["Свет", "VT-LT2-HR12076-90W + VT-LT4-2420PWDC-2", "Стабильность контраста"],
        ["Ноутбук", "16–32 ГБ RAM, SSD, Gigabit Ethernet", "Запись и real-time запуск"],
    ]
    story.append(p("Приложение A. Оборудование", ss["H1Ru"]))
    story.append(mk_table(equip, [35 * mm, 70 * mm, 75 * mm]))
    story.append(Spacer(1, 5 * mm))

    pinout = [
        ["Контакт", "Сигнал", "Назначение"],
        ["1", "Line0+", "Оптоизолированный вход +"],
        ["2", "GND", "Земля питания камеры и GPIO"],
        ["3", "Line0-", "Оптоизолированный вход -"],
        ["4", "POWER_IN", "Питание камеры +12…+24 VDC"],
        ["5", "Line2", "Программируемый GPIO I/O"],
        ["6", "Line3", "Программируемый GPIO I/O"],
        ["7", "Line1-", "Оптоизолированный выход -"],
        ["8", "Line1+", "Оптоизолированный выход +"],
    ]
    story.append(p("Приложение B. Официальная распиновка серии MERCURY2", ss["H1Ru"]))
    story.append(mk_table(pinout, [20 * mm, 38 * mm, 112 * mm]))
    story.extend(callout("Перед подачей питания нужно открыть официальный даташит серии MERCURY2 и сверить именно свой кабель. Без заземления руками камеру и I/O не трогать.", ss))
    story.append(PageBreak())

    perf = [
        ["Параметр", "Значение", "Комментарий"],
        ["Поток Mono8", "≈ 90 MiB/s", "Сырые данные 1920x1200x41 fps"],
        ["Сеть", "Gigabit Ethernet", "Нужен полноценный гигабит"],
        ["RAM", "16–32 ГБ", "Запись + обработка + preview"],
        ["Диск", "SSD / NVMe", "Жесткий диск как основной не рекомендую"],
        ["CPU", "4 ядра и выше", "Для реального времени без мучений"],
    ]
    story.append(p("Приложение C. Требования к ноутбуку", ss["H1Ru"]))
    story.append(mk_table(perf, [36 * mm, 32 * mm, 102 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.extend(fig(ROOT / "data" / "photos" / "photo_2025-07-28_14-54-51.jpg", 110, "Рисунок 2. Пример контрольного кадра с детекцией.", ss))
    story.append(PageBreak())

    append_markdown(story, ROOT / "docs" / "EXPERIMENT_MANUAL_RU.md", ss, pagebreak_h2=False, h2_pagebreak_every=1)
    story.append(PageBreak())
    append_markdown(story, ROOT / "docs" / "UNCERTAINTY_METHOD_RU.md", ss, pagebreak_h2=False, h2_pagebreak_every=1)

    story.append(PageBreak())
    story.append(p("Приложение D. Чеклист запуска стенда", ss["H1Ru"]))
    start_check = [
        ["Пункт", "Отметка"],
        ["Заземление проверено", ""],
        ["Распиновка камеры сверена с даташитом", ""],
        ["Питание камеры и света подключено корректно", ""],
        ["Ноутбук подключен по Gigabit Ethernet", ""],
        ["Камера видна в SDK", ""],
        ["Фокус и свет выставлены", ""],
        ["Ячейка чистая", ""],
        ["Конфиг серии сохранен", ""],
        ["Фото стенда сделаны", ""],
    ]
    story.append(mk_table(start_check, [145 * mm, 35 * mm]))
    story.append(PageBreak())

    story.append(p("Приложение E. Форма журнала серии", ss["H1Ru"]))
    series_log = [
        ["Поле", "Заполняемое значение"],
        ["ID серии", ""],
        ["Дата и время", ""],
        ["Оператор", ""],
        ["Камера", ""],
        ["Объектив / кольца", ""],
        ["Свет / яркость", ""],
        ["ball_to_cover_mm", ""],
        ["cell_to_cover_mm", ""],
        ["Базовый уровень LN2", ""],
        ["Подлив: время и объем", ""],
        ["Путь к видео", ""],
        ["Путь к конфигу", ""],
        ["Замечания", ""],
    ]
    story.append(mk_table(series_log, [70 * mm, 110 * mm]))
    story.append(PageBreak())

    story.append(p("Приложение F. Требования к будущей интеграции насоса", ss["H1Ru"]))
    for line in [
        "Программа должна формировать не силовую команду, а логический сигнал на внешний согласующий контур.",
        "I/O камеры использовать только после чтения официальной спецификации и схемы уровней.",
        "Нужны защита от дребезга, лог событий, таймауты и блокировка повторных срабатываний.",
        "Переходный режим после подлива должен маркироваться и исключаться из спокойной статистики.",
        "Контур управления насосом должен быть электрически и логически отделен от силовой части стенда.",
    ]:
        story.append(p(f"• {line}", ss["BodyRu"]))
    story.extend(callout("Нельзя подключать насос напрямую к камере. Сначала согласующее устройство, потом исполнительная силовая цепь.", ss, title="Интеграция насоса"))
    story.append(PageBreak())

    story.append(p("Приложение G. Приемочный протокол промышленной камеры", ss["H1Ru"]))
    accept = [
        ["Критерий", "Проверка", "Результат"],
        ["Определение в SDK", "Камера видна и открывается", ""],
        ["Поток", "Полный кадр без выпадений", ""],
        ["Фокус", "Рабочая геометрия достижима", ""],
        ["Свет", "Нет критического пересвета по кромке", ""],
        ["Эталон", "Серия с подшипником снята", ""],
        ["CV", "Preview и detection ratio приемлемы", ""],
        ["Длительная запись", "Серия без обрыва", ""],
    ]
    story.append(mk_table(accept, [55 * mm, 80 * mm, 45 * mm]))
    story.append(PageBreak())

    story.append(p("Приложение H. Пакет материалов для статьи", ss["H1Ru"]))
    article_pack = [
        ["Материал", "Статус"],
        ["Фото стенда", ""],
        ["Фото камеры и оптики", ""],
        ["Паспорт оборудования", ""],
        ["Параметры съемки", ""],
        ["Описание ячейки", ""],
        ["Метод измерения уровня", ""],
        ["Формулы погрешностей", ""],
        ["Таблицы серий", ""],
        ["Артефакты batch-анализа", ""],
        ["Выводы по точности", ""],
    ]
    story.append(mk_table(article_pack, [120 * mm, 60 * mm]))
    story.append(PageBreak())

    story.append(p("Источники", ss["H1Ru"]))
    sources = [
        "Официальная страница камеры Daheng MER2-231-41GM: https://en.daheng-imaging.com/show-104-1901-1.html",
        "Официальный PDF MERCURY2 с I/O и распиновкой: https://www.daheng-imaging.com/uploadfile/2022/1009/20221009092648317.pdf",
        "Страница объектива VT-LEM1214CB-H1: https://shop.contrastech.com/ar/products/400-1150nm-1-c-mount-fixed-lenses",
        "Локальные файлы проекта: README.md, docs/*.md, счета на оборудование, фото и видео стенда.",
    ]
    for item in sources:
        story.append(p(f"• {item}", ss["BodyRu"]))
    return story


def main():
    register_fonts()
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="Ivan Litvak stand manual",
        author="Ivan Litvak",
    )
    doc.build(build_story(), onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
