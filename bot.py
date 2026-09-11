"""
بات تلگرام برای تبدیل PDF های اسکن‌شده فارسی به متن (OCR)

نیازمندی‌های سیستمی (قبل از اجرا نصب کن):
    sudo apt-get install tesseract-ocr tesseract-ocr-fas poppler-utils

نیازمندی‌های پایتون:
    pip install python-telegram-bot pytesseract pdf2image Pillow

اجرا:
    export BOT_TOKEN="توکن بات شما"
    python bot.py
"""

import os
import io
import logging
import tempfile
import asyncio
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN")

# آدرس Bot API سرور محلی (برای فایل‌های حجیم بالای 20 مگابایت اجباریه)
# نمونه: "http://localhost:8081/bot"  و  "http://localhost:8081/file/bot"
LOCAL_API_BASE_URL = os.environ.get("LOCAL_API_BASE_URL")
LOCAL_API_BASE_FILE_URL = os.environ.get("LOCAL_API_BASE_FILE_URL")

# زبان OCR: فارسی. اگه سند ترکیبی فارسی/انگلیسی است می‌تونی بذاری "fas+eng"
OCR_LANG = "fas"

# DPI تبدیل PDF به عکس — بالاتر یعنی دقت بیشتر ولی کندتر و حافظه‌برتر
# برای فایل‌های خیلی حجیم پیشنهاد می‌شه 200 باشه، نه 300
CONVERT_DPI = 200

# چند صفحه رو با هم به عکس تبدیل کنیم (نه کل PDF یه‌جا) — کنترل مصرف حافظه
CHUNK_SIZE = 10

# چند صفحه رو موازی OCR کنیم — بسته به تعداد هسته CPU سرورت تنظیم کن
OCR_WORKERS = min(4, os.cpu_count() or 2)


def _ocr_single_image_bytes(png_bytes: bytes) -> str:
    """یک صفحه رو OCR می‌کنه. تو یه پردازش جدا اجرا می‌شه (برای موازی‌سازی)."""
    img = Image.open(io.BytesIO(png_bytes))
    return pytesseract.image_to_string(img, lang=OCR_LANG)


def ocr_pdf_streaming(pdf_path: str, out_path: str, status_callback=None) -> int:
    """
    PDF رو دسته‌دسته (CHUNK_SIZE صفحه در هر دسته) به عکس تبدیل می‌کنه،
    هر دسته رو موازی OCR می‌کنه و نتیجه رو بلافاصله رو دیسک می‌نویسه.
    این‌طوری حافظه هیچ‌وقت کل فایل رو نگه نمی‌داره و اگه یه‌جا قطع بشه
    صفحات قبلی از دست نمی‌رن. تعداد کل صفحات رو برمی‌گردونه.
    """
    info = pdfinfo_from_path(pdf_path)
    total_pages = info["Pages"]

    with open(out_path, "w", encoding="utf-8") as out_file, \
         ProcessPoolExecutor(max_workers=OCR_WORKERS) as pool:

        for start in range(1, total_pages + 1, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE - 1, total_pages)

            # فقط همین بازه از صفحات به عکس تبدیل می‌شه، نه کل فایل
            images = convert_from_path(
                pdf_path, dpi=CONVERT_DPI, first_page=start, last_page=end
            )

            # عکس‌ها رو به bytes تبدیل می‌کنیم تا بین پردازش‌ها قابل ارسال باشن
            image_bytes_list = []
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_bytes_list.append(buf.getvalue())

            results = list(pool.map(_ocr_single_image_bytes, image_bytes_list))

            for offset, text in enumerate(results):
                page_num = start + offset
                out_file.write(f"--- صفحه {page_num} ---\n{text.strip()}\n\n")

            out_file.flush()

            if status_callback:
                status_callback(end, total_pages)

    return total_pages


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document

    if not doc:
        return

    file_name = doc.file_name or "file"
    if not file_name.lower().endswith(".pdf"):
        await update.message.reply_text(
            "فعلاً فقط فایل PDF پشتیبانی می‌شه. لطفاً یه فایل PDF بفرست."
        )
        return

    status_msg = await update.message.reply_text(
        "📥 فایل دریافت شد. در حال دانلود..."
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / file_name
        tg_file = await doc.get_file()
        await tg_file.download_to_drive(custom_path=str(pdf_path))

        await status_msg.edit_text("🔍 در حال تبدیل PDF به تصویر و اجرای OCR...")

        loop = asyncio.get_event_loop()

        last_reported = {"page": 0}

        def status_callback(current_page, total_pages):
            # برای فایل‌های حجیم هر ۵۰ صفحه یا در صفحه آخر آپدیت بده
            if current_page - last_reported["page"] >= 50 or current_page == total_pages:
                last_reported["page"] = current_page
                asyncio.run_coroutine_threadsafe(
                    status_msg.edit_text(
                        f"🔍 در حال OCR... صفحه {current_page} از {total_pages}"
                    ),
                    loop,
                )

        out_path = Path(tmp_dir) / f"{Path(file_name).stem}.txt"

        try:
            # OCR کار سنگین CPU هست، تو thread جدا اجرا می‌کنیم تا بات بلاک نشه
            # نتیجه به‌صورت تدریجی رو دیسک نوشته می‌شه (ocr_pdf_streaming)
            await loop.run_in_executor(
                None, ocr_pdf_streaming, str(pdf_path), str(out_path), status_callback
            )
        except Exception as e:
            logger.exception("OCR failed")
            partial_note = (
                " (متن تا همون‌جایی که پردازش شده بود رو می‌فرستم)"
                if out_path.exists() and out_path.stat().st_size > 0
                else ""
            )
            await status_msg.edit_text(f"❌ خطا در پردازش فایل: {e}{partial_note}")
            if out_path.exists() and out_path.stat().st_size > 0:
                await update.message.reply_document(document=open(out_path, "rb"))
            return

        if not out_path.exists() or out_path.stat().st_size == 0:
            await status_msg.edit_text(
                "⚠️ متنی از فایل استخراج نشد. کیفیت اسکن رو بررسی کن."
            )
            return

        await status_msg.edit_text("✅ تمام شد. فایل متنی رو می‌فرستم...")
        await update.message.reply_document(document=open(out_path, "rb"))


async def handle_other(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "یه فایل PDF اسکن‌شده برام بفرست تا متنش رو استخراج کنم."
    )


def main():
    if not BOT_TOKEN:
        raise RuntimeError("متغیر محیطی BOT_TOKEN تنظیم نشده.")

    builder = Application.builder().token(BOT_TOKEN)

    # برای فایل‌های بالای 20 مگابایت، باید Bot API سرور محلی راه‌اندازی شده باشه
    # و آدرسش رو با LOCAL_API_BASE_URL / LOCAL_API_BASE_FILE_URL بدی
    if LOCAL_API_BASE_URL and LOCAL_API_BASE_FILE_URL:
        builder = builder.base_url(LOCAL_API_BASE_URL).base_file_url(
            LOCAL_API_BASE_FILE_URL
        )
        logger.info("در حال استفاده از Bot API سرور محلی: %s", LOCAL_API_BASE_URL)
    else:
        logger.warning(
            "LOCAL_API_BASE_URL تنظیم نشده — با API ابری تلگرام، دانلود فایل‌های "
            "بالای 20 مگابایت شکست می‌خوره."
        )

    # تایم‌اوت‌های بالاتر برای دانلود/آپلود فایل‌های حجیم
    builder = builder.read_timeout(120).write_timeout(120).connect_timeout(60)

    app = builder.build()

    app.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    app.add_handler(MessageHandler(~filters.Document.PDF, handle_other))

    logger.info("بات در حال اجراست...")
    app.run_polling()


if __name__ == "__main__":
    main()
