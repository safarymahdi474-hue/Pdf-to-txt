"""
بات تلگرام برای تبدیل PDF های اسکن‌شده فارسی به متن (OCR)
نسخه‌ی «لینک دانلود»: کاربر به‌جای آپلود مستقیم فایل، لینک دانلودش رو می‌فرسته.
این‌طوری هیچ فایلی از تلگرام رد نمی‌شه و محدودیت ۲۰ مگابایتی تلگرام اصلاً وارد بازی نمی‌شه.

نیازمندی‌های سیستمی (قبل از اجرا نصب کن):
    sudo apt-get install tesseract-ocr tesseract-ocr-fas poppler-utils

نیازمندی‌های پایتون:
    pip install -r requirements.txt

اجرا:
    export BOT_TOKEN="توکن بات شما"
    python bot.py
"""

import os
import io
import re
import logging
import tempfile
import asyncio
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import requests
import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image, ImageOps, ImageFilter
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

# زبان OCR: فارسی. اگه سند ترکیبی فارسی/انگلیسی است می‌تونی بذاری "fas+eng"
OCR_LANG = "fas"

# DPI تبدیل PDF به عکس — بالاتر یعنی دقت بیشتر ولی کندتر و حافظه‌برتر
# چون دقت اولویته، رو 300 گذاشتیمش
CONVERT_DPI = 300

# چند صفحه رو با هم به عکس تبدیل کنیم (نه کل PDF یه‌جا) — کنترل مصرف حافظه
# با DPI بالاتر، دسته‌ها رو کوچیک‌تر نگه می‌داریم تا حافظه سرور (مثلاً 1GB رو Railway) پر نشه
CHUNK_SIZE = 5

# چند صفحه رو موازی OCR کنیم — با DPI بالا و RAM محدود سرور، محافظه‌کارانه نگه داشته شده
OCR_WORKERS = min(2, os.cpu_count() or 1)

# تنظیمات Tesseract: oem 1 = موتور LSTM (دقت بهتر)، psm 6 = فرض یه بلوک متن یکنواخت
TESSERACT_CONFIG = "--oem 1 --psm 6"

# حداکثر حجم فایلی که دانلود می‌کنیم (بایت) — برای جلوگیری از پر شدن دیسک سرور
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 گیگابایت

URL_PATTERN = re.compile(r"https?://\S+")
DRIVE_PATTERNS = [
    re.compile(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)"),
    re.compile(r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)"),
    re.compile(r"[?&]id=([a-zA-Z0-9_-]+)"),
]


def _extract_drive_id(url: str):
    if "drive.google.com" not in url:
        return None
    for pattern in DRIVE_PATTERNS:
        m = pattern.search(url)
        if m:
            return m.group(1)
    return None


def download_from_url(url: str, dest_path: str, progress_callback=None) -> None:
    """
    فایل رو از یه لینک مستقیم یا لینک اشتراک‌گذاری Google Drive دانلود می‌کنه.
    به‌صورت استریم می‌نویسه رو دیسک تا کل فایل تو حافظه نره.
    """
    session = requests.Session()
    drive_id = _extract_drive_id(url)

    if drive_id:
        base = "https://drive.google.com/uc?export=download"
        response = session.get(base, params={"id": drive_id}, stream=True)

        # فایل‌های حجیم گوگل‌درایو یه صفحه تأیید نشون می‌دن؛ باید توکنش رو بگیریم
        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break
        if token:
            response = session.get(
                base, params={"id": drive_id, "confirm": token}, stream=True
            )
    else:
        response = session.get(url, stream=True)

    response.raise_for_status()

    downloaded = 0
    last_reported_mb = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded > MAX_DOWNLOAD_BYTES:
                raise ValueError("حجم فایل از حد مجاز (۲ گیگابایت) بیشتر شد.")
            if progress_callback:
                mb = downloaded // (1024 * 1024)
                if mb - last_reported_mb >= 20:
                    last_reported_mb = mb
                    progress_callback(mb)


def _preprocess_image(img: Image.Image) -> Image.Image:
    """
    قبل از OCR، تصویر رو برای خوانایی بهتر آماده می‌کنه:
    سیاه‌وسفید + افزایش کنتراست خودکار + شارپ کردن جزئی.
    این کار رو اسکن‌های کدر/کم‌نور معمولاً دقت رو محسوس بالا می‌بره.
    """
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.SHARPEN)
    return gray


def _ocr_single_image_bytes(png_bytes: bytes) -> str:
    """یک صفحه رو OCR می‌کنه. تو یه پردازش جدا اجرا می‌شه (برای موازی‌سازی)."""
    img = Image.open(io.BytesIO(png_bytes))
    img = _preprocess_image(img)
    return pytesseract.image_to_string(img, lang=OCR_LANG, config=TESSERACT_CONFIG)


def ocr_pdf_streaming(pdf_path: str, out_path: str, status_callback=None) -> int:
    """
    PDF رو دسته‌دسته (CHUNK_SIZE صفحه در هر دسته) به عکس تبدیل می‌کنه،
    هر دسته رو موازی OCR می‌کنه و نتیجه رو بلافاصله رو دیسک می‌نویسه.
    """
    info = pdfinfo_from_path(pdf_path)
    total_pages = info["Pages"]

    with open(out_path, "w", encoding="utf-8") as out_file, \
         ProcessPoolExecutor(max_workers=OCR_WORKERS) as pool:

        for start in range(1, total_pages + 1, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE - 1, total_pages)

            images = convert_from_path(
                pdf_path, dpi=CONVERT_DPI, first_page=start, last_page=end
            )

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


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    match = URL_PATTERN.search(text)

    if not match:
        await update.message.reply_text(
            "یه لینک دانلود PDF (مثلاً لینک اشتراک‌گذاری Google Drive یا هر "
            "لینک مستقیم دیگه) برام بفرست تا متنش رو استخراج کنم."
        )
        return

    url = match.group(0)
    status_msg = await update.message.reply_text("📥 در حال دانلود فایل از لینک...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / "input.pdf"

        loop = asyncio.get_event_loop()

        def dl_progress(mb):
            asyncio.run_coroutine_threadsafe(
                status_msg.edit_text(f"📥 در حال دانلود... {mb} مگابایت گرفته شد"),
                loop,
            )

        try:
            await loop.run_in_executor(
                None, download_from_url, url, str(pdf_path), dl_progress
            )
        except Exception as e:
            logger.exception("Download failed")
            await status_msg.edit_text(f"❌ دانلود فایل شکست خورد: {e}")
            return

        await status_msg.edit_text("🔍 در حال تبدیل PDF به تصویر و اجرای OCR...")

        last_reported = {"page": 0}

        def status_callback(current_page, total_pages):
            if current_page - last_reported["page"] >= 50 or current_page == total_pages:
                last_reported["page"] = current_page
                asyncio.run_coroutine_threadsafe(
                    status_msg.edit_text(
                        f"🔍 در حال OCR... صفحه {current_page} از {total_pages}"
                    ),
                    loop,
                )

        out_path = Path(tmp_dir) / "result.txt"

        try:
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
        "یه لینک دانلود PDF برام بفرست (نه خود فایل) تا متنش رو استخراج کنم."
    )


def main():
    if not BOT_TOKEN:
        raise RuntimeError("متغیر محیطی BOT_TOKEN تنظیم نشده.")

    builder = Application.builder().token(BOT_TOKEN)
    builder = builder.read_timeout(120).write_timeout(120).connect_timeout(60)
    app = builder.build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(~filters.TEXT, handle_other))

    logger.info("بات در حال اجراست...")
    app.run_polling()


if __name__ == "__main__":
    main()
