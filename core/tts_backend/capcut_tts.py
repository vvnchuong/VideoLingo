"""
capcut_tts.py

Module gọi CapCut TTS (SAMI engine) để sinh audio cho từng câu/segment,
dùng lại toàn bộ logic build request/sign của capcut_common_task_client.py
(thuần Python + requests, không native lib).

Cách dùng:
    from capcut_tts import synthesize, synthesize_batch

    path = synthesize(
        text="Xin chào, đây là thử nghiệm",
        voice="BV074_streaming",
        resource_id="7102355709945188865",
        out_path="output/segment_001.mp3",
    )

YÊU CẦU: đặt file này CÙNG THƯ MỤC với capcut_common_task_client.py
(hoặc thêm path của nó vào sys.path trước khi import module này).
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import requests

from .capcut_common_task_client import (
    BASE,
    DEFAULT_DEVICE,
    build_request,
    checked_json_response,
    common_query,
    base_headers,
    make_sign_header,
    make_tts_payload_sign,
    escape_xml,
    compact_json,
    query_body,
)

logger = logging.getLogger("capcut_tts")

# ---------------------------------------------------------------------------
# Cấu hình retry / device pool
# ---------------------------------------------------------------------------

MAX_RETRIES_SAME_DEVICE = 4      # thử lại cùng 1 device_id tối đa 4 lần
RETRY_BACKOFF_SECONDS = 1.5      # chờ giữa các lần retry (nhân dần theo attempt)
POLL_INTERVAL_SECONDS = 1.0      # nghỉ giữa các lần poll tts-query
POLL_TIMEOUT_SECONDS = 30        # timeout tổng chờ 1 task xong

# Pool device_id dự phòng, dùng khi device hiện tại bị "shark block" liên tục.
# DEFAULT_DEVICE gốc (None) đã xác nhận bị block do dùng chung bởi mọi người
# clone repo -> bỏ ra khỏi pool để đỡ tốn ~10s retry vô ích mỗi lần gọi.
DEVICE_OVERRIDE_POOL: list[dict | None] = [
    {"device_id": "7342995123456789012", "iid": "7342995198765432109", "tdid": "7342995123456789012"},
    {"device_id": "7453006234567890123", "iid": "7453006298765432109", "tdid": "7453006234567890123"},
    {"device_id": "7564117345678901234", "iid": "7564117398765432109", "tdid": "7564117345678901234"},
    {"device_id": "7675228456789012345", "iid": "7675228498765432109", "tdid": "7675228456789012345"},
    {"device_id": "7786339567890123456", "iid": "7786339598765432109", "tdid": "7786339567890123456"},
]

# Pool riêng cho batch round-robin (synthesize_batch_fast).
# KHÔNG chạy song song thật (đa luồng) từ 1 IP để tránh burst pattern dễ bị
# risk-control phát hiện là automation/device-farm — thay vào đó LUÂN PHIÊN
# tuần tự qua từng device_id với delay nhẹ. Vẫn nhanh hơn hẳn tuần tự 1 device
# vì không phải chờ hồi phục của cùng 1 identity giữa các request liên tiếp.
ROUND_ROBIN_DEVICE_POOL: list[dict] = [
    {"device_id": "7165910902483344908", "iid": "7019915595421604274", "tdid": "7165910902483344908"},
    {"device_id": "7231884012345678901", "iid": "7231884098765432109", "tdid": "7231884098765432109"},
    {"device_id": "7318765432109876543", "iid": "7318765401234567890", "tdid": "7318765432109876543"},
]

# Delay nhẹ giữa mỗi request trong round-robin (giây). Đủ nhỏ để không chậm,
# đủ lớn để không tạo pattern "spam tức thời" từ cùng 1 IP.
ROUND_ROBIN_DELAY_SECONDS = 0.4


class CapCutTTSError(Exception):
    """Raised khi tạo/lấy audio từ CapCut TTS thất bại sau khi đã retry hết pool."""


def _make_args(mode: str, device_override: dict | None, **kwargs) -> SimpleNamespace:
    """Tạo namespace giống argparse để tái sử dụng build_request() không sửa file gốc."""
    base = dict(
        mode=mode,
        device_json=None,
        text=None,
        text_file=None,
        voice="BV074_streaming",
        resource_id="7102355709945188865",
        rate="1.0",
        audio_vid=None,
        audio_md5=None,
        audio_file=None,
        duration_ms=None,
        language="vi-VN",
        translation_language="vi-VN",
        use_translation=False,
        task_id=None,
        token=None,
        bind_id="",
    )
    base.update(kwargs)
    args = SimpleNamespace(**base)

    # build_request() gọi load_json(args.device_json, {}) rồi device.update(...)
    # -> nếu có override, ghi ra file tạm để tái dùng đúng cơ chế gốc.
    if device_override:
        tmp_path = Path(".capcut_device_override_tmp.json")
        tmp_path.write_text(json.dumps(device_override), encoding="utf-8")
        args.device_json = str(tmp_path)
    return args


def _post(mode: str, device_override: dict | None, **kwargs) -> dict:
    args = _make_args(mode, device_override, **kwargs)
    url, headers, body_text = build_request(args)
    resp = requests.post(url, headers=headers, data=body_text.encode("utf-8"), timeout=30)
    return checked_json_response(resp, mode)


def _is_shark_block(response: dict) -> bool:
    return str(response.get("ret")) == "-6" or "shark block" in str(response.get("errmsg", "")).lower()


def _extract_speech_url(query_response: dict) -> str | None:
    task = query_response["data"]["tasks"][0]
    if task.get("status") != "succeed":
        return None
    payload = json.loads(task["payload"])
    subtitles = payload.get("audio_subtitles") or []
    if not subtitles:
        return None
    return subtitles[0].get("speech_url")


def _download(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path


# ---------------------------------------------------------------------------
# Merged batch: gộp nhiều câu vào 1 request tts-new duy nhất (giảm số lượng
# request lên server rất nhiều, thay vì 1 request/câu như synthesize()).
#
# LƯU Ý: CapCut vẫn trả về 1 file MP3 RIÊNG cho mỗi câu (need_merge_voice=True
# không thật sự gộp thành 1 file dài) -- nhưng vẫn giảm được số request vì
# nhiều câu được xử lý trong CÙNG 1 lần gọi tts-new + tts-query.
# ---------------------------------------------------------------------------

# Giới hạn ký tự SSML CapCut cho phép ~10.000 (theo bản web). Dùng ngưỡng
# thấp hơn để chừa margin an toàn cho XML tag overhead (<voice>, <prosody>...).
MAX_CHARS_PER_BATCH = 8000


def _chunk_segments_by_char_limit(
    segments: list[dict], max_chars: int = MAX_CHARS_PER_BATCH
) -> list[list[dict]]:
    """
    Chia segments thành nhiều batch, mỗi batch có tổng độ dài text không vượt
    quá max_chars. Không bao giờ cắt giữa 1 câu -- mỗi câu luôn nguyên vẹn
    trong đúng 1 batch. Giữ nguyên thứ tự gốc.
    """
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_len = 0

    for seg in segments:
        text_len = len(seg["text"])
        if current and current_len + text_len > max_chars:
            batches.append(current)
            current = []
            current_len = 0
        current.append(seg)
        current_len += text_len

    if current:
        batches.append(current)

    return batches


def _build_merged_tts_new_body(texts: list[str], voice: str, resource_id: str, rate: str, device: dict):
    """
    Bản có bật need_merge_voice=True và need_subtitle_timestamp=True,
    dựa trên tts_new_body() gốc trong capcut_common_task_client.py.
    Đã verify bằng test thật (xem test_merge_timestamp.py) -- response trả
    về audio_subtitles là list N object, mỗi object có speech_url + utterances
    (word-level timestamp) riêng cho từng câu.
    """
    babi = {
        "feature_entrance": "editor",
        "feature_entrance_detail": "editor-feature-text_to_speech",
        "feature_key": "text_to_speech",
        "scenario": "video_editor",
    }
    voice_blocks = []
    for text in texts:
        voice_blocks.append(
            f'    <voice name="{voice}" mock_tone_info="" platform="sami" '
            f'resource_id="{resource_id}" emotion="" emotion_scale="0" style="" role="" '
            f'moyin_emotion="" is_clone_tone="false" need_subtitle_timestamp="true">\n'
            f'        <prosody rate="{rate}">{escape_xml(text)}</prosody>\n'
            f'    </voice>'
        )
    ssml = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">\n'
        + "\n".join(voice_blocks)
        + "\n</speak>"
    )
    extra_info = compact_json({"benefit_info": {}})
    payload = {
        "audio_format": "mp3",
        "babi_param": compact_json(babi),
        "credit_disable": False,
        "extra_info": extra_info,
        "need_merge_voice": True,
        "need_subtitle_timestamp": True,
        "scene": "text_to_speech",
        "ssml": ssml,
    }
    payload["sign"] = make_tts_payload_sign(ssml, extra_info, device["device_id"], device["aid"])
    body = {
        "bind_id": str(uuid.uuid4()),
        "can_queue": True,
        "enter_from": "text_to_speech",
        "tasks": [
            {
                "context": str(uuid.uuid4()),
                "payload": compact_json(payload),
                "req_key": "sami_text_to_speech",
                "task_version": "v3",
            }
        ],
    }
    return babi, body


def _build_merged_request(mode: str, device: dict, **kwargs):
    """
    Tương đương build_request() gốc nhưng dùng _build_merged_tts_new_body()
    cho nhánh tts-new (để bật 2 flag merge/timestamp).
    """
    if mode == "tts-new":
        babi, body = _build_merged_tts_new_body(
            kwargs["texts"], kwargs["voice"], kwargs["resource_id"], kwargs["rate"], device
        )
        path = "/lv/v1/common_task/new"
        query = common_query(device, babi, include_region=True)
        appid = True
    elif mode == "tts-query":
        body = query_body(kwargs["task_id"], kwargs["token"], "sami_text_to_speech", kwargs.get("bind_id", ""))
        path = "/lv/v1/common_task/query"
        query = common_query(device, None, include_region=False)
        appid = True
    else:
        raise ValueError(f"mode không hỗ trợ: {mode}")

    body_text = compact_json(body)
    url = BASE + path + "?" + urlencode(query)
    headers = base_headers(device, body_text, appid=appid)
    lower_headers = {k.lower(): v for k, v in headers.items()}
    if "sign" not in lower_headers:
        headers["sign"] = make_sign_header(url, device["appvr"], lower_headers["device-time"], device["tdid"])
    return url, headers, body_text


def _post_merged(mode: str, device: dict, **kwargs) -> dict:
    url, headers, body_text = _build_merged_request(mode, device, **kwargs)
    resp = requests.post(url, headers=headers, data=body_text.encode("utf-8"), timeout=30)
    return checked_json_response(resp, mode)


def _synthesize_one_batch(
    texts: list[str],
    voice: str,
    resource_id: str,
    rate: str,
    device: dict,
) -> list[dict]:
    """
    Gửi 1 batch (nhiều câu) trong 1 request tts-new, poll tới khi xong,
    trả về list audio_subtitles gốc (mỗi phần tử có speech_url + utterances
    + text, theo ĐÚNG thứ tự texts đầu vào).

    Raises CapCutTTSError nếu request thất bại (shark block, timeout, v.v).
    """
    created = _post_merged("tts-new", device, texts=texts, voice=voice, resource_id=resource_id, rate=rate)

    if _is_shark_block(created):
        raise CapCutTTSError(f"shark block (device={device['device_id']})")
    if str(created.get("ret")) != "0":
        raise CapCutTTSError(f"tts-new lỗi: {created.get('errmsg')}")

    task = created["data"]["tasks"][0]
    task_id, token = task["id"], task["token"]

    deadline = time.time() + POLL_TIMEOUT_SECONDS * max(1, len(texts) // 20)  # batch lớn cần chờ lâu hơn
    while time.time() < deadline:
        query_resp = _post_merged("tts-query", device, task_id=task_id, token=token)
        status = query_resp["data"]["tasks"][0].get("status")
        if status == "succeed":
            payload = json.loads(query_resp["data"]["tasks"][0]["payload"])
            return payload.get("audio_subtitles") or []
        if status == "failed":
            raise CapCutTTSError(f"Batch task failed: {query_resp}")
        time.sleep(POLL_INTERVAL_SECONDS)

    raise CapCutTTSError("Timeout chờ batch xử lý xong")


def synthesize_merged_batch(
    segments: list[dict],
    voice: str,
    resource_id: str,
    out_dir: str,
    rate: str = "1.0",
    filename_prefix: str = "segment",
    max_chars_per_batch: int = MAX_CHARS_PER_BATCH,
    device_override: dict | None = None,
) -> list[dict]:
    """
    Sinh audio cho nhiều segment, TỰ ĐỘNG CHIA thành nhiều batch theo giới
    hạn ký tự SSML (mặc định 8000), mỗi batch = 1 request tts-new + 1 request
    tts-query -- giảm số lượng request lên server RẤT NHIỀU so với
    synthesize_batch()/synthesize_batch_fast() (vốn 1 request/câu).

    Ví dụ: video 200 câu (~10.000 ký tự) -> chỉ cần 1-2 batch thay vì
    200 request riêng lẻ.

    segments: list dict {"text": "...", "id": tuỳ ý}.
    device_override: nếu không truyền, dùng device_id đầu tiên trong
        DEVICE_OVERRIDE_POOL. Toàn bộ batch trong 1 lần gọi dùng CHUNG 1
        device (vì số lượng request đã rất thấp, không cần rotate).

    Trả về list dict, mỗi phần tử gồm:
        {"id": ..., "path": "...", "text": "...", "utterances": [...], "duration": ...}
    theo ĐÚNG thứ tự segments đầu vào.

    Raises:
        CapCutTTSError nếu 1 batch nào đó thất bại.
    """
    device = copy.deepcopy(DEFAULT_DEVICE)
    device.update(device_override or DEVICE_OVERRIDE_POOL[0])

    out_dir_p = Path(out_dir)
    batches = _chunk_segments_by_char_limit(segments, max_chars_per_batch)
    logger.info("Chia %d câu thành %d batch (giới hạn %d ký tự/batch)", len(segments), len(batches), max_chars_per_batch)

    results: list[dict] = []
    global_index = 0

    for batch_num, batch_segments in enumerate(batches, start=1):
        texts = [seg["text"] for seg in batch_segments]
        logger.info("Batch %d/%d: %d câu, tổng %d ký tự", batch_num, len(batches), len(texts), sum(len(t) for t in texts))

        audio_subtitles = _synthesize_one_batch(texts, voice, resource_id, rate, device)

        if len(audio_subtitles) != len(batch_segments):
            logger.warning(
                "Batch %d: số kết quả trả về (%d) khác số câu gửi đi (%d) -- kiểm tra lại thứ tự!",
                batch_num, len(audio_subtitles), len(batch_segments),
            )

        for seg, subtitle in zip(batch_segments, audio_subtitles):
            seg_id = seg.get("id", global_index)
            suffix = f"{seg_id:04d}" if isinstance(seg_id, int) else str(seg_id)
            out_path = out_dir_p / f"{filename_prefix}_{suffix}.mp3"

            speech_url = subtitle.get("speech_url")
            if speech_url:
                _download(speech_url, out_path)

            results.append({
                "id": seg_id,
                "path": str(out_path) if speech_url else None,
                "text": subtitle.get("text", seg["text"]),
                "utterances": subtitle.get("utterances", []),
                "duration_ms": subtitle.get("duration"),
            })
            global_index += 1

    return results


# ---------------------------------------------------------------------------
# API chính (từng câu 1 request -- dùng khi cần rotate nhiều device hoặc
# khi cần retry chi tiết theo từng câu riêng lẻ)
# ---------------------------------------------------------------------------

def synthesize(
    text: str,
    voice: str,
    resource_id: str,
    out_path: str,
    rate: str = "1.0",
) -> str:
    """
    Sinh audio cho 1 đoạn text, trả về path file đã lưu.
    Tự retry cùng device_id vài lần; nếu vẫn bị "shark block" thì rotate
    sang device_id khác trong DEVICE_OVERRIDE_POOL.

    Raises:
        CapCutTTSError nếu thất bại sau khi thử hết pool.
    """
    out_path_p = Path(out_path)
    last_error: Exception | None = None

    for device_override in DEVICE_OVERRIDE_POOL:
        device_label = device_override["device_id"] if device_override else "DEFAULT_DEVICE"

        for attempt in range(1, MAX_RETRIES_SAME_DEVICE + 1):
            try:
                logger.info("tts-new: device=%s attempt=%d/%d", device_label, attempt, MAX_RETRIES_SAME_DEVICE)
                created = _post(
                    "tts-new",
                    device_override,
                    text=[text],
                    voice=voice,
                    resource_id=resource_id,
                    rate=rate,
                )
            except Exception as exc:
                last_error = exc
                logger.warning("tts-new lỗi mạng/http: %s", exc)
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue

            if _is_shark_block(created):
                last_error = CapCutTTSError(f"shark block (device={device_label})")
                logger.warning("Bị shark block với device=%s, thử lại...", device_label)
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue

            if str(created.get("ret")) != "0":
                last_error = CapCutTTSError(f"tts-new lỗi: {created}")
                logger.warning("tts-new trả lỗi khác: %s", created.get("errmsg"))
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue

            task = created["data"]["tasks"][0]
            task_id, token = task["id"], task["token"]

            deadline = time.time() + POLL_TIMEOUT_SECONDS
            speech_url = None
            while time.time() < deadline:
                query_resp = _post("tts-query", device_override, task_id=task_id, token=token)
                status = query_resp["data"]["tasks"][0].get("status")

                if status == "succeed":
                    speech_url = _extract_speech_url(query_resp)
                    break
                if status == "failed":
                    last_error = CapCutTTSError(f"Task failed: {query_resp}")
                    break
                time.sleep(POLL_INTERVAL_SECONDS)

            if speech_url:
                _download(speech_url, out_path_p)
                logger.info("Đã lưu audio: %s", out_path_p)
                return str(out_path_p)

            logger.warning("Không lấy được speech_url (device=%s), retry...", device_label)
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise CapCutTTSError(
        f"Thất bại sau khi thử {len(DEVICE_OVERRIDE_POOL)} device(s). Lỗi cuối: {last_error}"
    )


def synthesize_batch(
    segments: list[dict],
    voice: str,
    resource_id: str,
    out_dir: str,
    rate: str = "1.0",
    filename_prefix: str = "segment",
) -> list[str]:
    """
    Sinh audio cho nhiều segment (dùng cho flow tách câu -> ghép lại kiểu VideoLingo).

    segments: list dict tối thiểu có {"text": "..."} — có thể thêm "id" tuỳ ý,
              nếu không có "id" sẽ dùng index.
    Trả về list path audio theo đúng thứ tự segments đầu vào.
    """
    out_dir_p = Path(out_dir)
    paths = []
    for i, seg in enumerate(segments):
        seg_id = seg.get("id", i)
        suffix = f"{seg_id:04d}" if isinstance(seg_id, int) else str(seg_id)
        out_path = out_dir_p / f"{filename_prefix}_{suffix}.mp3"
        path = synthesize(
            text=seg["text"],
            voice=voice,
            resource_id=resource_id,
            out_path=str(out_path),
            rate=rate,
        )
        paths.append(path)
    return paths


def _synthesize_one_attempt(
    text: str,
    voice: str,
    resource_id: str,
    out_path: Path,
    rate: str,
    device_override: dict,
) -> bool:
    """
    Thử tạo audio với ĐÚNG 1 device_id, KHÔNG retry nội bộ (round-robin tự
    xử lý việc đổi device ở tầng ngoài). Trả về True nếu thành công.
    """
    try:
        created = _post(
            "tts-new",
            device_override,
            text=[text],
            voice=voice,
            resource_id=resource_id,
            rate=rate,
        )
    except Exception as exc:
        logger.warning("tts-new lỗi mạng/http (device=%s): %s", device_override["device_id"], exc)
        return False

    if _is_shark_block(created) or str(created.get("ret")) != "0":
        logger.warning("device=%s bị chặn/lỗi: %s", device_override["device_id"], created.get("errmsg"))
        return False

    task = created["data"]["tasks"][0]
    task_id, token = task["id"], task["token"]

    deadline = time.time() + POLL_TIMEOUT_SECONDS
    speech_url = None
    while time.time() < deadline:
        try:
            query_resp = _post("tts-query", device_override, task_id=task_id, token=token)
        except Exception as exc:
            logger.warning("tts-query lỗi mạng/http: %s", exc)
            return False
        status = query_resp["data"]["tasks"][0].get("status")
        if status == "succeed":
            speech_url = _extract_speech_url(query_resp)
            break
        if status == "failed":
            logger.warning("Task failed (device=%s)", device_override["device_id"])
            return False
        time.sleep(POLL_INTERVAL_SECONDS)

    if not speech_url:
        return False

    _download(speech_url, out_path)
    return True


def synthesize_batch_fast(
    segments: list[dict],
    voice: str,
    resource_id: str,
    out_dir: str,
    rate: str = "1.0",
    filename_prefix: str = "segment",
    device_pool: list[dict] | None = None,
    delay_seconds: float | None = None,
) -> list[str]:
    """
    Sinh audio cho nhiều segment, LUÂN PHIÊN qua nhiều device_id (round-robin)
    thay vì lặp lại 1 device_id cho tất cả. Không dùng đa luồng thật (tránh
    burst pattern dễ bị phát hiện là automation từ 1 IP) — nhưng vẫn nhanh
    hơn hẳn vì mỗi request dùng 1 identity khác nhau, không cần chờ hồi phục
    của cùng 1 device giữa các câu liên tiếp.

    Nếu 1 device bị block/lỗi giữa chừng, tự động thử device tiếp theo trong
    pool cho ĐÚNG câu đó (không rotate ngược lại device đã chết trong phiên
    chạy này).

    segments: list dict tối thiểu có {"text": "..."} — có thể thêm "id" tuỳ ý.
    device_pool: mặc định dùng ROUND_ROBIN_DEVICE_POOL nếu không truyền vào.
    delay_seconds: mặc định dùng ROUND_ROBIN_DELAY_SECONDS nếu không truyền vào.

    Raises:
        CapCutTTSError nếu 1 câu nào đó fail hết toàn bộ pool device khả dụng.
    """
    pool = list(device_pool or ROUND_ROBIN_DEVICE_POOL)
    delay = ROUND_ROBIN_DELAY_SECONDS if delay_seconds is None else delay_seconds

    if not pool:
        raise CapCutTTSError("device_pool rỗng, không có device nào để dùng.")

    out_dir_p = Path(out_dir)
    paths: list[str] = []
    dead_devices: set[str] = set()  # device_id đã xác nhận chết trong phiên này
    pool_index = 0

    def _next_alive_device() -> dict | None:
        nonlocal pool_index
        attempts = 0
        while attempts < len(pool):
            candidate = pool[pool_index % len(pool)]
            pool_index += 1
            attempts += 1
            if candidate["device_id"] not in dead_devices:
                return candidate
        return None

    for i, seg in enumerate(segments):
        seg_id = seg.get("id", i)
        suffix = f"{seg_id:04d}" if isinstance(seg_id, int) else str(seg_id)
        out_path = out_dir_p / f"{filename_prefix}_{suffix}.mp3"

        success = False
        tried_this_segment: set[str] = set()

        while len(tried_this_segment) < len(pool):
            device = _next_alive_device()
            if device is None:
                break  # hết device sống trong pool

            device_id = device["device_id"]
            if device_id in tried_this_segment:
                # đã thử device này cho câu hiện tại rồi, tránh lặp vô hạn
                if len(tried_this_segment) >= len(pool) - len(dead_devices):
                    break
                continue
            tried_this_segment.add(device_id)

            logger.info("segment=%s device=%s", suffix, device_id)
            ok = _synthesize_one_attempt(seg["text"], voice, resource_id, out_path, rate, device)

            if ok:
                success = True
                paths.append(str(out_path))
                break
            else:
                # Đánh dấu chết TẠM THỜI trong phiên này để không quay lại ngay;
                # nếu muốn thử lại device đó cho câu SAU, xoá dòng dưới.
                dead_devices.add(device_id)
                logger.warning("Đánh dấu device=%s là chết trong phiên này, chuyển device khác.", device_id)

        if not success:
            raise CapCutTTSError(
                f"Segment '{suffix}' thất bại với toàn bộ {len(pool)} device khả dụng "
                f"(đã chết: {dead_devices})."
            )

        time.sleep(delay)

    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = synthesize(
        text="Xin chào, đây là bài kiểm tra module CapCut TTS.",
        voice="BV074_streaming",
        resource_id="7102355709945188865",
        out_path="test_output/demo.mp3",
    )
    print("Đã tạo:", result)