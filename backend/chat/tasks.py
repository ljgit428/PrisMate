import base64
import json
import logging
import os
import re
import tempfile
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import google.generativeai as genai
import requests
from celery import shared_task

from .attachments import (
    MAX_VIDEO_ATTACHMENT_BYTES,
    describe_attachment_for_prompt,
    get_message_attachments,
)
from .memory.filesystem import CharacterMemoryFilesystem
from .memory.manager import MemoryManager
from .memory.prompts import build_memory_extraction_prompt, get_memory_crud_tool_specs
from .models import (
    AttachmentKind,
    Character,
    CharacterMemoryItem,
    ChatSession,
    Message,
    ModelConfiguration,
    ModelRole,
    ModelRoleAssignment,
    UserProfile,
)
from .search import search_web
from .soul import (
    build_character_prompt_context,
    build_memory_explorer_manifest,
)

logger = logging.getLogger(__name__)

ANTHROPIC_DEFAULT_BASE_URL = 'https://api.anthropic.com'
ANTHROPIC_API_VERSION = '2023-06-01'
ANTHROPIC_COMPLETION_MAX_TOKENS = 8192
ANTHROPIC_MEDIA_ANALYSIS_MAX_TOKENS = 1024
MEDIA_ANALYSIS_MAX_BYTES = 20 * 1024 * 1024
ANTHROPIC_IMAGE_MAX_BYTES = 5 * 1024 * 1024
MEDIA_ANALYSIS_PROMPT_MAX_CHARS = 2000

MEDIA_ANALYSIS_PROMPTS = {
    AttachmentKind.IMAGE: (
        'You are an image analysis assistant. Describe this image objectively: '
        'main subjects, setting, visible text, and important details. '
        'Do not guess beyond what is visible. Keep the description within 200 words.'
    ),
    AttachmentKind.AUDIO: (
        'You are an audio understanding assistant. First transcribe any speech in this audio, '
        'keeping the original language. Then briefly note non-speech sounds (music, ambient noise) '
        'if clearly present. Be concise and factual; do not invent content.'
    ),
    AttachmentKind.VIDEO: (
        'You are a video analysis assistant. Describe what happens in this video: '
        'main subjects, actions, setting, on-screen text, and any audible speech if available. '
        'Be concise and factual; do not invent content.'
    ),
}

OPENAI_VIDEO_FRAME_FPS = 2.0
CHARACTER_REFERENCE_IMAGE_LIMIT = 4
OPENAI_LOCAL_TOOL_CALL_LIMIT = 6
MEMORY_TOOL_DEFAULT_MAX_CHARS = 6000
STREAM_MEMORY_SECTION_LIMIT = 900
LONG_TERM_MEMORY_DESC_LIMIT = 200
LONG_TERM_MEMORY_SECTION_LIMIT = 64
LONG_TERM_MEMORY_TOOL_ROUND_TRIP_LIMIT = 8


LOCAL_SEARCH_KEYWORDS = (
    'nearby',
    'local',
    'around me',
    'weather',
    'forecast',
    'temperature',
    'restaurant',
    'cafe',
    'coffee',
    'park',
    'museum',
    'bar',
    'store',
    'shop',
    '附近',
    '周边',
    '本地',
    '当地',
    '天气',
    '气温',
    '温度',
    '餐厅',
    '咖啡',
    '咖啡馆',
    '公园',
    '博物馆',
    '商店',
)

WEATHER_QUERY_KEYWORDS = (
    'weather',
    'forecast',
    'temperature',
    'rain',
    'snow',
    'sunny',
    'cloudy',
    'storm',
    'humidity',
    'weather like',
    '天气',
    '气温',
    '温度',
    '下雨',
    '下雪',
    '晴',
    '阴',
    '暴雨',
    '湿度',
)

TODAY_QUERY_KEYWORDS = (
    'today',
    'tonight',
    'this morning',
    'this afternoon',
    'this evening',
    '今天',
    '今晚',
    '今早',
    '今天早上',
    '今天下午',
    '今天晚上',
)

TOMORROW_QUERY_KEYWORDS = (
    'tomorrow',
    'tomorrow morning',
    'tomorrow night',
    '明天',
    '明早',
    '明天早上',
    '明晚',
    '明天晚上',
)

YESTERDAY_QUERY_KEYWORDS = (
    'yesterday',
    '昨晚',
    '昨天',
)


def _model_config_to_runtime(model_config):
    return {
        'provider': model_config.provider,
        'model_name': model_config.model_name,
        'api_key': model_config.api_key,
        'base_url': model_config.base_url,
    }


def _get_runtime_model_config(chat_session):
    model_config = ModelRoleAssignment.get_role_config(chat_session.user, ModelRole.TEXT)
    if not model_config:
        # 正常流程不会到这里（首个配置自动分配 text、PUT 禁止清空/跳过）；
        # 触发即数据状态异常（如 admin 批量删除绕过 destroy 保护），回退并留日志。
        model_config = ModelConfiguration.objects.filter(user=chat_session.user).order_by('id').first()
        if model_config:
            logger.warning(
                'User %s has model configs but no text role assignment; falling back to config %s',
                chat_session.user_id,
                model_config.id,
            )
    if not model_config:
        raise ValueError('No user model configuration is available for this chat session')

    return _model_config_to_runtime(model_config)


def _get_role_configs(user):
    """按角色返回 {role: runtime_config}，未分配的角色不出现。"""
    assignments = ModelRoleAssignment.get_role_configs(user)
    return {role: _model_config_to_runtime(config) for role, config in assignments.items()}


# 媒体路由（替代历史遗留的模型名能力正则）：
# - analyze: 对应角色槽位已配置 -> 槽位模型分析媒体，产出文本注入对话
# - native:  槽位为空但文本模型提供商原生支持该媒体 -> 直接发送
# - unsupported: 均不满足 -> 保留附件，提示用户配置槽位
MEDIA_KIND_ROLE = {
    AttachmentKind.IMAGE: ModelRole.IMAGE,
    AttachmentKind.AUDIO: ModelRole.AUDIO,
    AttachmentKind.VIDEO: ModelRole.VIDEO,
}

NATIVE_MEDIA_PROVIDERS = {
    'gemini': {AttachmentKind.IMAGE, AttachmentKind.AUDIO, AttachmentKind.VIDEO},
    'anthropic': {AttachmentKind.IMAGE},
}


def _route_media_kind(attachment_kind, role_configs, text_config):
    role = MEDIA_KIND_ROLE.get(attachment_kind)
    if role and role_configs.get(role):
        return 'analyze'
    if attachment_kind in NATIVE_MEDIA_PROVIDERS.get(text_config['provider'], set()):
        return 'native'
    return 'unsupported'


def _build_openai_endpoint(base_url):
    normalized_base_url = (base_url or 'https://api.openai.com/v1').rstrip('/')
    if normalized_base_url.endswith('/chat/completions'):
        return normalized_base_url
    return f"{normalized_base_url}/chat/completions"


def _supports_memory_tool_mode(runtime_config):
    return runtime_config['provider'] in {'openai_compatible', 'anthropic'}


def _build_memory_tool_specs():
    return [
        {
            'type': 'function',
            'function': {
                'name': 'list_memory_files',
                'description': (
                    'Browse the character memory filesystem. Use it like a folder explorer '
                    'before reading any specific long-term memory file.'
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'path_prefix': {
                            'type': 'string',
                            'description': 'Optional directory path such as schema, wiki, raw, raw/chat_sessions, or raw/character_setup.',
                        },
                        'recursive': {
                            'type': 'boolean',
                            'description': 'Set true to list all descendants under the selected path.',
                        },
                        'max_entries': {
                            'type': 'integer',
                            'description': 'Maximum number of entries to return, between 1 and 200.',
                        },
                    },
                    'required': [],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'read_memory_file',
                'description': (
                    'Read one file from the character memory filesystem after locating it. '
                    'Use the exact path returned by list_memory_files or shown in MEMORY FILESYSTEM.'
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'path': {
                            'type': 'string',
                            'description': 'Exact memory file path, for example schema/soul.md or raw/chat_sessions/session_12/transcript.md.',
                        },
                        'max_chars': {
                            'type': 'integer',
                            'description': 'Optional character limit between 200 and 12000.',
                        },
                    },
                    'required': ['path'],
                },
            },
        },
    ]


def _extract_text_from_content(content):
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
        return ''.join(text_parts).strip()

    return ''


def _extract_openai_content(response_json):
    choices = response_json.get('choices') or []
    if not choices:
        raise ValueError('OpenAI compatible API returned no choices')

    message = choices[0].get('message') or {}
    content = _extract_text_from_content(message.get('content'))
    if not content:
        raise ValueError('OpenAI compatible API returned an unsupported message format')
    return content


def _extract_openai_assistant_message(response_json):
    choices = response_json.get('choices') or []
    if not choices:
        raise ValueError('OpenAI compatible API returned no choices')
    return choices[0].get('message') or {}


def _should_retry_without_tools(exc):
    text = str(exc or '').lower()
    if not text:
        return False
    tool_markers = (
        'tool',
        'tools',
        'tool_choice',
        'tool_calls',
        'function calling',
        'function call',
    )
    retry_markers = (
        'unsupported',
        'unknown',
        'invalid',
        'not support',
        'unexpected',
        'extra inputs',
        'unrecognized',
    )
    return any(marker in text for marker in tool_markers) and any(marker in text for marker in retry_markers)


def _build_data_url(path, mime_type):
    with open(path, 'rb') as file_handle:
        encoded = base64.b64encode(file_handle.read()).decode('ascii')
    return f"data:{mime_type or 'application/octet-stream'};base64,{encoded}"


def _upload_generativeai_file(path, display_name, api_key):
    if not api_key:
        raise ValueError('API key is required for the selected model configuration')

    genai.configure(api_key=api_key)
    uploaded_file = genai.upload_file(path=path, display_name=display_name)
    state = getattr(getattr(uploaded_file, 'state', None), 'name', '') or ''

    deadline = time.time() + 120
    while state == 'PROCESSING' and time.time() < deadline:
        time.sleep(2)
        uploaded_file = genai.get_file(uploaded_file.name)
        state = getattr(getattr(uploaded_file, 'state', None), 'name', '') or ''

    if state and state not in {'ACTIVE', 'SUCCEEDED'}:
        raise ValueError(f"Uploaded media is not ready: {state}")

    return uploaded_file


def _get_or_upload_generativeai_file(cache_holder, path, display_name, api_key):
    """优先复用已上传的 Gemini Files 资源（48 小时有效），失效或未上传时重传。

    文件名缓存在 cache_holder.gemini_file_name（DB 字段；legacy 代理无 save
    时仅本轮内存生效）。get_file 是一次轻量元数据请求，远廉于重复上传+轮询。
    """
    cached_name = (getattr(cache_holder, 'gemini_file_name', '') or '').strip() if cache_holder else ''
    if cached_name:
        try:
            genai.configure(api_key=api_key)
            return genai.get_file(cached_name)
        except Exception as exc:  # noqa: BLE001
            logger.info('Cached Gemini file %s unavailable, re-uploading: %s', cached_name, exc)

    uploaded = _upload_generativeai_file(path, display_name, api_key)
    uploaded_name = getattr(uploaded, 'name', '')
    if cache_holder is not None and isinstance(uploaded_name, str) and uploaded_name:
        try:
            cache_holder.gemini_file_name = uploaded_name
            cache_holder.save(update_fields=['gemini_file_name', 'updated_at'])
        except Exception as exc:  # noqa: BLE001
            logger.warning('Failed to cache Gemini file name for %s: %s', path, exc)
    return uploaded


def _read_media_base64(path):
    with open(path, 'rb') as file_handle:
        return base64.b64encode(file_handle.read()).decode('ascii')


def _audio_format_from_name(path, mime_type):
    """input_audio 的 format 字段：优先用扩展名，退回从 mime 推断。"""
    extension = os.path.splitext(path)[1].lower().lstrip('.')
    if extension:
        return extension
    subtype = (mime_type or '').split('/')[-1].split(';')[0]
    return subtype or 'mp3'


def _request_openai_media_analysis(role_config, path, mime_type, attachment_kind, prompt):
    content_blocks = [{'type': 'text', 'text': prompt}]
    media_b64 = _read_media_base64(path)

    if attachment_kind == AttachmentKind.IMAGE:
        content_blocks.append({
            'type': 'image_url',
            'image_url': {'url': f"data:{mime_type or 'image/png'};base64,{media_b64}"},
        })
    elif attachment_kind == AttachmentKind.AUDIO:
        content_blocks.append({
            'type': 'input_audio',
            'input_audio': {
                'data': media_b64,
                'format': _audio_format_from_name(path, mime_type),
            },
        })
    elif attachment_kind == AttachmentKind.VIDEO:
        content_blocks.append({
            'type': 'video_url',
            'video_url': {'url': f"data:{mime_type or 'video/mp4'};base64,{media_b64}"},
            'fps': OPENAI_VIDEO_FRAME_FPS,
        })
    else:
        raise ValueError(f'Unsupported media kind for analysis: {attachment_kind}')

    headers = {'Content-Type': 'application/json'}
    if role_config['api_key']:
        headers['Authorization'] = f"Bearer {role_config['api_key']}"

    response = requests.post(
        _build_openai_endpoint(role_config.get('base_url', '')),
        headers=headers,
        json={
            'model': role_config['model_name'],
            'messages': [{'role': 'user', 'content': content_blocks}],
            'max_tokens': 1024,
        },
        timeout=120,
    )
    response.raise_for_status()
    return _extract_openai_content(response.json())


def _request_gemini_media_analysis(role_config, path, mime_type, attachment_kind, prompt, display_name='', cache_holder=None):
    if not role_config['api_key']:
        raise ValueError('API key is required for the selected model configuration')

    genai.configure(api_key=role_config['api_key'])
    model = genai.GenerativeModel(role_config['model_name'])

    if os.path.getsize(path) <= MEDIA_ANALYSIS_MAX_BYTES:
        media_part = {
            'mime_type': mime_type or 'application/octet-stream',
            'data': _read_media_base64(path),
        }
    else:
        media_part = _get_or_upload_generativeai_file(
            cache_holder,
            path,
            display_name,
            role_config['api_key'],
        )

    response = model.generate_content([prompt, media_part])
    text = (getattr(response, 'text', '') or '').strip()
    if not text:
        raise ValueError('Gemini returned an empty media analysis')
    return text


def _request_anthropic_media_analysis(role_config, path, mime_type, attachment_kind, prompt):
    if attachment_kind != AttachmentKind.IMAGE:
        raise ValueError('Anthropic only supports image media analysis')

    media_type = mime_type or 'image/png'
    if media_type not in {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}:
        raise ValueError(f'Anthropic does not accept image type: {media_type}')

    response = requests.post(
        f"{_build_anthropic_base_url(role_config.get('base_url', ''))}/v1/messages",
        headers={
            'Content-Type': 'application/json',
            'x-api-key': role_config['api_key'],
            'anthropic-version': ANTHROPIC_API_VERSION,
        },
        json={
            'model': role_config['model_name'],
            'max_tokens': ANTHROPIC_MEDIA_ANALYSIS_MAX_TOKENS,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': media_type,
                            'data': _read_media_base64(path),
                        },
                    },
                ],
            }],
        },
        timeout=120,
    )
    response.raise_for_status()
    return _extract_anthropic_text(response.json())


def _get_media_analysis_size_limit(provider):
    """槽位分析前的体积预检上限：

    - gemini：超过 inline 上限（20MB）会自动改走 Files API，用附件类型
      上限（视频 100MB）兜底；
    - anthropic：图片 base64 官方上限 5MB，超过必被 400 拒绝，提前跳过；
    - openai_compatible：base64 data URL / input_audio，维持 20MB。
    """
    if provider == 'gemini':
        return MAX_VIDEO_ATTACHMENT_BYTES
    if provider == 'anthropic':
        return ANTHROPIC_IMAGE_MAX_BYTES
    return MEDIA_ANALYSIS_MAX_BYTES


def _analyze_media_via_role(attachment, role_config):
    """用角色槽位模型分析媒体附件，结果缓存在 media_analysis 上避免重复调用。

    返回分析文本；失败返回 None（调用方降级为诚实提示）。
    """
    cached = (getattr(attachment, 'media_analysis', '') or '').strip()
    if cached:
        return cached

    attachment_kind = getattr(attachment, 'attachment_kind', '') or ''
    prompt = MEDIA_ANALYSIS_PROMPTS.get(attachment_kind)
    file_obj = getattr(attachment, 'file', None)
    if not prompt or not file_obj:
        return None

    path = file_obj.path
    mime_type = getattr(attachment, 'attachment_mime_type', '') or ''
    provider = role_config['provider']
    if os.path.getsize(path) > _get_media_analysis_size_limit(provider):
        logger.warning('Media analysis skipped, file too large: %s', path)
        return None

    try:
        if provider == 'gemini':
            analysis = _request_gemini_media_analysis(
                role_config, path, mime_type, attachment_kind, prompt,
                display_name=getattr(attachment, 'attachment_name', '') or os.path.basename(file_obj.name),
                cache_holder=attachment,
            )
        elif provider == 'openai_compatible':
            analysis = _request_openai_media_analysis(role_config, path, mime_type, attachment_kind, prompt)
        elif provider == 'anthropic':
            analysis = _request_anthropic_media_analysis(role_config, path, mime_type, attachment_kind, prompt)
        else:
            raise ValueError(f"Unsupported analysis provider: {provider}")
    except Exception as exc:  # noqa: BLE001
        logger.warning('Media analysis failed for %s: %s', path, exc)
        return None

    analysis = analysis.strip()[:MEDIA_ANALYSIS_PROMPT_MAX_CHARS]
    if not analysis:
        return None

    # 缓存失败（如 legacy 附件代理无 save）不应丢弃已成功的分析结果。
    attachment.media_analysis = analysis
    try:
        attachment.save(update_fields=['media_analysis', 'updated_at'])
    except Exception as exc:  # noqa: BLE001
        logger.warning('Failed to cache media analysis for %s: %s', path, exc)
    return analysis


MEDIA_LIMITATION_NOTES = {
    AttachmentKind.IMAGE: (
        "The current model cannot directly inspect images and no image model slot is configured. "
        "Acknowledge the limitation briefly, then ask the user to describe what matters in the image "
        "or configure an image model in AI settings."
    ),
    AttachmentKind.AUDIO: (
        "The current model cannot directly listen to audio and no audio model slot is configured. "
        "Acknowledge the limitation briefly, then ask the user to describe or transcribe the audio "
        "or configure an audio model in AI settings."
    ),
    AttachmentKind.VIDEO: (
        "The current model cannot directly inspect videos and no video model slot is configured. "
        "Acknowledge the limitation briefly, then ask the user for key frames, a summary, "
        "or configure a video model in AI settings."
    ),
}


def _build_attachment_prompt_text(message, role_configs, text_config, include_text_body=True, include_native_media_summary=True):
    parts = []

    for attachment in get_message_attachments(message):
        attachment_summary = describe_attachment_for_prompt(attachment, allow_text_body=include_text_body)
        if not attachment_summary:
            continue

        attachment_kind = getattr(attachment, 'attachment_kind', '') or ''
        if attachment_kind not in MEDIA_KIND_ROLE:
            parts.append(attachment_summary)
            continue

        route = _route_media_kind(attachment_kind, role_configs, text_config)
        if route == 'analyze':
            analysis = (getattr(attachment, 'media_analysis', '') or '').strip()
            if analysis:
                parts.append(f"{attachment_summary}\n[Media analysis by {attachment_kind} model]\n{analysis}")
            else:
                parts.append(
                    f"{attachment_summary}\n"
                    "The dedicated media model failed to analyze this attachment. "
                    "Do not invent its content; tell the user the analysis is unavailable right now."
                )
            continue

        if route == 'native':
            if include_native_media_summary:
                parts.append(f"{attachment_summary}\nAnalyze the attached media directly before replying.")
            continue

        parts.append(f"{attachment_summary}\n{MEDIA_LIMITATION_NOTES[attachment_kind]}")

    return '\n\n'.join(parts).strip()


def _build_message_text_content(message, role_configs, text_config, include_text_body=True, include_native_media_summary=True):
    parts = []
    content = (getattr(message, 'content', '') or '').strip()
    if content:
        parts.append(content)

    attachment_text = _build_attachment_prompt_text(
        message,
        role_configs=role_configs,
        text_config=text_config,
        include_text_body=include_text_body,
        include_native_media_summary=include_native_media_summary,
    )
    if attachment_text:
        parts.append(attachment_text)

    return '\n\n'.join(parts).strip()


def _get_native_media_attachments(message, role_configs, text_config):
    return [
        attachment
        for attachment in get_message_attachments(message)
        if _route_media_kind(
            getattr(attachment, 'attachment_kind', '') or '',
            role_configs,
            text_config,
        ) == 'native'
    ]


def _build_openai_compatible_multimodal_content(message, role_configs, text_config):
    content_blocks = []
    text_content = _build_message_text_content(
        message,
        role_configs=role_configs,
        text_config=text_config,
        include_text_body=False,
        include_native_media_summary=False,
    )
    if text_content:
        content_blocks.append({'type': 'text', 'text': text_content})

    for attachment in _get_native_media_attachments(message, role_configs, text_config):
        attachment_kind = getattr(attachment, 'attachment_kind', '') or ''
        file_obj = getattr(attachment, 'file', None)
        if not file_obj:
            continue

        if attachment_kind == AttachmentKind.IMAGE:
            content_blocks.append({
                'type': 'image_url',
                'image_url': {
                    'url': _build_data_url(file_obj.path, getattr(attachment, 'attachment_mime_type', '')),
                },
            })
        elif attachment_kind == AttachmentKind.AUDIO:
            content_blocks.append({
                'type': 'input_audio',
                'input_audio': {
                    'data': _read_media_base64(file_obj.path),
                    'format': _audio_format_from_name(
                        file_obj.path,
                        getattr(attachment, 'attachment_mime_type', ''),
                    ),
                },
            })
        elif attachment_kind == AttachmentKind.VIDEO:
            content_blocks.append({
                'type': 'video_url',
                'video_url': {
                    'url': _build_data_url(file_obj.path, getattr(attachment, 'attachment_mime_type', '')),
                },
                'fps': OPENAI_VIDEO_FRAME_FPS,
            })

    return content_blocks


def _get_character_reference_image_assets(character):
    return list(
        character.knowledge_assets.filter(attachment_kind=AttachmentKind.IMAGE)[:CHARACTER_REFERENCE_IMAGE_LIMIT]
    )


def _build_character_reference_message(character, runtime_config, role_configs, prompt_context, use_memory_tools=False):
    reference_sections = [] if use_memory_tools else [
        prompt_context.get("uploaded_index", ""),
        prompt_context.get("uploaded_background", ""),
        prompt_context.get("uploaded_visual_refs", ""),
    ]
    reference_text = "\n\n".join(
        section.strip()
        for section in reference_sections
        if section and section.strip()
    ).strip()

    image_assets = _get_character_reference_image_assets(character)
    if not image_assets:
        return None

    route = _route_media_kind(AttachmentKind.IMAGE, role_configs, runtime_config)

    if route == 'analyze':
        # 参考图经图片槽位 lazily 分析一次并缓存，文本模型只看描述文本。
        image_role_config = role_configs[MEDIA_KIND_ROLE[AttachmentKind.IMAGE]]
        descriptions = []
        for asset in image_assets:
            analysis = _analyze_media_via_role(asset, image_role_config)
            if analysis:
                descriptions.append(f"- {asset.attachment_name or os.path.basename(asset.file.name)}: {analysis}")
        if not descriptions:
            return None
        image_context = "[Character reference image analysis by image model]\n" + "\n".join(descriptions)
        return {
            'role': 'user',
            'content': '\n\n'.join(part for part in [reference_text, image_context] if part),
        }

    if route != 'native':
        return None

    if runtime_config['provider'] == 'gemini':
        parts = []
        if reference_text:
            parts.append(reference_text)
        for asset in image_assets:
            parts.append(
                _get_or_upload_generativeai_file(
                    asset,
                    asset.file.path,
                    asset.attachment_name or os.path.basename(asset.file.name),
                    runtime_config['api_key'],
                )
            )
        return {'role': 'user', 'parts': parts or ['']}

    if runtime_config['provider'] == 'openai_compatible':
        content = []
        if reference_text:
            content.append({'type': 'text', 'text': reference_text})
        for asset in image_assets:
            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': _build_data_url(asset.file.path, asset.attachment_mime_type),
                },
            })
        return {'role': 'user', 'content': content}

    if runtime_config['provider'] == 'anthropic':
        content = []
        if reference_text:
            content.append({'type': 'text', 'text': reference_text})
        skipped_assets = []
        for asset in image_assets:
            media_type = asset.attachment_mime_type or 'image/png'
            if media_type not in ANTHROPIC_SUPPORTED_IMAGE_MIME:
                skipped_assets.append(asset.attachment_name or os.path.basename(asset.file.name))
                continue
            content.append({
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': media_type,
                    'data': _read_media_base64(asset.file.path),
                },
            })
        if skipped_assets:
            content.append({
                'type': 'text',
                'text': f"[Reference images skipped (unsupported image type): {', '.join(skipped_assets)}]",
            })
        return {'role': 'user', 'content': content} if content else None

    return None


def _build_provider_message_entry(message, runtime_config, role_configs):
    role = 'assistant' if message.role == 'assistant' else 'user'
    provider = runtime_config['provider']
    native_media_attachments = _get_native_media_attachments(message, role_configs, runtime_config)

    if provider == 'gemini':
        parts = []
        text_content = _build_message_text_content(
            message,
            role_configs=role_configs,
            text_config=runtime_config,
            include_text_body=True,
            include_native_media_summary=False,
        )
        if text_content:
            parts.append(text_content)
        for attachment in native_media_attachments:
            parts.append(
                _get_or_upload_generativeai_file(
                    attachment,
                    attachment.file.path,
                    getattr(attachment, 'attachment_name', '') or os.path.basename(attachment.file.name),
                    runtime_config['api_key'],
                )
            )
        return {
            'role': 'model' if role == 'assistant' else 'user',
            'parts': parts or [''],
        }

    if provider in {'openai_compatible', 'anthropic'} and native_media_attachments:
        return {
            'role': role,
            'content': _build_openai_compatible_multimodal_content(message, role_configs, runtime_config),
        }

    return {
        'role': role,
        'content': _build_message_text_content(
            message,
            role_configs=role_configs,
            text_config=runtime_config,
            include_text_body=True,
        ),
    }


def _extract_token_usage(usage_payload):
    """Normalize an OpenAI-compatible ``usage`` dict into the token_usage
    shape stored on Message. Returns ``None`` when nothing usable is present.

    Cached-token field names differ per vendor: OpenAI/GLM put it in
    ``prompt_tokens_details.cached_tokens``, DeepSeek uses
    ``prompt_cache_hit_tokens``; accept the common variants.
    """
    if not isinstance(usage_payload, dict):
        return None

    def _int(value):
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    prompt_tokens = _int(usage_payload.get('prompt_tokens'))
    completion_tokens = _int(usage_payload.get('completion_tokens'))
    total_tokens = _int(usage_payload.get('total_tokens'))
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    cached_tokens = None
    details = usage_payload.get('prompt_tokens_details')
    if isinstance(details, dict):
        cached_tokens = _int(details.get('cached_tokens'))
    if cached_tokens is None:
        cached_tokens = _int(usage_payload.get('cached_tokens'))
    if cached_tokens is None:
        cached_tokens = _int(usage_payload.get('prompt_cache_hit_tokens'))

    return {
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens if total_tokens is not None else prompt_tokens + completion_tokens,
        'cached_tokens': min(cached_tokens or 0, prompt_tokens),
    }


def _merge_token_usage(accumulated, usage):
    """Sum usage dicts across multi-round tool loops (None starts a fresh total)."""
    if not usage:
        return accumulated
    if not accumulated:
        return dict(usage)
    return {
        key: accumulated.get(key, 0) + usage.get(key, 0)
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cached_tokens')
    }


def _request_openai_compatible_completion(
    *,
    model_name,
    api_key,
    messages,
    base_url,
    tools=None,
):
    # openai_compatible 允许本地反代网关自鉴权：仅有 key 时附加 Authorization header。
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    response = requests.post(
        _build_openai_endpoint(base_url),
        headers=headers,
        json={
            'model': model_name,
            'messages': messages,
            **({'tools': tools, 'tool_choice': 'auto'} if tools else {}),
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def _execute_local_memory_tool(filesystem, tool_name, raw_arguments):
    try:
        arguments = json.loads(raw_arguments or '{}')
    except json.JSONDecodeError:
        arguments = {}

    if tool_name == 'list_memory_files':
        return filesystem.list_memory_files(
            path_prefix=arguments.get('path_prefix', ''),
            recursive=bool(arguments.get('recursive', False)),
            max_entries=arguments.get('max_entries', 40),
        )

    if tool_name == 'read_memory_file':
        return filesystem.read_memory_file(
            path=arguments.get('path', ''),
            max_chars=arguments.get('max_chars', MEMORY_TOOL_DEFAULT_MAX_CHARS),
        )

    return {
        'error': f'Unknown local tool: {tool_name}',
    }


def _generate_openai_compatible_response(model_name, api_key, messages, base_url, tools=None, filesystem=None, event_sink=None):
    """Run the OpenAI-compatible tool loop and return the final text.

    When ``event_sink`` (a list) is provided, tool calls and the final round's
    native reasoning text are appended to it as ``{'type': 'tool' | 'thinking', ...}``
    dicts so callers can surface them to the user without extra LLM calls.
    """
    if not tools:
        response_json = _request_openai_compatible_completion(
            model_name=model_name,
            api_key=api_key,
            messages=messages,
            base_url=base_url,
        )
        if event_sink is not None:
            usage = _extract_token_usage(response_json.get('usage'))
            if usage:
                event_sink.append({'type': 'usage', 'usage': usage})
        return _extract_openai_content(response_json)

    usage_total = None
    history = list(messages)
    for _ in range(OPENAI_LOCAL_TOOL_CALL_LIMIT):
        try:
            response_json = _request_openai_compatible_completion(
                model_name=model_name,
                api_key=api_key,
                messages=history,
                base_url=base_url,
                tools=tools,
            )
        except Exception as exc:
            if _should_retry_without_tools(exc):
                logger.warning(
                    "OpenAI-compatible backend rejected local memory tools for model %s; retrying without tools.",
                    model_name,
                )
                if event_sink is not None:
                    del event_sink[:]
                response_json = _request_openai_compatible_completion(
                    model_name=model_name,
                    api_key=api_key,
                    messages=messages,
                    base_url=base_url,
                )
                if event_sink is not None:
                    usage = _extract_token_usage(response_json.get('usage'))
                    if usage:
                        event_sink.append({'type': 'usage', 'usage': usage})
                return _extract_openai_content(response_json)
            raise
        usage_total = _merge_token_usage(usage_total, _extract_token_usage(response_json.get('usage')))
        assistant_message = _extract_openai_assistant_message(response_json)
        tool_calls = assistant_message.get('tool_calls') or []
        history.append({
            'role': 'assistant',
            'content': assistant_message.get('content'),
            **({'tool_calls': tool_calls} if tool_calls else {}),
        })

        if not tool_calls:
            content = _extract_text_from_content(assistant_message.get('content'))
            if content:
                reasoning = assistant_message.get('reasoning_content') or assistant_message.get('reasoning')
                if reasoning and event_sink is not None:
                    event_sink.append({'type': 'thinking', 'content': reasoning})
                if event_sink is not None and usage_total:
                    event_sink.append({'type': 'usage', 'usage': dict(usage_total)})
                return content
            raise ValueError('OpenAI compatible API returned an empty response after tool execution')

        if filesystem is None:
            raise ValueError('Local tool execution requires a memory filesystem context')

        for tool_call in tool_calls:
            function_payload = tool_call.get('function') or {}
            tool_name = function_payload.get('name', '')
            raw_arguments = function_payload.get('arguments', '{}')
            try:
                tool_arguments = json.loads(raw_arguments or '{}')
            except json.JSONDecodeError:
                tool_arguments = {}
            if event_sink is not None:
                event_sink.append({'type': 'tool', 'tool': tool_name, 'arguments': tool_arguments})
            tool_result = _execute_local_memory_tool(
                filesystem,
                tool_name=tool_name,
                raw_arguments=raw_arguments,
            )
            history.append({
                'role': 'tool',
                'tool_call_id': tool_call.get('id', ''),
                'content': json.dumps(tool_result, ensure_ascii=False),
            })

    raise ValueError('OpenAI compatible API exceeded the local memory tool call limit')


def _stream_openai_compatible_response(model_name, api_key, messages, base_url):
    # openai_compatible 允许本地反代网关自鉴权：仅有 key 时附加 Authorization header。
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    def _open_stream(include_usage):
        payload = {
            'model': model_name,
            'messages': messages,
            'stream': True,
        }
        if include_usage:
            payload['stream_options'] = {'include_usage': True}
        return requests.post(
            _build_openai_endpoint(base_url),
            headers=headers,
            json=payload,
            stream=True,
            timeout=90,
        )

    response = _open_stream(include_usage=True)
    if response.status_code == 400:
        # Older gateways reject unknown body params; retry without the flag —
        # many vendors still attach usage to the final chunk by default.
        response.close()
        response = _open_stream(include_usage=False)
    try:
        response.raise_for_status()

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            line = raw_line.strip()
            if line.startswith('data:'):
                line = line[5:].strip()

            if line == '[DONE]':
                break

            data = json.loads(line)

            # The usage chunk (when requested via stream_options) carries an
            # empty choices array, so parse it before the choices guard below.
            usage = _extract_token_usage(data.get('usage'))
            if usage:
                yield {'type': 'usage', 'usage': usage}

            choices = data.get('choices') or []
            if not choices:
                continue

            delta = choices[0].get('delta') or {}

            # DeepSeek-style reasoning_content / OpenAI o-series reasoning.
            # Surfaced verbatim as a thinking event: no extra LLM call needed.
            reasoning = delta.get('reasoning_content') or delta.get('reasoning')
            if isinstance(reasoning, str) and reasoning:
                yield {'type': 'thinking', 'content': reasoning}
                continue

            content = delta.get('content')

            if isinstance(content, str) and content:
                yield {'type': 'delta', 'content': content}
                continue

            text = _extract_text_from_content(content)
            if text:
                yield {'type': 'delta', 'content': text}
    finally:
        response.close()


def _iter_buffered_chunks(text, chunk_size=160):
    """Re-stream buffered text as ``{'type': 'delta', ...}`` events so the
    buffered tools path matches the streaming providers' event contract."""
    normalized = (text or '').strip()
    if not normalized:
        return

    for start_index in range(0, len(normalized), chunk_size):
        yield {'type': 'delta', 'content': normalized[start_index:start_index + chunk_size]}


# ---------------------------------------------------------------------------
# Anthropic（Claude 官方直连）
# ---------------------------------------------------------------------------

ANTHROPIC_SUPPORTED_IMAGE_MIME = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}


def _build_anthropic_base_url(base_url):
    normalized = (base_url or ANTHROPIC_DEFAULT_BASE_URL).rstrip('/')
    if normalized.endswith('/v1'):
        normalized = normalized[:-3]
    return normalized


def _anthropic_headers(api_key):
    return {
        'Content-Type': 'application/json',
        'x-api-key': api_key,
        'anthropic-version': ANTHROPIC_API_VERSION,
    }


def _extract_anthropic_text(response_json):
    parts = []
    for block in response_json.get('content') or []:
        if block.get('type') == 'text':
            parts.append(block.get('text', ''))
    text = ''.join(parts).strip()
    if not text:
        raise ValueError('Anthropic API returned no text content')
    return text


def _convert_openai_content_blocks_to_anthropic(content):
    # tool_use / tool_result / image 是 Anthropic 原生块（工具循环回放或参考图
    # 直发），必须原样透传：丢弃 tool_use 会使配对的 tool_result 被 API 拒绝，
    # 丢弃 image 会让原生直发的参考图静默消失。
    if isinstance(content, str):
        return content if content else []
    if not isinstance(content, list):
        return [{'type': 'text', 'text': str(content)}]

    blocks = []
    for item in content:
        if not isinstance(item, dict):
            continue
        block_type = item.get('type')
        if block_type == 'text':
            text = (item.get('text') or '').strip()
            if text:
                blocks.append({'type': 'text', 'text': text})
        elif block_type in ('tool_use', 'tool_result', 'image'):
            blocks.append(item)
        elif block_type == 'image_url':
            data_url = (item.get('image_url') or {}).get('url', '')
            header, _, encoded = data_url.partition(',')
            media_type = header.partition(';')[0].removeprefix('data:')
            if media_type not in ANTHROPIC_SUPPORTED_IMAGE_MIME or not encoded:
                # 不静默丢图：给出占位说明，模型才能向用户解释图片缺席的原因
                blocks.append({
                    'type': 'text',
                    'text': f'[Attached image skipped: {media_type or "unknown"} images are not supported by this provider]',
                })
                continue
            blocks.append({
                'type': 'image',
                'source': {'type': 'base64', 'media_type': media_type, 'data': encoded},
            })

    return blocks


def _build_anthropic_request_messages(messages):
    system_parts = []
    request_messages = []

    for message in messages:
        role = message.get('role')
        content = message.get('content')
        if role == 'system':
            text = content if isinstance(content, str) else _extract_text_from_content(content)
            if text:
                system_parts.append(text)
            continue

        converted = _convert_openai_content_blocks_to_anthropic(content)
        if not converted:
            # 空 text 块会被 Anthropic 拒绝（"text content blocks must be
            # non-empty"），转换后为空的消息直接跳过。
            continue

        message_role = 'user' if role != 'assistant' else 'assistant'
        if request_messages and request_messages[-1]['role'] == message_role:
            # Anthropic 要求 user/assistant 交替：连续同角色消息（如角色参考
            # 消息紧跟首条用户消息）合并为一条多块消息。
            previous_blocks = request_messages[-1]['content']
            if isinstance(previous_blocks, str):
                request_messages[-1]['content'] = [{'type': 'text', 'text': previous_blocks}]
            if isinstance(converted, str):
                converted = [{'type': 'text', 'text': converted}]
            request_messages[-1]['content'].extend(converted)
        else:
            request_messages.append({'role': message_role, 'content': converted})

    if not request_messages:
        request_messages = [{'role': 'user', 'content': [{'type': 'text', 'text': ' '}]}]

    system = '\n\n'.join(part for part in system_parts if part)
    return (system or None), request_messages


def _convert_tools_to_anthropic(tools):
    converted = []
    for tool in tools or []:
        function_payload = tool.get('function') or {}
        converted.append({
            'name': function_payload.get('name', ''),
            'description': function_payload.get('description', ''),
            'input_schema': function_payload.get('parameters') or {'type': 'object', 'properties': {}},
        })
    return converted


def _request_anthropic_completion(*, model_name, api_key, messages, base_url, tools=None, max_tokens=None):
    if not api_key:
        raise ValueError('API key is required for the selected model configuration')

    system, request_messages = _build_anthropic_request_messages(messages)
    response = requests.post(
        f"{_build_anthropic_base_url(base_url)}/v1/messages",
        headers=_anthropic_headers(api_key),
        json={
            'model': model_name,
            'max_tokens': max_tokens or ANTHROPIC_COMPLETION_MAX_TOKENS,
            **({'system': system} if system else {}),
            'messages': request_messages,
            **({'tools': _convert_tools_to_anthropic(tools)} if tools else {}),
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def _generate_anthropic_response(model_name, api_key, messages, base_url, tools=None, filesystem=None, event_sink=None):
    """Run the Anthropic tool loop and return the final text.

    When ``event_sink`` (a list) is provided, tool calls and the final round's
    ``thinking`` blocks are appended as ``{'type': 'tool' | 'thinking', ...}``
    dicts for surfacing in the UI.
    """
    if not tools:
        return _extract_anthropic_text(
            _request_anthropic_completion(
                model_name=model_name,
                api_key=api_key,
                messages=messages,
                base_url=base_url,
            )
        )

    system, request_messages = _build_anthropic_request_messages(messages)
    anthropic_tools = _convert_tools_to_anthropic(tools)

    for _ in range(OPENAI_LOCAL_TOOL_CALL_LIMIT):
        response_json = _request_anthropic_completion(
            model_name=model_name,
            api_key=api_key,
            messages=([{'role': 'system', 'content': system}] if system else [])
            + [{'role': entry['role'], 'content': entry['content']} for entry in request_messages],
            base_url=base_url,
            tools=anthropic_tools,
        )
        content_blocks = response_json.get('content') or []
        tool_use_blocks = [block for block in content_blocks if block.get('type') == 'tool_use']

        request_messages = request_messages + [{'role': 'assistant', 'content': content_blocks}]

        if not tool_use_blocks:
            text = _extract_anthropic_text(response_json)
            if text:
                if event_sink is not None:
                    thinking_text = '\n\n'.join(
                        block.get('thinking', '')
                        for block in content_blocks
                        if block.get('type') == 'thinking' and block.get('thinking')
                    ).strip()
                    if thinking_text:
                        event_sink.append({'type': 'thinking', 'content': thinking_text})
                return text
            raise ValueError('Anthropic API returned an empty response after tool execution')

        if filesystem is None:
            raise ValueError('Local tool execution requires a memory filesystem context')

        tool_result_blocks = []
        for block in tool_use_blocks:
            tool_name = block.get('name', '')
            tool_arguments = block.get('input') or {}
            if event_sink is not None:
                event_sink.append({'type': 'tool', 'tool': tool_name, 'arguments': tool_arguments})
            tool_result = _execute_local_memory_tool(
                filesystem,
                tool_name=tool_name,
                raw_arguments=json.dumps(tool_arguments, ensure_ascii=False),
            )
            tool_result_blocks.append({
                'type': 'tool_result',
                'tool_use_id': block.get('id', ''),
                'content': json.dumps(tool_result, ensure_ascii=False),
            })
        request_messages = request_messages + [{'role': 'user', 'content': tool_result_blocks}]

    raise ValueError('Anthropic API exceeded the local memory tool call limit')


def _stream_anthropic_response(model_name, api_key, messages, base_url):
    if not api_key:
        raise ValueError('API key is required for the selected model configuration')

    system, request_messages = _build_anthropic_request_messages(messages)
    with requests.post(
        f"{_build_anthropic_base_url(base_url)}/v1/messages",
        headers=_anthropic_headers(api_key),
        json={
            'model': model_name,
            'max_tokens': ANTHROPIC_COMPLETION_MAX_TOKENS,
            'stream': True,
            **({'system': system} if system else {}),
            'messages': request_messages,
        },
        stream=True,
        timeout=90,
    ) as response:
        response.raise_for_status()

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            line = raw_line.strip()
            if not line.startswith('data:'):
                continue

            payload = line[5:].strip()
            if not payload or payload == '[DONE]':
                continue

            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if event.get('type') != 'content_block_delta':
                continue
            delta = event.get('delta') or {}
            if delta.get('type') == 'text_delta' and delta.get('text'):
                yield {'type': 'delta', 'content': delta['text']}
            elif delta.get('type') == 'thinking_delta' and delta.get('thinking'):
                # Anthropic extended thinking: streamed verbatim as a thinking event.
                yield {'type': 'thinking', 'content': delta['thinking']}


def _stream_gemini_response(model_name, api_key, prompt_or_messages, tools=None):
    if not api_key:
        raise ValueError('API key is required for the selected model configuration')

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name, tools=tools if tools else None)
    response = model.generate_content(prompt_or_messages, stream=True)

    for chunk in response:
        parts = getattr(chunk, 'parts', None)
        if parts:
            for part in parts:
                text = getattr(part, 'text', '') or ''
                if not text:
                    continue
                if getattr(part, 'thought', False):
                    # Gemini 2.5-style thinking parts: surfaced as a thinking event.
                    yield {'type': 'thinking', 'content': text}
                else:
                    yield {'type': 'delta', 'content': text}
            continue

        text = getattr(chunk, 'text', '') or ''
        if text:
            yield {'type': 'delta', 'content': text}


def _iter_text_chunks(runtime_config, prompt_or_messages, tools=None, filesystem=None):
    """Yield stream events as dicts: ``{'type': 'delta', 'content': ...}``,
    ``{'type': 'thinking', 'content': ...}`` or
    ``{'type': 'tool', 'tool': ..., 'arguments': ...}``.

    Tool calls force buffered generation (the loop must complete before the
    final text exists); tool/thinking events are emitted first, then the text
    is re-streamed in chunks so the UI still feels live.
    """
    provider = runtime_config['provider']
    model_name = runtime_config['model_name']
    api_key = runtime_config['api_key']

    if provider == 'gemini':
        yield from _stream_gemini_response(model_name, api_key, prompt_or_messages, tools=tools)
        return

    if provider == 'openai_compatible':
        if not isinstance(prompt_or_messages, list):
            prompt_or_messages = [{'role': 'user', 'content': str(prompt_or_messages)}]
        if tools:
            event_sink: list[dict] = []
            buffered_text = _generate_openai_compatible_response(
                model_name=model_name,
                api_key=api_key,
                messages=prompt_or_messages,
                base_url=runtime_config.get('base_url', ''),
                tools=tools,
                filesystem=filesystem,
                event_sink=event_sink,
            )
            yield from event_sink
            yield from _iter_buffered_chunks(buffered_text)
            return
        yield from _stream_openai_compatible_response(
            model_name=model_name,
            api_key=api_key,
            messages=prompt_or_messages,
            base_url=runtime_config.get('base_url', ''),
        )
        return

    if provider == 'anthropic':
        if not isinstance(prompt_or_messages, list):
            prompt_or_messages = [{'role': 'user', 'content': str(prompt_or_messages)}]
        if tools:
            event_sink = []
            buffered_text = _generate_anthropic_response(
                model_name=model_name,
                api_key=api_key,
                messages=prompt_or_messages,
                base_url=runtime_config.get('base_url', ''),
                tools=tools,
                filesystem=filesystem,
                event_sink=event_sink,
            )
            yield from event_sink
            yield from _iter_buffered_chunks(buffered_text)
            return
        yield from _stream_anthropic_response(
            model_name=model_name,
            api_key=api_key,
            messages=prompt_or_messages,
            base_url=runtime_config.get('base_url', ''),
        )
        return

    raise ValueError(f"Unsupported model provider: {provider}")


def _generate_text(runtime_config, prompt_or_messages, tools=None, filesystem=None):
    provider = runtime_config['provider']
    model_name = runtime_config['model_name']
    api_key = runtime_config['api_key']

    if provider == 'gemini':
        if not api_key:
            raise ValueError('API key is required for the selected model configuration')

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name, tools=tools if tools else None)
        response = model.generate_content(prompt_or_messages)
        return response.text.strip()

    if provider == 'openai_compatible':
        if not isinstance(prompt_or_messages, list):
            prompt_or_messages = [{'role': 'user', 'content': str(prompt_or_messages)}]
        return _generate_openai_compatible_response(
            model_name=model_name,
            api_key=api_key,
            messages=prompt_or_messages,
            base_url=runtime_config.get('base_url', ''),
            tools=tools,
            filesystem=filesystem,
        )

    if provider == 'anthropic':
        if not isinstance(prompt_or_messages, list):
            prompt_or_messages = [{'role': 'user', 'content': str(prompt_or_messages)}]
        return _generate_anthropic_response(
            model_name=model_name,
            api_key=api_key,
            messages=prompt_or_messages,
            base_url=runtime_config.get('base_url', ''),
            tools=tools,
            filesystem=filesystem,
        )

    raise ValueError(f"Unsupported model provider: {provider}")


def _append_section(sections, title, content):
    normalized = (content or '').strip()
    if normalized:
        sections.append(f"[{title}]\n{normalized}")


def _truncate_text(value, max_length):
    normalized = (value or '').strip()
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + '...'


def _try_parse_json_object(text):
    """Parse text as JSON, returning a dict on success, None otherwise.
    """
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _slice_balanced_json_object(text, start_index):
    """Return the substring from start_index to the matching closing '}'.

    Tracks nested braces and string literals (with backslash escapes) so
    that braces appearing inside JSON strings do not unbalance the scan.
    Returns None if no balanced match is found.
    """
    depth = 0
    in_string = False
    escape = False
    for end_index in range(start_index, len(text)):
        char = text[end_index]
        if escape:
            escape = False
            continue
        if char == '\\' and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start_index:end_index + 1]
    return None


def _extract_json_object(raw_text):
    """Best-effort extraction of a top-level JSON object from a model response.

    Handles: raw JSON, ```` ```json ... ``` ```` and ```` ``` ... ``` ```` fences,
    JSON embedded in surrounding prose, and nested objects. Returns an empty
    dict if no valid JSON object can be recovered.
    """
    text = (raw_text or '').strip()
    if not text:
        return {}

    # 1) Direct parse
    parsed = _try_parse_json_object(text)
    if parsed is not None:
        return parsed

    # 2) Strip a leading/trailing markdown code fence and retry
    if text.startswith('```'):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        stripped = '\n'.join(lines).strip()
        if stripped and stripped != text:
            parsed = _try_parse_json_object(stripped)
            if parsed is not None:
                return parsed
            text = stripped

    # 3) Walk the text looking for balanced {...} regions
    start = text.find('{')
    while start != -1:
        candidate = _slice_balanced_json_object(text, start)
        if candidate is not None:
            parsed = _try_parse_json_object(candidate)
            if parsed is not None:
                return parsed
        start = text.find('{', start + 1)

    # Dump the full raw response to a file so we can see what the model
    # actually sent (the 300-char preview in the log is often not enough
    # to diagnose exotic responses such as a JSON object that ends mid-
    # string, or extra prose wrapping a truncated object). The file lives
    # in the OS temp directory (tempfile.gettempdir()) which is
    # git-ignored; do not commit it. Note: it may echo content from the
    # user's uploaded source files, so treat it as potentially containing
    # PII.
    DUMP_MAX_BYTES = 1_000_000  # cap at 1 MB so a chatty model can't fill the disk
    try:
        dump_path = os.path.join(
            tempfile.gettempdir(),
            f'ai_draft_raw_{int(time.time() * 1_000_000)}.txt',
        )
        raw_for_dump = raw_text or ''
        truncated_for_dump = len(raw_for_dump) > DUMP_MAX_BYTES
        with open(dump_path, 'w', encoding='utf-8', errors='replace') as _dump_file:
            if truncated_for_dump:
                _dump_file.write(raw_for_dump[:DUMP_MAX_BYTES])
                _dump_file.write('\n\n... [truncated for size; full response was '
                                 f'{len(raw_for_dump)} chars] ...\n')
            else:
                _dump_file.write(raw_for_dump)
        logger.info(
            "Failed to extract JSON object from model response (length=%d). "
            "Full raw response written to %s. Preview: %r",
            len(raw_for_dump),
            dump_path,
            raw_for_dump[:300],
        )
    except Exception as _dump_exc:
        logger.info(
            "Failed to extract JSON object from model response (length=%d). "
            "Preview: %r (raw-dump error: %s)",
            len(raw_text or ''),
            (raw_text or '')[:300],
            _dump_exc,
        )
    return {}


def _is_legacy_bootstrap_message(message):
    if message.role != 'user':
        return False

    content = (message.content or '').strip()
    return content.startswith('=== CHARACTER IDENTITY ===') and 'Please provide your initial greeting based on your character settings.' in content


def _build_user_turn_summary(message, include_text_body=False):
    summary = _build_message_text_content(
        message,
        role_configs={},
        text_config={'provider': 'openai_compatible'},
        include_text_body=include_text_body,
    )
    return summary or '[User sent an attachment]'


def _get_visible_history_messages(chat_session):
    history_messages = Message.objects.filter(chat_session=chat_session).order_by('timestamp')
    return [message for message in history_messages if not _is_legacy_bootstrap_message(message)]


def _get_user_profile(chat_session):
    return UserProfile.get_or_create_for_user(chat_session.user)


def _get_user_local_time(profile):
    if not profile.share_local_time or not profile.timezone:
        return ""

    try:
        return datetime.now(ZoneInfo(profile.timezone)).strftime('%Y-%m-%d %H:%M %Z')
    except Exception:
        return ""


def _get_user_local_datetime(profile):
    if not profile.share_local_time or not profile.timezone:
        return None

    try:
        return datetime.now(ZoneInfo(profile.timezone))
    except Exception:
        return None


def _contains_any_keyword(text, keywords):
    return any(keyword in text for keyword in keywords)


def _describe_daypart(hour):
    if 5 <= hour < 12:
        return 'morning'
    if 12 <= hour < 18:
        return 'afternoon'
    if 18 <= hour < 22:
        return 'evening'
    return 'night'


def _get_query_local_date_reference(profile, lowered_query):
    local_now = _get_user_local_datetime(profile)
    if not local_now:
        return ''

    if _contains_any_keyword(lowered_query, TOMORROW_QUERY_KEYWORDS):
        return (local_now + timedelta(days=1)).date().isoformat()
    if _contains_any_keyword(lowered_query, YESTERDAY_QUERY_KEYWORDS):
        return (local_now - timedelta(days=1)).date().isoformat()
    if (
        _contains_any_keyword(lowered_query, TODAY_QUERY_KEYWORDS)
        or _contains_any_keyword(lowered_query, WEATHER_QUERY_KEYWORDS)
    ):
        return local_now.date().isoformat()
    return ''


def _normalize_prompt_memory_section(title, content):
    text = (content or '').strip()
    if not text:
        return ''
    return f"# {title}\n{text}"


def _extract_prompt_memory_body(section):
    text = (section or '').strip()
    if not text:
        return ''

    parts = text.split('\n', 1)
    if len(parts) == 2 and parts[0].startswith('# '):
        return parts[1].strip()
    return text


def _build_account_runtime_sections(chat_session):
    profile = _get_user_profile(chat_session)

    context_lines = []
    local_now = _get_user_local_datetime(profile)
    if local_now:
        context_lines.append(f"User Local Time: {local_now.strftime('%Y-%m-%d %H:%M %Z')}")
        context_lines.append(f"User Local Daypart: {_describe_daypart(local_now.hour)}")
        context_lines.append(
            "Interpret relative time words such as today, tonight, and tomorrow in the user's local timezone."
        )
    if profile.share_location and profile.location_label:
        context_lines.append(
            f"Location Hint ({profile.get_location_precision_display()} level): {profile.location_label}"
        )
        context_lines.append("Do not imply a more precise real-world location than the user explicitly shared.")
    if profile.share_weather and profile.share_location and profile.location_label:
        context_lines.append(
            "Weather Context: If weather comes up, ground it in the shared location hint. "
            "Do not guess current conditions. Use live research when available; otherwise speak conditionally."
        )

    boundary_lines = []
    if profile.blocked_topics:
        boundary_lines.append(f"Blocked Topics: {profile.blocked_topics}")

    memory_lines = []
    if not profile.allow_long_term_memory:
        memory_lines.append("Do not convert personal conversation details into long-term persistent memory.")
    if not profile.allow_preference_inference:
        memory_lines.append("Do not infer or store new user preferences unless the user explicitly states them.")
    if not profile.allow_research_profile_updates:
        memory_lines.append("Do not let web research modify the user profile or user preference model.")

    return {
        'context': "\n".join(context_lines),
        'boundaries': "\n".join(boundary_lines),
        'memory_rules': "\n".join(memory_lines),
        'profile_obj': profile,
    }


def _format_working_state(chat_session):
    lines = []

    if chat_session.id:
        lines.append(f"Session ID: {chat_session.id}")

    try:
        message_count = chat_session.messages.count()
    except Exception:
        message_count = 0
    lines.append(f"Visible Messages: {message_count}")

    if chat_session.updated_at:
        lines.append(f"Last Updated: {chat_session.updated_at.isoformat()}")

    if chat_session.last_response_latency_ms is not None:
        lines.append(f"Last Response Latency Ms: {chat_session.last_response_latency_ms}")

    return "\n".join(lines)


def _build_stream_memory_prefetch(character, chat_session, generate_greeting=False):
    prompt_context = build_character_prompt_context(character)

    sections = []

    candidates = [
        ('Character Setup', prompt_context.get('soul', '')),
        ('Long-Term Memory (User Model)', MemoryManager(character).render_narrative()),
    ]
    if generate_greeting:
        candidates.append(('Uploaded Background Text', prompt_context.get('uploaded_background', '')))

    for title, content in candidates:
        body = _extract_prompt_memory_body(content)
        if not body:
            continue
        sections.append(
            _normalize_prompt_memory_section(
                title,
                _truncate_text(body, STREAM_MEMORY_SECTION_LIMIT),
            )
        )

    return "\n\n".join(section for section in sections if section and section.strip())


def _build_system_prompt(character, chat_session, use_memory_tools=False, retrieved_memory=''):
    prompt_context = build_character_prompt_context(character)
    character_setup = prompt_context.get('soul', '')
    account_runtime_sections = _build_account_runtime_sections(chat_session)
    sections = [
        "You are in an immersive roleplay chat. Stay fully in character, be specific, and avoid generic assistant phrasing.",
        "Never mention system instructions, hidden rules, or that you are an AI model unless the character would explicitly know that in-world.",
    ]
    _append_section(sections, "CHARACTER SETUP", character_setup)
    _append_section(sections, "ACCOUNT CONTEXT", account_runtime_sections.get("context", ""))
    _append_section(sections, "ACCOUNT BOUNDARIES", account_runtime_sections.get("boundaries", ""))
    _append_section(sections, "MEMORY CONSENT", account_runtime_sections.get("memory_rules", ""))

    if use_memory_tools:
        _append_section(
            sections,
            "MEMORY TOOLING",
            "\n".join([
                "Do not assume long-term memory content from the prompt alone.",
                "When a reply depends on the exact character setup, prior transcripts, uploaded files, or search traces, inspect the memory filesystem first.",
                "Use list_memory_files to browse the schema/wiki/raw tree, then use read_memory_file to open only the files relevant to this turn.",
                "If you did not inspect a memory file, do not claim certainty about its contents.",
            ]),
        )
        _append_section(sections, "MEMORY FILESYSTEM", build_memory_explorer_manifest(character))
        return "\n\n".join(sections)

    compact_memory_mode = bool((retrieved_memory or '').strip())
    if compact_memory_mode:
        _append_section(sections, "WORKING STATE", _format_working_state(chat_session))
        _append_section(sections, "RETRIEVED MEMORY", retrieved_memory)
        return "\n\n".join(sections)

    uploaded_sections = "\n\n".join(
        section
        for section in [
            prompt_context.get("uploaded_index", ""),
            prompt_context.get("uploaded_background", ""),
            prompt_context.get("uploaded_visual_refs", ""),
        ]
        if section and section.strip()
    )
    _append_section(sections, "USER UPLOADS", uploaded_sections)
    _append_section(sections, "WORKING STATE", _format_working_state(chat_session))

    return "\n\n".join(sections)


def _build_search_query(chat_session, user_message=None):
    profile = _get_user_profile(chat_session)
    candidate_parts = []

    if user_message:
        candidate_parts.append(_build_user_turn_summary(user_message, include_text_body=False))

    recent_user_messages = list(
        Message.objects.filter(chat_session=chat_session, role='user').order_by('-timestamp')[:2]
    )
    for message in reversed(recent_user_messages):
        if not user_message or message.id != user_message.id:
            candidate_parts.append(_build_user_turn_summary(message, include_text_body=False))

    query = " ".join(part.strip() for part in candidate_parts if part and part.strip())
    lowered_query = query.lower()
    if (
        profile.share_location
        and profile.location_label
        and _contains_any_keyword(lowered_query, LOCAL_SEARCH_KEYWORDS)
        and profile.location_label.lower() not in lowered_query
    ):
        query = f"{query} in {profile.location_label}"

    local_date_reference = _get_query_local_date_reference(profile, lowered_query)
    if local_date_reference and local_date_reference not in query:
        query = f"{query} {local_date_reference}"

    return _truncate_text(query, 300)


def build_research_context(chat_session, user_message=None):
    profile = _get_user_profile(chat_session)
    if not profile.default_enable_web_search:
        return {
            'query': '',
            'items': [],
            'provider': '',
            'error': '',
        }

    query = _build_search_query(chat_session, user_message=user_message)
    if not query:
        return {
            'query': '',
            'items': [],
            'provider': '',
            'error': '',
        }

    return search_web(query, chat_session=chat_session)


def _format_research_context(research_context):
    items = research_context.get('items') or []
    if not items:
        error = research_context.get('error', '')
        if error:
            return f"Web search requested but unavailable: {error}"
        return ""

    lines = [f"Search Query: {research_context.get('query', '')}"]
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. {item.get('title', 'Untitled')}")
        lines.append(f"URL: {item.get('url', '')}")
        if item.get('snippet'):
            lines.append(f"Snippet: {item.get('snippet')}")
    return "\n".join(lines)


def _get_tools(chat_session, runtime_config, allow_memory_tools=True):
    if allow_memory_tools and _supports_memory_tool_mode(runtime_config):
        return _build_memory_tool_specs()

    return []


def _analyze_latest_user_media(visible_history, role_configs, text_config):
    """发送前对最新一条用户消息的媒体附件跑槽位分析（结果缓存在附件上）。

    历史消息只读缓存，不重新调用，避免每轮重复付费。
    """
    latest_user_message = next(
        (
            item
            for item in reversed(visible_history)
            if isinstance(item, Message) and item.role == 'user'
        ),
        None,
    )
    if latest_user_message is None:
        return

    for attachment in get_message_attachments(latest_user_message):
        attachment_kind = getattr(attachment, 'attachment_kind', '') or ''
        role = MEDIA_KIND_ROLE.get(attachment_kind)
        if not role or _route_media_kind(attachment_kind, role_configs, text_config) != 'analyze':
            continue
        _analyze_media_via_role(attachment, role_configs[role])


def _build_provider_messages(
    chat_session,
    character,
    generate_greeting=False,
    research_context=None,
    allow_memory_tools=True,
    retrieved_memory='',
):
    runtime_config = _get_runtime_model_config(chat_session)
    role_configs = _get_role_configs(chat_session.user)
    use_memory_tools = allow_memory_tools and _supports_memory_tool_mode(runtime_config)
    tools = _get_tools(chat_session, runtime_config, allow_memory_tools=allow_memory_tools)
    system_prompt = _build_system_prompt(
        character,
        chat_session,
        use_memory_tools=use_memory_tools,
        retrieved_memory=retrieved_memory,
    )
    prompt_context = build_character_prompt_context(character)
    if research_context:
        formatted_research = _format_research_context(research_context)
        if formatted_research:
            system_prompt = f"{system_prompt}\n\n[LIVE WEB RESEARCH]\n{formatted_research}"
    visible_history = _get_visible_history_messages(chat_session)
    _analyze_latest_user_media(visible_history, role_configs, runtime_config)
    character_reference_message = _build_character_reference_message(
        character,
        runtime_config,
        role_configs,
        prompt_context,
        use_memory_tools=use_memory_tools,
    )

    if generate_greeting:
        visible_history.append({
            'role': 'user',
            'content': (
                "Start the conversation now. Send the first in-character message proactively, "
                "grounded in the scenario and relationship context. Do not wait for the user to speak first."
            ),
        })

    if runtime_config['provider'] == 'gemini':
        formatted_history = [{'role': 'user', 'parts': [system_prompt]}]
        if character_reference_message:
            formatted_history.append(character_reference_message)
        for message in visible_history:
            if isinstance(message, Message):
                formatted_history.append(_build_provider_message_entry(message, runtime_config, role_configs))
                continue
            formatted_history.append({'role': 'user', 'parts': [message['content']]})
        return runtime_config, formatted_history, tools

    formatted_history = [{'role': 'system', 'content': system_prompt}]
    if character_reference_message:
        formatted_history.append(character_reference_message)
    for message in visible_history:
        if isinstance(message, Message):
            formatted_history.append(_build_provider_message_entry(message, runtime_config, role_configs))
            continue
        formatted_history.append({'role': 'user', 'content': message['content']})
    return runtime_config, formatted_history, tools


def _prepare_generation(chat_session, character, generate_greeting=False, research_context=None):
    """Shared generation configuration for the streaming and non-streaming paths.

    The memory strategy is decided exactly once here and used by both paths so
    they never drift apart: providers that support local tools get the memory
    tooling mode, everyone else falls back to a compact prefetch injected into
    the prompt (``_build_stream_memory_prefetch`` is used by both paths even
    though its name predates that). Returns ``(runtime_config,
    formatted_history, tools)``.
    """
    runtime_config = _get_runtime_model_config(chat_session)
    use_memory_tools = _supports_memory_tool_mode(runtime_config)
    retrieved_memory = ''
    if not use_memory_tools:
        retrieved_memory = _build_stream_memory_prefetch(
            character,
            chat_session,
            generate_greeting=generate_greeting,
        )
    runtime_config, formatted_history, tools = _build_provider_messages(
        chat_session=chat_session,
        character=character,
        generate_greeting=generate_greeting,
        research_context=research_context,
        allow_memory_tools=use_memory_tools,
        retrieved_memory=retrieved_memory,
    )
    return runtime_config, formatted_history, tools


# ---------------------------------------------------------------------------
# Long-term memory pipeline (per-turn, SonettoHere parity)
# ---------------------------------------------------------------------------
def _publish_memory_event(chat_session_id, action):
    """Best-effort Redis pub/sub for SSE consumers; failures are logged but
    never fatal.
    """
    try:
        from django.conf import settings
        import redis

        client = redis.Redis.from_url(settings.CELERY_BROKER_URL or 'redis://localhost:6379/0')
        client.publish(
            f'chat:memory_updates:{chat_session_id}',
            json.dumps(action, ensure_ascii=False, default=str),
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            'Failed to publish memory event for session %s: %s', chat_session_id, exc,
        )


def _execute_memory_crud_tool(character, manager, source_message, tool_name, raw_args):
    """Local tool implementation invoked by the Celery worker ReAct loop."""
    try:
        arguments = json.loads(raw_args or '{}')
    except json.JSONDecodeError:
        arguments = {}

    if tool_name == 'create_memory':
        item = manager.create_item(
            section=arguments.get('section', ''),
            description=arguments.get('description', ''),
            reason=arguments.get('reason', ''),
            source_message=source_message,
        )
        return {
            'status': 'created',
            'short_id': item.short_id,
            'section': item.section,
            'description': item.description,
        }
    if tool_name == 'read_memories':
        section = (arguments.get('section') or '').strip()
        items = manager.list_items()
        if section:
            items = [item for item in items if item.section == section]
        return {
            'items': [
                {
                    'short_id': item.short_id,
                    'section': item.section,
                    'description': item.description,
                }
                for item in items
            ],
        }
    if tool_name == 'update_memory':
        item = manager.update_item(
            short_id=arguments.get('id', ''),
            description=arguments.get('description', ''),
            section=arguments.get('section'),
            reason=arguments.get('reason', ''),
            source_message=source_message,
        )
        return {'status': 'updated', 'short_id': item.short_id, 'description': item.description}
    if tool_name == 'delete_memory':
        removed = manager.delete_item(
            short_id=arguments.get('id', ''),
            reason=arguments.get('reason', ''),
            source_message=source_message,
        )
        return {'status': 'deleted', 'description': removed}
    if tool_name == 'merge_memories':
        item = manager.merge_items(
            id1=arguments.get('id1', ''),
            id2=arguments.get('id2', ''),
            content=arguments.get('content', ''),
            section=arguments.get('section', ''),
            reason=arguments.get('reason', ''),
            source_message=source_message,
        )
        return {'status': 'merged', 'kept_short_id': item.short_id, 'description': item.description}
    return {'error': f'Unknown memory tool: {tool_name}'}


def _collect_memory_actions(
    runtime_config,
    prompt,
    character,
    manager,
    source_message,
    tool_specs,
):
    """Drive the ReAct loop for the per-turn memory extraction call.

    Returns a list of action dicts suitable for SSE publish / audit logging.
    """
    provider = runtime_config['provider']
    messages = [
        {'role': 'system', 'content': prompt['system']},
        {'role': 'user', 'content': prompt['user']},
    ]

    actions: list[dict] = []
    if provider == 'openai_compatible':
        for _ in range(LONG_TERM_MEMORY_TOOL_ROUND_TRIP_LIMIT):
            try:
                response_json = _request_openai_compatible_completion(
                    model_name=runtime_config['model_name'],
                    api_key=runtime_config['api_key'],
                    messages=messages,
                    base_url=runtime_config.get('base_url', ''),
                    tools=tool_specs,
                )
            except Exception as exc:  # noqa: BLE001
                if _should_retry_without_tools(exc):
                    logger.warning(
                        'OpenAI-compatible backend rejected memory tools for %s; retrying without.',
                        runtime_config['model_name'],
                    )
                    response_json = _request_openai_compatible_completion(
                        model_name=runtime_config['model_name'],
                        api_key=runtime_config['api_key'],
                        messages=messages,
                        base_url=runtime_config.get('base_url', ''),
                    )
                else:
                    raise
            assistant_message = _extract_openai_assistant_message(response_json)
            tool_calls = assistant_message.get('tool_calls') or []
            messages.append({
                'role': 'assistant',
                'content': assistant_message.get('content'),
                **({'tool_calls': tool_calls} if tool_calls else {}),
            })
            if not tool_calls:
                break
            for tool_call in tool_calls:
                function_payload = tool_call.get('function') or {}
                tool_name = function_payload.get('name', '')
                result = _execute_memory_crud_tool(
                    character, manager, source_message,
                    tool_name, function_payload.get('arguments', '{}'),
                )
                actions.append({'tool': tool_name, 'result': result, 'short_id': (result.get('short_id') or result.get('kept_short_id') or '')})
                messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.get('id', ''),
                    'content': json.dumps(result, ensure_ascii=False),
                })
        return actions

    if provider == 'gemini':
        # Gemini path: ask the model for plain text describing what to do, parse
        # out JSON action triples, dispatch through the same local tool layer.
        try:
            if not runtime_config['api_key']:
                raise ValueError('API key is required for the selected model configuration')
            import google.generativeai as genai

            genai.configure(api_key=runtime_config['api_key'])
            response = genai.GenerativeModel(runtime_config['model_name']).generate_content(
                f"{prompt['system']}\n\n{prompt['user']}\n\n"
                'Respond with a JSON array of tool calls, e.g. '
                '[{"tool": "create_memory", "args": {"section": "身份", "description": "..."}}]. '
                'Allowed tools: create_memory / read_memories / update_memory / delete_memory / merge_memories.'
            )
            raw_text = (getattr(response, 'text', '') or '').strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning('Gemini memory call failed: %s', exc)
            return actions

        # We expect an array; if we got an object, wrap.
        try:
            data = _extract_json_object(raw_text)
        except (ValueError, TypeError):
            data = {}
        candidates = data.get('actions') if isinstance(data, dict) and data.get('actions') else None
        if candidates is None and isinstance(data, dict):
            # fall back: pluck the first list value we find
            for value in data.values():
                if isinstance(value, list):
                    candidates = value
                    break
        if not candidates:
            # As a last resort, look for the first JSON array in raw_text.
            match = re.search(r'\[[^\]]*\]', raw_text, re.DOTALL)
            if match:
                try:
                    candidates = json.loads(match.group(0))
                except json.JSONDecodeError:
                    candidates = []
        for entry in candidates or []:
            if not isinstance(entry, dict):
                continue
            tool_name = entry.get('tool') or entry.get('name') or ''
            args = entry.get('args') or entry.get('arguments') or {}
            result = _execute_memory_crud_tool(
                character, manager, source_message, tool_name, json.dumps(args, ensure_ascii=False),
            )
            actions.append({'tool': tool_name, 'result': result, 'short_id': (result.get('short_id') or result.get('kept_short_id') or '')})
        return actions

    raise ValueError(f'Unsupported model provider for memory sync: {provider}')


@shared_task(
    bind=True,
    autoretry_for=(requests.RequestException,),
    retry_backoff=True,
    retry_kwargs={'max_retries': 2},
    rate_limit='30/m',
)
def sync_long_term_memory(self, message_id, chat_session_id, character_id):
    """Per-turn long-term-memory sync (SonettoHere parity).

    Mirrors ``LongTermMemoryInterface.send_history`` + the consumer
    coroutine from SonettoHere, but driven by Celery instead of an
    asyncio queue. The DB write happens here; the chat view never waits.

    Lock scope: we deliberately do **not** wrap the heading read in a
    ``transaction.atomic`` + ``select_for_update``. The per-tool
    ``MemoryManager`` CRUD methods each open their own short atomic block,
    so no character row lock is held across the multi-round ReAct turn.
    A long Gemini call (30s+) therefore no longer blocks other writers.

    Concurrency caveat for v1: two near-simultaneous turns for the same
    character can race on the same ``short_id`` because each per-tool
    ``transaction.atomic`` only protects a single write. Acceptable for
    the typical single-session flow; serialize per character with a
    Celery header or Redis flag in v2.
    """
    try:
        character = Character.objects.get(pk=character_id)
        chat_session = ChatSession.objects.get(pk=chat_session_id)
        message = Message.objects.select_related('chat_session').get(pk=message_id)

        profile = UserProfile.get_or_create_for_user(character.created_by)
        if not profile.allow_long_term_memory:
            logger.info('Long-term memory disabled for user %s; skipping.', character.created_by_id)
            return {'status': 'skipped', 'reason': 'user_disabled_long_term_memory'}
        if chat_session.is_private_mode:
            logger.info('Private mode active for session %s; skipping.', chat_session_id)
            return {'status': 'skipped', 'reason': 'private_mode'}

        items = list(
            CharacterMemoryItem.objects
            .filter(character=character)
            .order_by('section', 'short_id')
        )
        manager = MemoryManager(character)
        prompt = build_memory_extraction_prompt(
            character_name=character.name,
            items=items,
            chat_session=chat_session,
            new_message=message,
            timezone_name=profile.timezone or 'UTC',
        )
        runtime_config = _get_runtime_model_config(chat_session)
        actions = _collect_memory_actions(
            runtime_config=runtime_config,
            prompt=prompt,
            character=character,
            manager=manager,
            source_message=message,
            tool_specs=get_memory_crud_tool_specs(),
        )

        for action in actions:
            _publish_memory_event(chat_session_id, action)

        logger.info(
            'Long-term memory sync for session=%s character=%s actions=%s',
            chat_session_id, character_id, len(actions),
        )
        return {'status': 'ok', 'actions': len(actions)}
    except Exception as exc:  # noqa: BLE001
        logger.exception('Long-term memory sync failed: session=%s message=%s', chat_session_id, message_id)
        return {'status': 'error', 'error': str(exc)}


@shared_task(retry_backoff=True)
def update_session_title(chat_session_id, history_text, runtime_config):
    try:
        chat_session = ChatSession.objects.get(id=chat_session_id)
    except ChatSession.DoesNotExist:
        return

    # A user-renamed title must never be overwritten by the generator.
    if chat_session.is_title_manual:
        return

    try:
        prompt = (
            "Analyze the following short conversation start.\n"
            "Generate a short, engaging title (2-6 words) that summarizes the topic.\n"
            "Rules:\n"
            "1. Use the same language as the conversation.\n"
            "2. Do not use quotation marks.\n"
            "3. Do not include words like Chat, Conversation, or Title.\n"
            "4. Return only the title text.\n\n"
            f"Conversation:\n{history_text}"
        )

        new_title = _generate_text(runtime_config, prompt).replace('"', '').replace("'", "").strip()
        if new_title:
            chat_session.title = new_title[:200]
            chat_session.save(update_fields=['title'])
    except Exception as exc:
        logger.error("Failed to auto-generate title for session %s: %s", chat_session_id, exc)


def _dispatch_session_title_update(chat_session, history_text, runtime_config):
    """Enqueue the title LLM call so it never blocks the streamed reply.

    Falls back to an inline call when no broker is reachable (dev without
    Redis keeps working, just synchronously)."""
    config_payload = {
        'provider': runtime_config.get('provider'),
        'model_name': runtime_config.get('model_name'),
        'api_key': runtime_config.get('api_key'),
        'base_url': runtime_config.get('base_url', ''),
    }
    try:
        update_session_title.delay(chat_session.id, history_text, config_payload)
        return
    except Exception as exc:
        logger.warning(
            'Failed to enqueue title update for session %s; running inline: %s',
            chat_session.id, exc,
        )

    try:
        update_session_title(chat_session.id, history_text, config_payload)
    except Exception as exc:
        logger.error('Failed to auto-generate title for session %s: %s', chat_session.id, exc)


def _build_research_payload(chat_session, research_context):
    return {
        'query': research_context.get('query', ''),
        'provider': research_context.get('provider', ''),
        'items': research_context.get('items', []),
        'error': research_context.get('error', ''),
    }


def _finalize_ai_response(
    chat_session,
    character,
    runtime_config,
    ai_response_text,
    user_message=None,
    latency_ms=None,
    research_context=None,
    thinking='',
    tool_calls=None,
    token_usage=None,
):
    ai_message = Message.objects.create(
        chat_session=chat_session,
        role='assistant',
        content=ai_response_text,
        character=character,
        research_payload={},
        thinking=thinking or '',
        tool_calls=tool_calls or [],
        token_usage=token_usage or {},
    )

    update_fields = ['updated_at']
    if latency_ms is not None:
        chat_session.last_response_latency_ms = latency_ms
        update_fields.append('last_response_latency_ms')
    chat_session.save(update_fields=update_fields)

    if user_message is not None and not chat_session.is_title_manual:
        # Name the topic exactly once: while the title is still the default,
        # generated from the full conversation opening (greeting + first user
        # turn + first reply). A successful run — or a manual rename — freezes
        # it; a failed run leaves the default title so the next turn retries.
        is_default_title = chat_session.title.startswith("Chat with ")
        if is_default_title:
            history_messages = _get_visible_history_messages(chat_session)
            conversation_text_for_title = '\n'.join(
                f"{'User' if message.role == 'user' else 'Character'}: {(message.content or '')[:200]}"
                for message in history_messages[-6:]
            )
            _dispatch_session_title_update(chat_session, conversation_text_for_title, runtime_config)

    research_payload = _build_research_payload(chat_session, research_context or {})
    ai_message.research_payload = research_payload
    ai_message.save(update_fields=['research_payload'])

    # Async long-term memory write (SonettoHere parity). Fire-and-forget:
    # if the broker is unavailable we still return the AI reply successfully.
    try:
        sync_long_term_memory.delay(
            message_id=ai_message.id,
            chat_session_id=chat_session.id,
            character_id=character.id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning('Failed to enqueue sync_long_term_memory: %s', exc)

    return ai_message


@shared_task(retry_backoff=True)
def generate_ai_response(message_id, character_id, generate_greeting=False, chat_session_id=None):
    try:
        character = Character.objects.get(id=character_id)
        user_message = Message.objects.get(id=message_id) if message_id else None
        chat_session = user_message.chat_session if user_message else None
        if chat_session is None and chat_session_id is not None:
            chat_session = character.chat_sessions.get(id=chat_session_id)
        if chat_session is None:
            raise ValueError('Chat session not found for response generation')
        research_context = build_research_context(chat_session, user_message=user_message)
        runtime_config, formatted_history, tools = _prepare_generation(
            chat_session=chat_session,
            character=character,
            generate_greeting=generate_greeting,
            research_context=research_context,
        )

        started_at = time.perf_counter()
        ai_response_text = _generate_text(
            runtime_config,
            formatted_history,
            tools=tools,
            filesystem=CharacterMemoryFilesystem(character),
        )
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        ai_message = _finalize_ai_response(
            chat_session=chat_session,
            character=character,
            runtime_config=runtime_config,
            ai_response_text=ai_response_text,
            user_message=user_message,
            latency_ms=latency_ms,
            research_context=research_context,
        )

        return {
            'success': True,
            'message_id': ai_message.id,
            'content': ai_response_text,
            'latency_ms': latency_ms,
        }
    except Exception as exc:
        return {
            'success': False,
            'error': str(exc),
        }


def stream_ai_response(chat_session, character, user_message=None, generate_greeting=False):
    # Surface web search as a tool line before the search HTTP call starts.
    collected_tool_calls = []
    profile = _get_user_profile(chat_session)
    if profile.default_enable_web_search:
        search_query = _build_search_query(chat_session, user_message=user_message)
        if search_query:
            search_arguments = {'query': search_query}
            collected_tool_calls.append({'tool': 'web_search', 'arguments': search_arguments})
            yield {'type': 'tool', 'tool': 'web_search', 'arguments': search_arguments}

    research_context = build_research_context(chat_session, user_message=user_message)

    # Shared generation configuration: memory strategy (tools vs prefetch),
    # prompt building and tool specs are decided once in _prepare_generation
    # so the streaming and non-streaming paths can never drift apart.
    runtime_config, formatted_history, tools = _prepare_generation(
        chat_session=chat_session,
        character=character,
        generate_greeting=generate_greeting,
        research_context=research_context,
    )

    started_at = time.perf_counter()
    collected_chunks = []
    collected_thinking = []
    collected_usage = None

    for event in _iter_text_chunks(
        runtime_config,
        formatted_history,
        tools=tools,
        filesystem=CharacterMemoryFilesystem(character),
    ):
        event_type = event.get('type')
        if event_type == 'usage':
            # Internal bookkeeping event; folded into the done payload below.
            collected_usage = _merge_token_usage(collected_usage, event.get('usage'))
            continue
        if event_type == 'delta':
            content = event.get('content') or ''
            if not content:
                continue
            collected_chunks.append(content)
            yield {'type': 'delta', 'content': content}
            continue
        if event_type == 'thinking':
            content = event.get('content') or ''
            if not content:
                continue
            collected_thinking.append(content)
            yield {'type': 'thinking', 'content': content}
            continue
        if event_type == 'tool':
            tool_name = event.get('tool') or ''
            tool_arguments = event.get('arguments') or {}
            collected_tool_calls.append({'tool': tool_name, 'arguments': tool_arguments})
            yield {'type': 'tool', 'tool': tool_name, 'arguments': tool_arguments}
            continue

    ai_response_text = ''.join(collected_chunks).strip()
    latency_ms = int((time.perf_counter() - started_at) * 1000)

    if not ai_response_text:
        raise ValueError('The model returned an empty response')

    ai_message = _finalize_ai_response(
        chat_session=chat_session,
        character=character,
        runtime_config=runtime_config,
        ai_response_text=ai_response_text,
        user_message=user_message,
        latency_ms=latency_ms,
        research_context=research_context,
        thinking=''.join(collected_thinking).strip(),
        tool_calls=collected_tool_calls,
        token_usage=collected_usage,
    )

    yield {
        'type': 'done',
        'message_id': ai_message.id,
        'content': ai_message.content,
        'timestamp': ai_message.timestamp.isoformat(),
        'latency_ms': latency_ms,
        'provider': runtime_config['provider'],
        'model_name': runtime_config['model_name'],
        'research_payload': ai_message.research_payload,
        'thinking': ai_message.thinking,
        'tool_calls': ai_message.tool_calls,
        'token_usage': ai_message.token_usage,
    }
