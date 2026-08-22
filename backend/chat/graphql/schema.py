import strawberry
from typing import List, Optional
from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.core.files import File
import json
import logging
import mimetypes
import os
from urllib.parse import urlparse, unquote

from .types import CharacterType, ChatSessionType, CharacterInput, PrisMateDraft
from chat.attachments import extract_text_attachment_content, guess_attachment_kind, validate_attachment_size
from chat.character_reduce import _normalize_target_name, reduce_result_to_draft, run_reduce_pipeline
from chat.memory.filesystem import StagedUploadMemoryFilesystem
from chat.models import AttachmentKind, Character, CharacterKnowledgeAsset, ChatSession, ModelConfiguration, ModelRole, ModelRoleAssignment, UserProfile
from chat.tasks import (
    _build_memory_tool_specs,
    _extract_json_object,
    _generate_text,
    _supports_memory_tool_mode,
)

logger = logging.getLogger(__name__)
SUPPORTED_BACKGROUND_TEXT_EXTENSIONS = {'.txt', '.md', '.markdown', '.json'}
SUPPORTED_CHARACTER_REFERENCE_KINDS = {AttachmentKind.TEXT, AttachmentKind.IMAGE}


def _get_authenticated_user(info):
    user = getattr(info.context.request, 'user', None)
    if not user or isinstance(user, AnonymousUser) or not user.is_authenticated:
        raise Exception("Authentication required")
    return user


def _get_owned_character(user, character_id):
    try:
        return Character.objects.get(pk=character_id, created_by=user)
    except Character.DoesNotExist as exc:
        raise Exception("Character not found") from exc


def _get_owned_session(user, session_id):
    try:
        return ChatSession.objects.get(pk=session_id, user=user)
    except ChatSession.DoesNotExist as exc:
        raise Exception("Chat session not found") from exc


def _get_required_user_model_config(user):
    model_config = ModelRoleAssignment.get_role_config(user, ModelRole.TEXT)
    if not model_config:
        # 正常流程不会到这里（首个配置自动分配 text、PUT 禁止清空/跳过）；
        # 触发即数据状态异常，回退并留日志。
        model_config = ModelConfiguration.objects.filter(user=user).order_by('id').first()
        if model_config:
            logger.warning(
                'User %s has model configs but no text role assignment; falling back to config %s',
                user.id,
                model_config.id,
            )
    if not model_config:
        raise ValueError("Please configure your own model API before using this feature.")

    # Gemini/Anthropic 路径必须显式 api_key；openai_compatible 允许本地反代网关自鉴权，所以这里放过。
    if not model_config.api_key and model_config.provider in {'gemini', 'anthropic'}:
        raise ValueError("The default user model configuration is missing an API key.")

    return model_config


def _get_draft_runtime_config(user):
    model_config = _get_required_user_model_config(user)

    return {
        "provider": model_config.provider,
        "model_name": model_config.model_name,
        "api_key": model_config.api_key,
        "base_url": model_config.base_url,
    }


def _normalize_draft_locale(value: Optional[str]) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"zh", "zh-cn", "chinese", "simplified chinese"}:
        return "zh-CN"
    return "en-US"


def _get_draft_prompt_locale(user, locale: Optional[str] = None) -> str:
    if locale:
        return _normalize_draft_locale(locale)

    profile = UserProfile.objects.filter(user=user).only("interface_language").first()
    if not profile:
        return "en-US"

    return _normalize_draft_locale(profile.interface_language)


DRAFT_PER_FILE_CHAR_LIMIT = 8000
DRAFT_TOTAL_CHAR_LIMIT = 24000

# 参考文件数达到该阈值时，草稿生成走 reduce 流水线（分层精读 → 笔记 → 合并）
# 而不是单次 Memory Tools ReAct loop。
REDUCE_PIPELINE_MIN_FILES = 12


def _truncate_draft_contents(file_contents: List[str]) -> tuple[List[str], int]:
    """Truncate per-file and total to keep prompts from blowing up.

    Returns the kept contents in order plus the number of files that were dropped
    from the tail because of the total-char budget.
    """
    if not file_contents:
        return [], 0

    truncated: List[str] = []
    running_total = 0
    dropped_tail = 0

    for raw in file_contents:
        if not raw:
            continue
        if len(raw) > DRAFT_PER_FILE_CHAR_LIMIT:
            raw = raw[: DRAFT_PER_FILE_CHAR_LIMIT - 3].rstrip() + "..."
        if running_total + len(raw) > DRAFT_TOTAL_CHAR_LIMIT:
            dropped_tail += 1
            continue
        truncated.append(raw)
        running_total += len(raw)

    return truncated, dropped_tail


def _build_character_draft_prompt(locale: str, text_context: Optional[str], uploaded_file_contents: List[str], dropped_tail_count: int = 0) -> str:
    if locale == "zh-CN":
        prompt_sections = [
            (
                "你是一名专业的角色设计师。\n"
                "请分析提供的上下文，提取稳定的角色锚点和说话风格。\n\n"
                "只返回原始 JSON 对象，不要使用 markdown，不要添加额外说明。JSON 必须包含这些键：\n"
                "- name（字符串）：角色名\n"
                "- description（字符串）：完整的背景与概述，至少 3 句话\n"
                "- affiliation（字符串）：组织、阵营或所属\n"
                "- personality（字符串）：1~2 句话概括角色的语气 / 性格 / 价值倾向，用于快速定调\n"
                "- example_dialogue（字符串）：5 段不同的示例对话。每段格式必须是\n"
                "  \"User: <一句提问或陈述>\\nCharacter: <一句完整回答>\"\n"
                "- tags（字符串数组）：3 到 6 个关键词\n\n"
                "要求：\n"
                "- 优先直接从源材料提取；不要发明设定、场景、外貌、长篇 lore 总结。\n"
                "- personality 要抓\"怎么说\"，而不是\"是谁\"。\n"
                "- example_dialogue 的 5 段要覆盖：日常、提问、情绪、命令/拒绝、玩笑，每段回答不超过 2 句。\n"
                "- 找不到线索时，对应字段返回空字符串（不要编造）。\n\n"
                "待分析的上下文："
            )
        ]

        if text_context:
            prompt_sections.append(f"[用户输入上下文]\n{text_context}")

        for index, file_content_str in enumerate(uploaded_file_contents, start=1):
            prompt_sections.append(f"[上传文件内容 {index}]\n{file_content_str}")

        if dropped_tail_count:
            prompt_sections.append(
                f"[注意：上传的文件中末尾 {dropped_tail_count} 个已被截断以保护上下文窗口]"
            )

        return "\n\n".join(prompt_sections)

    prompt_sections = [
        (
            "You are an expert Character Designer.\n"
            "Analyze the provided context to extract stable character anchors and a voice style.\n\n"
            "Return ONLY a raw JSON object (no markdown formatting) with the following keys:\n"
            "- name (string): Character name\n"
            "- description (string): A comprehensive background and summary (at least 3 sentences)\n"
            "- affiliation (string): Organization or faction\n"
            "- personality (string): 1-2 sentences capturing the character's tone / demeanor / values for quick framing\n"
            "- example_dialogue (string): Exactly 5 distinct example exchanges. Each MUST follow the format\n"
            '  "User: <one short prompt or statement>\\nCharacter: <one reply of up to 2 sentences>"\n'
            "- tags (list of strings): 3-6 keywords\n"
            "\n"
            "Rules:\n"
            "- Prefer direct extraction from the source material; do NOT invent lore, appearance, scenario, or opening lines.\n"
            "- personality should capture HOW the character speaks, not WHO they are.\n"
            "- The 5 example_dialogue exchanges should cover: casual, a question, emotional, a refusal or command, and a joke.\n"
            "- If a field has no signal, return an empty string (never fabricate).\n\n"
            "Context to analyze:"
        )
    ]

    if text_context:
        prompt_sections.append(f"[User Input Context]\n{text_context}")

    for index, file_content_str in enumerate(uploaded_file_contents, start=1):
        prompt_sections.append(f"[Uploaded File Content {index}]\n{file_content_str}")

    if dropped_tail_count:
        prompt_sections.append(
            f"[Note: The last {dropped_tail_count} uploaded file(s) were truncated to protect the context window.]"
        )

    return "\n\n".join(prompt_sections)


def _is_supported_background_text_path(file_path):
    return os.path.splitext(file_path.lower())[1] in SUPPORTED_BACKGROUND_TEXT_EXTENSIONS


def _resolve_local_media_path(file_url: Optional[str]) -> Optional[str]:
    if not file_url:
        return None

    parsed_url = urlparse(file_url)
    relative_path = unquote(parsed_url.path).lstrip('/')
    media_url_path = urlparse(settings.MEDIA_URL).path.lstrip('/')
    if media_url_path and relative_path.startswith(media_url_path):
        relative_path = relative_path[len(media_url_path):].lstrip('/')
    elif relative_path.startswith('media/'):
        relative_path = relative_path[6:]

    file_path = os.path.normpath(os.path.join(settings.MEDIA_ROOT, relative_path))
    media_root = os.path.normpath(settings.MEDIA_ROOT)
    try:
        if os.path.commonpath([media_root, file_path]) != media_root:
            return None
    except ValueError:
        return None

    return file_path if os.path.exists(file_path) else None


def _decode_text_content(raw_bytes: bytes) -> str:
    for encoding in ('utf-8-sig', 'utf-8', 'utf-16'):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode('utf-8', errors='replace')


def _read_local_text_file(file_url: Optional[str]) -> str:
    file_path = _resolve_local_media_path(file_url)
    if not file_path or not _is_supported_background_text_path(file_path):
        return ""

    try:
        with open(file_path, 'rb') as uploaded_file:
            return _decode_text_content(uploaded_file.read())
    except OSError as exc:
        logger.warning("Failed to read uploaded background text from %s: %s", file_path, exc)
        return ""


def _resolve_staged_uploads(file_urls: Optional[List[str]]) -> List[dict]:
    """Resolve the just-uploaded file URLs into the staged-upload records that
    the draft Memory Tools browse. Text files carry their extracted content;
    images carry only their URL/metadata (they cannot be read as text)."""
    if not file_urls:
        return []

    uploads = []
    seen_urls = set()
    for file_url in file_urls:
        if not file_url or file_url in seen_urls:
            continue
        seen_urls.add(file_url)

        file_path = _resolve_local_media_path(file_url)
        if not file_path:
            continue

        name = os.path.basename(file_path)
        mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        if _is_supported_background_text_path(file_path):
            uploads.append({
                "name": name,
                "kind": AttachmentKind.TEXT,
                "mime_type": mime_type,
                "content": _read_local_text_file(file_url),
                "file_url": file_url,
            })
        elif mime_type.startswith("image/"):
            uploads.append({
                "name": name,
                "kind": AttachmentKind.IMAGE,
                "mime_type": mime_type,
                "content": "",
                "file_url": file_url,
            })

    return uploads


def _build_character_draft_tool_prompt(locale: str, text_context: Optional[str], upload_count: int) -> List[dict]:
    """Build the system/user messages for the tool-driven character draft.

    Unlike ``_build_character_draft_prompt``, uploaded file bodies are *not*
    inlined here. The model must use ``list_memory_files`` / ``read_memory_file``
    to read the files it actually needs.
    """
    if locale == "zh-CN":
        system_prompt = (
            "你是一名专业的角色设计师。\n"
            "请分析提供的上下文，提取稳定的角色锚点和说话风格。\n\n"
            f"用户上传了 {upload_count} 个参考文件，它们通过记忆文件系统暴露，"
            "并不会出现在这条提示里。\n"
            "必须使用工具按需查阅文件，而不是假设内容：\n"
            "- 先用 list_memory_files 浏览上传文件（路径前缀 raw/character_setup/uploads）。\n"
            "- 再用 read_memory_file 只读取与角色塑造相关的文件。\n"
            "- 没有实际读取过的文件，不得声称知道其内容。\n\n"
            "只返回原始 JSON 对象，不要使用 markdown，不要添加额外说明。JSON 必须包含这些键：\n"
            "- name（字符串）：角色名\n"
            "- description（字符串）：完整的背景与概述，至少 3 句话\n"
            "- affiliation（字符串）：组织、阵营或所属\n"
            "- personality（字符串）：1~2 句话概括角色的语气 / 性格 / 价值倾向，用于快速定调\n"
            "- example_dialogue（字符串）：5 段不同的示例对话。每段格式必须是\n"
            "  \"User: <一句提问或陈述>\\nCharacter: <一句完整回答>\"\n"
            "- tags（字符串数组）：3 到 6 个关键词\n\n"
            "要求：\n"
            "- 优先直接从读取到的源材料提取；不要发明设定、场景、外貌、长篇 lore 总结。\n"
            "- personality 要抓\"怎么说\"，而不是\"是谁\"。\n"
            "- example_dialogue 的 5 段要覆盖：日常、提问、情绪、命令/拒绝、玩笑，每段回答不超过 2 句。\n"
            "- 找不到线索时，对应字段返回空字符串（不要编造）。"
        )
        user_prompt = (
            (f"[用户输入上下文]\n{text_context}" if text_context else "[用户输入上下文]\n（未提供额外上下文）")
            + "\n\n请先用工具浏览并读取你需要的上传文件，然后输出角色草稿 JSON。"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    system_prompt = (
        "You are an expert Character Designer.\n"
        "Analyze the provided context to extract stable character anchors and a voice style.\n\n"
        f"The user uploaded {upload_count} reference file(s). They are exposed through a memory "
        "filesystem and are NOT included in this prompt.\n"
        "You MUST use the tools to read files on demand instead of assuming their content:\n"
        "- Call list_memory_files first to browse the uploaded files (path prefix raw/character_setup/uploads).\n"
        "- Call read_memory_file to open only the files relevant to the character you are building.\n"
        "- Never claim facts about a file you have not actually read.\n\n"
        "Return ONLY a raw JSON object (no markdown formatting) with the following keys:\n"
        "- name (string): Character name\n"
        "- description (string): A comprehensive background and summary (at least 3 sentences)\n"
        "- affiliation (string): Organization or faction\n"
        "- personality (string): 1-2 sentences capturing the character's tone / demeanor / values for quick framing\n"
        "- example_dialogue (string): Exactly 5 distinct example exchanges. Each MUST follow the format\n"
        '  "User: <one short prompt or statement>\\nCharacter: <one reply of up to 2 sentences>"\n'
        "- tags (list of strings): 3-6 keywords\n"
        "\n"
        "Rules:\n"
        "- Prefer direct extraction from the source material you read; do NOT invent lore, appearance, scenario, or opening lines.\n"
        "- personality should capture HOW the character speaks, not WHO they are.\n"
        "- The 5 example_dialogue exchanges should cover: casual, a question, emotional, a refusal or command, and a joke.\n"
        "- If a field has no signal, return an empty string (never fabricate)."
    )
    user_prompt = (
        (f"[User Input Context]\n{text_context}" if text_context else "[User Input Context]\n(no extra context provided)")
        + "\n\nBrowse and read the uploaded files you need via the tools, then output the character draft JSON."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _normalize_character_reference_inputs(input: CharacterInput):
    if input.background_files is not None:
        return [
            {
                "uploaded_url": item.uploaded_url,
                "file_name": item.file_name,
            }
            for item in input.background_files
            if item.uploaded_url
        ]

    if input.background_file_url:
        return [{
            "uploaded_url": input.background_file_url,
            "file_name": input.background_file_name or os.path.basename(input.background_file_url),
        }]

    return None


def _attach_character_reference_assets(character, uploaded_assets, replace_existing=True):
    if uploaded_assets is None:
        return

    existing_assets = list(character.knowledge_assets.all()) if replace_existing else []

    for index, uploaded_asset in enumerate(uploaded_assets):
        file_path = _resolve_local_media_path(uploaded_asset.get("uploaded_url"))
        if not file_path:
            continue

        file_name = (uploaded_asset.get("file_name") or os.path.basename(file_path)).strip() or os.path.basename(file_path)
        mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"

        with open(file_path, 'rb') as source_file:
            django_file = File(source_file, name=file_name)
            setattr(django_file, 'content_type', mime_type)
            attachment_kind, attachment_mime_type = guess_attachment_kind(django_file)
            if attachment_kind not in SUPPORTED_CHARACTER_REFERENCE_KINDS:
                raise ValueError("Only text files and images are supported for character reference uploads.")

            validate_attachment_size(django_file, attachment_kind)
            attachment_text_content = extract_text_attachment_content(django_file) if attachment_kind == AttachmentKind.TEXT else ""
            django_file.seek(0)

            asset = CharacterKnowledgeAsset(
                character=character,
                attachment_name=file_name,
                attachment_mime_type=attachment_mime_type,
                attachment_kind=attachment_kind,
                attachment_text_content=attachment_text_content,
                sort_order=index,
            )
            asset.file.save(file_name, django_file, save=False)
            asset.save()

    for existing_asset in existing_assets:
        existing_asset.file.delete(save=False)
        existing_asset.delete()

@strawberry.input
class ChatSessionInput:
    character_id: strawberry.ID
    title: Optional[str] = ""

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def generate_character_draft(
        self,
        info,
        file_url: Optional[str] = None,
        file_urls: Optional[List[str]] = None,
        text_context: Optional[str] = None,
        locale: Optional[str] = None,
    ) -> PrisMateDraft:
        """
        Calls the user's default model configuration to analyze text and return a structured Character Draft.

        Uploaded reference files are not inlined into the prompt. When the
        runtime model supports tool calls (OpenAI-compatible / Anthropic),
        the files are exposed through the ``list_memory_files`` /
        ``read_memory_file`` tools so the model reads only what it needs.
        Gemini falls back to reading text files locally.
        """
        user = await sync_to_async(_get_authenticated_user)(info)

        try:
            runtime_config = await sync_to_async(_get_draft_runtime_config)(user)
            draft_locale = await sync_to_async(_get_draft_prompt_locale)(user, locale)

            normalized_file_urls = []
            if file_urls:
                normalized_file_urls.extend(file_urls)
            if file_url:
                normalized_file_urls.append(file_url)

            staged_uploads = _resolve_staged_uploads(normalized_file_urls)

            # 大量参考文件：走 reduce 流水线（分层精读 → 结构化笔记 → 合并），
            # 避免单次 ReAct loop 无法覆盖全部文件。少量文件仍走 Memory Tools
            # 按需读取，保证小批量响应速度。
            text_uploads = [
                upload for upload in staged_uploads
                if upload.get("kind") == AttachmentKind.TEXT and upload.get("content")
            ]
            if len(text_uploads) >= REDUCE_PIPELINE_MIN_FILES:
                target_name = _normalize_target_name(text_context)

                def reduce_llm_call(system_prompt: str, user_prompt: str) -> str:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    raw = _generate_text(runtime_config, messages)
                    data = _extract_json_object(raw)
                    if not data:
                        raise ValueError(
                            "Model did not return a valid JSON object for the reduce step. "
                            f"Raw model response preview: {(raw or '')[:300]!r}"
                        )
                    return json.dumps(data, ensure_ascii=False)

                pipeline_result = await sync_to_async(run_reduce_pipeline)(
                    text_uploads,
                    target_name,
                    llm_call=reduce_llm_call,
                )
                draft_data = reduce_result_to_draft(pipeline_result)
                return PrisMateDraft(
                    name=draft_data.get("name", "Unknown"),
                    description=draft_data.get("description", ""),
                    affiliation=draft_data.get("affiliation", ""),
                    personality=draft_data.get("personality", ""),
                    appearance=draft_data.get("appearance", ""),
                    tags=draft_data.get("tags", []) or [],
                    visual_summary=draft_data.get("visual_summary", ""),
                    example_dialogue=(draft_data.get("example_dialogue") or "").strip(),
                )

            use_memory_tools = bool(staged_uploads) and _supports_memory_tool_mode(runtime_config)

            if use_memory_tools:
                filesystem = StagedUploadMemoryFilesystem(staged_uploads)
                messages = _build_character_draft_tool_prompt(
                    draft_locale,
                    text_context,
                    len(staged_uploads),
                )
                raw_text = await sync_to_async(_generate_text)(
                    runtime_config,
                    messages,
                    tools=_build_memory_tool_specs(),
                    filesystem=filesystem,
                )
            else:
                raw_file_contents = [
                    upload["content"]
                    for upload in staged_uploads
                    if upload["kind"] == AttachmentKind.TEXT and upload["content"]
                ]
                truncated_contents, dropped_tail_count = _truncate_draft_contents(raw_file_contents)
                prompt = _build_character_draft_prompt(
                    draft_locale,
                    text_context,
                    truncated_contents,
                    dropped_tail_count=dropped_tail_count,
                )
                raw_text = await sync_to_async(_generate_text)(runtime_config, prompt)
            data = _extract_json_object(raw_text)
            if not data:
                # Hard fail: the prompt contract is "return ONLY a raw JSON
                # object". When the model returns prose (or nothing) the user
                # should see a clear error, not a half-populated form. The
                # error message embeds a preview of the raw model response
                # so the user can see what the proxy sent without silent
                # degradation. The parser has also written the complete text
                # to a file in the OS temp directory and logged the full
                # path at INFO level, so the backend log is the source of
                # truth for the dump file location (don't hardcode /tmp
                # here — on Windows that path doesn't exist).
                raw_for_preview = raw_text or ''
                preview = raw_for_preview[:500]
                if len(raw_for_preview) > 500:
                    preview += (
                        f'... [truncated; full response was {len(raw_for_preview)} chars; '
                        f'see the backend log for the parser dump file path]'
                    )
                raise ValueError(
                    f"Model did not return a valid JSON object for the character draft. "
                    f"Raw model response preview: {preview or '(empty response)'}"
                )

            return PrisMateDraft(
                name=data.get("name", "Unknown"),
                description=data.get("description", ""),
                affiliation=data.get("affiliation", ""),
                personality=(data.get("personality") or "").strip(),
                appearance="",
                tags=data.get("tags", []) or [],
                visual_summary="",
                example_dialogue=(data.get("example_dialogue") or "").strip(),
            )

        except Exception as e:
            logger.error(f"AI Generation Error: {e}")
            return PrisMateDraft(
                name="Generation Failed",
                description=f"Error generating draft: {str(e)}",
                personality="", appearance="", affiliation="",
                tags=[], visual_summary="", example_dialogue=""
            )

    @strawberry.mutation
    async def create_character(self, info, input: CharacterInput) -> CharacterType:
        @sync_to_async
        def create_char_sync():
            user = _get_authenticated_user(info)
            character = Character.objects.create(
                name=input.name,
                avatar_url=input.avatar_url,
                description=input.description,
                user_address=input.user_address,
                personality=input.personality,
                appearance=input.appearance,
                response_guidelines=input.response_guidelines,
                scenario=input.scenario,
                example_dialogue=input.example_dialogue,
                affiliation=input.affiliation,
                system_prompt_preview=input.system_prompt_preview,
                tags=input.tags,
                created_by=user
            )
            _attach_character_reference_assets(
                character,
                _normalize_character_reference_inputs(input),
                replace_existing=True,
            )
            character.save()
            return character
        character = await create_char_sync()
        return character

    @strawberry.mutation
    async def delete_character(self, info, id: strawberry.ID) -> bool:
        @sync_to_async
        def delete_sync():
            user = _get_authenticated_user(info)
            character = _get_owned_character(user, id)

            if character.chat_sessions.exists():
                return False

            character.delete()
            return True
        return await delete_sync()

    @strawberry.mutation
    async def update_character(self, info, id: strawberry.ID, input: CharacterInput) -> CharacterType:
        @sync_to_async
        def update_char_sync():
            user = _get_authenticated_user(info)
            character = _get_owned_character(user, id)
            uploaded_assets = _normalize_character_reference_inputs(input)
            character.name = input.name
            character.avatar_url = input.avatar_url
            character.description = input.description
            character.user_address = input.user_address
            character.personality = input.personality
            character.appearance = input.appearance
            character.response_guidelines = input.response_guidelines
            character.scenario = input.scenario
            character.example_dialogue = input.example_dialogue
            character.affiliation = input.affiliation
            character.system_prompt_preview = input.system_prompt_preview
            character.tags = input.tags
            _attach_character_reference_assets(
                character,
                uploaded_assets,
                replace_existing=uploaded_assets is not None,
            )
            character.save()
            return character
        character = await update_char_sync()
        return character

    @strawberry.mutation
    async def create_chat_session(self, info, input: ChatSessionInput) -> ChatSessionType:
        @sync_to_async
        def create_session_sync():
            user = _get_authenticated_user(info)
            character = _get_owned_character(user, input.character_id)
            _get_required_user_model_config(user)

            return ChatSession.objects.create(
                character=character,
                user=user,
                title=input.title or f"Chat with {character.name}",
            )
        return await create_session_sync()

    @strawberry.mutation
    async def update_chat_session(self, info, id: strawberry.ID, input: ChatSessionInput) -> ChatSessionType:
        @sync_to_async
        def update_session_sync():
            user = _get_authenticated_user(info)
            session = _get_owned_session(user, id)
            session.title = input.title or session.title
            session.save()
            return session
        return await update_session_sync()

@strawberry.type
class Query:
    @strawberry.django.field
    def characters(self, info) -> List[CharacterType]:
        user = _get_authenticated_user(info)
        return Character.objects.filter(created_by=user)

    @strawberry.django.field
    def character(self, info, id: strawberry.ID) -> CharacterType:
        user = _get_authenticated_user(info)
        return _get_owned_character(user, id)

    @strawberry.django.field
    def chat_sessions(self, info) -> List[ChatSessionType]:
        user = _get_authenticated_user(info)
        return ChatSession.objects.filter(user=user).order_by('-updated_at')

    @strawberry.django.field
    def chat_session(self, info, id: strawberry.ID) -> ChatSessionType:
        user = _get_authenticated_user(info)
        return _get_owned_session(user, id)

schema = strawberry.Schema(query=Query, mutation=Mutation)
