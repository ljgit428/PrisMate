import json

from django.http import StreamingHttpResponse
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.exceptions import ValidationError
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .attachments import extract_text_attachment_content, guess_attachment_kind, validate_attachment_size
from .memory.interface import LongTermMemoryInterface as CharacterLongTermMemory
from .memory.manager import MemoryItemNotFoundError, MemoryManager
from .models import (
    AttachmentKind,
    Character,
    CharacterKnowledgeAsset,
    ChatSession,
    MemoryAuditSource,
    Message,
    MessageAttachment,
    ModelConfiguration,
    ModelRole,
    ModelRoleAssignment,
    UserProfile,
    WebSearchConfiguration,
)
from .model_catalog import probe_provider_models
from .search import search_web
from .serializers import (
    CharacterSerializer,
    CharacterKnowledgeAssetSerializer,
    ChatSessionSerializer,
    ChatSessionWriteSerializer,
    MessageSerializer,
    MessageCreateSerializer,
    ModelConfigurationSerializer,
    UserProfileSerializer,
    WebSearchConfigurationSerializer,
)
from .tasks import generate_ai_response, stream_ai_response
from .soul import list_memory_explorer_path, read_memory_explorer_file
import logging

logger = logging.getLogger(__name__)


def _message_serializer(message, request):
    return MessageSerializer(message, context={'request': request})


def _get_required_model_config(user, model_config_id=None):
    if model_config_id not in [None, "", "null"]:
        model_config = ModelConfiguration.objects.get(id=model_config_id, user=user)
    else:
        model_config = ModelRoleAssignment.get_role_config(user, ModelRole.TEXT) or (
            ModelConfiguration.objects.filter(user=user).order_by('id').first()
        )

    if not model_config:
        raise ValueError('Please configure your own model API in Project Settings before starting a chat.')

    # Gemini/Anthropic 必须显式 api_key；openai_compatible 允许本地反代网关自鉴权，所以这里放过。
    if not model_config.api_key and model_config.provider in {'gemini', 'anthropic'}:
        raise ValueError('The selected model configuration is missing an API key.')

    return model_config
class CharacterViewSet(viewsets.ModelViewSet):
    queryset = Character.objects.all()
    serializer_class = CharacterSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Character.objects.filter(created_by=self.request.user)

    def perform_create(self, serializer):
        user = self.request.user
        serializer.save(created_by=user)

    def perform_update(self, serializer):
        serializer.save()

    def destroy(self, request, *args, **kwargs):
        character = self.get_object()
        if character.chat_sessions.exists():
            return Response(
                {'error': 'Cannot delete a character with existing chat sessions'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        return super().destroy(request, *args, **kwargs)

    @action(detail=True, methods=['get'])
    def soul_files(self, request, pk=None):
        character = self.get_object()
        return Response(list_memory_explorer_path(
            character,
            path_prefix=request.query_params.get('path_prefix', ''),
            recursive=request.query_params.get('recursive', '').lower() == 'true',
            max_entries=request.query_params.get('max_entries', 200),
        ))

    @action(detail=True, methods=['get'])
    def soul_file(self, request, pk=None):
        character = self.get_object()
        path = request.query_params.get('path', '')
        if not path:
            raise ValidationError({'path': 'path is required'})
        return Response(read_memory_explorer_file(
            character,
            path=path,
            max_chars=request.query_params.get('max_chars', 6000),
        ))

    @action(detail=True, methods=['post'], url_path='knowledge_assets')
    def upload_knowledge_assets(self, request, pk=None):
        character = self.get_object()
        files = list(request.FILES.getlist('files'))
        if not files:
            single_file = request.FILES.get('file')
            if single_file:
                files = [single_file]

        if not files:
            raise ValidationError({'files': 'At least one file is required.'})

        next_sort_order = (
            character.knowledge_assets.order_by('-sort_order').values_list('sort_order', flat=True).first() or 0
        )
        created_assets = []
        for index, uploaded_file in enumerate(files, start=1):
            attachment_kind, attachment_mime_type = guess_attachment_kind(uploaded_file)
            if attachment_kind not in {AttachmentKind.TEXT, AttachmentKind.IMAGE}:
                raise ValidationError({'files': 'Only text files and images are supported for character reference uploads.'})

            validate_attachment_size(uploaded_file, attachment_kind)
            attachment_text_content = ''
            if attachment_kind == AttachmentKind.TEXT:
                attachment_text_content = extract_text_attachment_content(uploaded_file)

            created_assets.append(
                CharacterKnowledgeAsset.objects.create(
                    character=character,
                    file=uploaded_file,
                    attachment_name=uploaded_file.name or '',
                    attachment_mime_type=attachment_mime_type,
                    attachment_kind=attachment_kind,
                    attachment_text_content=attachment_text_content,
                    sort_order=next_sort_order + index,
                )
            )

        serializer = CharacterKnowledgeAssetSerializer(created_assets, many=True, context={'request': request})
        return Response({'assets': serializer.data}, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['delete'], url_path=r'knowledge_assets/(?P<asset_id>[^/.]+)')
    def delete_knowledge_asset(self, request, pk=None, asset_id=None):
        character = self.get_object()
        try:
            asset = character.knowledge_assets.get(pk=asset_id)
        except CharacterKnowledgeAsset.DoesNotExist as exc:
            raise ValidationError({'asset_id': 'Knowledge asset not found for this character.'}) from exc

        asset.file.delete(save=False)
        asset.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=True, methods=['get', 'post', 'delete'], url_path='memory')
    def memory_collection(self, request, pk=None):
        view = CharacterMemoryViewSet()
        view.permission_classes = self.permission_classes
        view.request = request
        if request.method == 'GET':
            return view.list(request=request, pk=pk)
        if request.method == 'POST':
            return view.create(request=request, pk=pk)
        return view.wipe_all(request=request, pk=pk)

    # NOTE: ``memory/merge`` is declared BEFORE ``memory/{short_id}`` so the
    # literal route is tried first by DRF's router. The ``memory_item``
    # regex below ALSO uses a negative lookahead to reject reserved
    # sub-action names ("merge", "audit", "narrative", "wipe"). The
    # lookahead uses ``/?$`` (not bare ``$``) so reserved names match
    # even when the trailing slash is present (``memory/merge/`` would
    # otherwise slip past the bare ``$`` anchor and be captured as a
    # ``short_id``). The lookahead is the structural enforcement; the
    # declaration ordering is belt-and-suspenders that future readers
    # can safely ignore.
    @action(detail=True, methods=['post'], url_path='memory/merge')
    def memory_merge(self, request, pk=None):
        view = CharacterMemoryViewSet()
        view.permission_classes = self.permission_classes
        view.request = request
        return view.merge(request=request, pk=pk)

    @action(
        detail=True,
        methods=['patch', 'delete'],
        url_path=r'memory/(?P<short_id>(?!merge/?$|audit/?$|narrative/?$|wipe/?$)[^/.]+)',
    )
    def memory_item(self, request, pk=None, short_id=None):
        view = CharacterMemoryViewSet()
        view.permission_classes = self.permission_classes
        view.request = request
        if request.method == 'PATCH':
            return view.update(request=request, pk=pk, short_id=short_id)
        return view.destroy(request=request, pk=pk, short_id=short_id)


class ModelConfigurationViewSet(viewsets.ModelViewSet):
    queryset = ModelConfiguration.objects.none()
    serializer_class = ModelConfigurationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return ModelConfiguration.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        model_config = serializer.save(user=self.request.user)
        # 首个配置自动接管 text 槽位：避免「有配置但无 text 分配」的状态
        # （该状态下聊天会静默回退到 id 最小的配置，可能选错模型）。
        if not ModelRoleAssignment.objects.filter(user=self.request.user, role=ModelRole.TEXT).exists():
            ModelRoleAssignment.objects.get_or_create(
                user=self.request.user,
                role=ModelRole.TEXT,
                defaults={'model_config': model_config},
            )

    def perform_update(self, serializer):
        serializer.save(user=self.request.user)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        if ModelRoleAssignment.objects.filter(model_config=instance, role=ModelRole.TEXT).exists():
            return Response(
                {'error': 'This model is assigned to the text role. Switch the text model before deleting it.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        # 媒体槽位（图片/音频/视频）引用随 CASCADE 自动清空。
        return super().destroy(request, *args, **kwargs)


class ModelRoleAssignmentView(APIView):
    """四角色槽位（text/image/audio/video）的读取与更新。

    PUT 语义：payload 中出现的角色会被设置（显式 null 表示清空），
    未出现的角色保持现状；text 角色必须始终指向一个有效配置。
    """

    permission_classes = [IsAuthenticated]

    def _serialize_assignments(self, user):
        data = {role: None for role, _label in ModelRole.choices}
        assignments = (
            ModelRoleAssignment.objects.filter(user=user)
            .select_related('model_config')
        )
        for assignment in assignments:
            data[assignment.role] = ModelConfigurationSerializer(
                assignment.model_config,
                context={'request': self.request},
            ).data
        return data

    def get(self, request):
        return Response(self._serialize_assignments(request.user))

    def put(self, request):
        payload = request.data if isinstance(request.data, dict) else {}

        requested = {}
        for role, _label in ModelRole.choices:
            if role not in payload:
                continue
            value = payload.get(role)
            requested[role] = None if value in [None, '', 'null'] else str(value)

        if not requested:
            return Response({'error': 'No role assignments provided'}, status=status.HTTP_400_BAD_REQUEST)

        if ModelRole.TEXT in requested and requested[ModelRole.TEXT] is None:
            return Response(
                {'error': 'A text model is required and cannot be empty'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # payload 未提及 text 时，必须已存在 text 分配；否则会留下
        # 「有配置但无 text」的静默错误状态（聊天回退到任意第一个配置）。
        if ModelRole.TEXT not in requested and not ModelRoleAssignment.objects.filter(
            user=request.user, role=ModelRole.TEXT
        ).exists():
            return Response(
                {'error': 'A text model is required. Assign a text model before configuring other roles.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        config_ids = {value for value in requested.values() if value}
        configs = {
            str(config.id): config
            for config in ModelConfiguration.objects.filter(user=request.user, id__in=config_ids)
        }
        for role, value in requested.items():
            if value and value not in configs:
                return Response(
                    {'error': f'Model configuration not found or access denied for role "{role}"'},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        from django.db import transaction

        with transaction.atomic():
            for role, value in requested.items():
                ModelRoleAssignment.objects.filter(user=request.user, role=role).delete()
                if value:
                    ModelRoleAssignment.objects.create(
                        user=request.user,
                        role=role,
                        model_config=configs[value],
                    )

        return Response(self._serialize_assignments(request.user))


class ModelCatalogViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]

    # POST 而非 GET：api_key 放 body，避免泄漏进 access log / 浏览器历史
    @action(detail=False, methods=['post'])
    def probe(self, request):
        payload = request.data if isinstance(request.data, dict) else {}
        provider = str(payload.get('provider') or '').strip()
        base_url = str(payload.get('base_url') or '').strip()
        api_key = str(payload.get('api_key') or '').strip()

        try:
            models = probe_provider_models(provider, base_url=base_url, api_key=api_key)
        except ValueError as exc:
            return Response({'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            logger.warning('Model probe failed for provider=%s base_url=%s: %s', provider, base_url, exc)
            return Response(
                {'error': 'Failed to fetch the model list. Check the provider, base URL, and API key, or enter the model name manually.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response({'models': models})


class UserProfileViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=['get', 'patch'])
    def me(self, request):
        profile = UserProfile.get_or_create_for_user(request.user)

        if request.method == 'GET':
            return Response(UserProfileSerializer(profile).data)

        serializer = UserProfileSerializer(profile, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)


class WebSearchConfigurationViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=['get', 'patch'])
    def me(self, request):
        config = WebSearchConfiguration.get_for_user(request.user)

        if request.method == 'GET':
            config = config or WebSearchConfiguration(
                user=request.user,
                provider='tavily',
                api_key='',
                max_results=5,
            )
            return Response(WebSearchConfigurationSerializer(config).data)

        serializer = WebSearchConfigurationSerializer(
            config,
            data=request.data,
            partial=True,
            context={'user': request.user},
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

    @action(detail=False, methods=['post'])
    def test(self, request):
        query = (request.data.get('query') or '').strip()
        if not query:
            raise ValidationError({'query': 'query is required'})

        config = WebSearchConfiguration.get_for_user(request.user)
        payload = search_web(query, user=request.user, config=config)
        return Response(payload)


class ChatSessionViewSet(viewsets.ModelViewSet):
    queryset = ChatSession.objects.none()
    permission_classes = [IsAuthenticated]

    def get_serializer_class(self):
        if self.action in ['create', 'update', 'partial_update']:
            return ChatSessionWriteSerializer
        return ChatSessionSerializer

    def get_queryset(self):
        queryset = ChatSession.objects.filter(user=self.request.user).select_related('character').prefetch_related('messages__attachments')
        character_id = self.request.query_params.get('character_id')
        if character_id:
            queryset = queryset.filter(character_id=character_id)

        origin = self.request.query_params.get('origin')
        if origin in dict(ChatSession.ORIGIN_CHOICES):
            queryset = queryset.filter(origin=origin)

        return queryset.order_by('-updated_at')

    def perform_create(self, serializer):
        user = self.request.user
        try:
            _get_required_model_config(user=user)
        except ValueError as exc:
            raise ValidationError({'title': str(exc)}) from exc
        serializer.save(user=user)

    def partial_update(self, request, *args, **kwargs):
        """Allow toggling ``is_private_mode`` from the chat composer without
        sending every other field. A manually patched ``title`` marks the
        session so the auto title generator will not overwrite it."""
        instance = self.get_object()
        if 'is_private_mode' in request.data:
            is_private_mode = self._parse_bool(request.data.get('is_private_mode'))
            instance.is_private_mode = is_private_mode
            instance.save(update_fields=['is_private_mode', 'updated_at'])
            return Response(ChatSessionSerializer(instance).data)

        if 'title' in request.data and (request.data.get('title') or '').strip():
            # Set the manual flag before the generic update so the auto
            # generator can never race a just-renamed session.
            instance.is_title_manual = True
            instance.save(update_fields=['is_title_manual'])

        super().partial_update(request, *args, **kwargs)
        instance.refresh_from_db()
        return Response(ChatSessionSerializer(instance, context={'request': request}).data)

    @staticmethod
    def _parse_bool(value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).lower() in {'1', 'true', 'yes', 'on'}


class CharacterMemoryViewSet(viewsets.ViewSet):
    """REST surface for ``wiki/memory.md`` (per-character long-term memory).

    Five CRUD verbs mirroring SonettoHeres tools byte-for-byte:

    - ``GET /api/characters/{id}/memory/`` — snapshot grouped by section.
    - ``POST /api/characters/{id}/memory/`` — ``create_memory`` action.
    - ``PATCH /api/characters/{id}/memory/{short_id}/`` — ``update_memory``.
    - ``DELETE /api/characters/{id}/memory/{short_id}/`` — ``delete_memory``.
    - ``POST /api/characters/{id}/memory/merge/`` — ``merge_memories``.

    Every write logs to ``MemoryAuditLog`` with source ``user_edit``,
    matching the audit trail that ``sync_long_term_memory`` already writes.
    """

    permission_classes = [IsAuthenticated]

    def _memory(self, request, pk):
        # Use get_object_or_404 so cross-user access surfaces as 404
        # instead of leaking a 500 from an uncaught DoesNotExist.
        character = get_object_or_404(
            Character,
            pk=pk,
            created_by=request.user,
        )
        return character, MemoryManager(character)

    def list(self, request, pk=None):
        character, manager = self._memory(request, pk)
        snapshot = CharacterLongTermMemory.snapshot(character)
        return Response({
            'sections': snapshot['sections'],
            'wiki_markdown': manager.render_wiki_markdown(),
            'count': sum(len(section['items']) for section in snapshot['sections']),
        })

    def create(self, request, pk=None):
        character, manager = self._memory(request, pk)
        description = request.data.get('description', '')
        section = request.data.get('section', '')
        reason = request.data.get('reason', '')
        try:
            item = manager.create_item(
                section=section,
                description=description,
                source=MemoryAuditSource.USER_EDIT,
                reason=reason or 'User edit via /memory page.',
            )
        except ValueError as exc:
            raise ValidationError({'description': str(exc)}) from exc
        return Response(_serialise_memory_item(item), status=status.HTTP_201_CREATED)

    def update(self, request, pk=None, short_id=None):
        character, manager = self._memory(request, pk)
        description = request.data.get('description', '')
        section = request.data.get('section')
        reason = request.data.get('reason', '')
        try:
            item = manager.update_item(
                short_id=short_id,
                description=description,
                section=section,
                reason=reason or 'User edit via /memory page.',
                source=MemoryAuditSource.USER_EDIT,
            )
        except ValueError as exc:
            raise ValidationError({'description': str(exc)}) from exc
        except MemoryItemNotFoundError as exc:
            raise ValidationError({'short_id': str(exc)}) from exc
        return Response(_serialise_memory_item(item))

    def destroy(self, request, pk=None, short_id=None):
        character, manager = self._memory(request, pk)
        reason = request.data.get('reason', '') or 'User delete via /memory page.'
        try:
            deleted_desc = manager.delete_item(
                short_id=short_id,
                reason=reason,
                source=MemoryAuditSource.USER_EDIT,
            )
        except MemoryItemNotFoundError as exc:
            raise ValidationError({'short_id': str(exc)}) from exc
        return Response({'deleted': short_id, 'description': deleted_desc}, status=status.HTTP_200_OK)

    @action(detail=False, methods=['post'])
    def merge(self, request, pk=None):
        character, manager = self._memory(request, pk)
        try:
            item = manager.merge_items(
                id1=request.data.get('id1', ''),
                id2=request.data.get('id2', ''),
                content=request.data.get('content', ''),
                section=request.data.get('section', ''),
                reason=request.data.get('reason', '') or 'User merge via /memory page.',
                source=MemoryAuditSource.USER_EDIT,
            )
        except ValueError as exc:
            raise ValidationError({'content': str(exc)}) from exc
        except MemoryItemNotFoundError as exc:
            raise ValidationError({'short_id': str(exc)}) from exc
        return Response(_serialise_memory_item(item))

    @action(detail=False, methods=['delete'])
    def wipe_all(self, request, pk=None):
        character, _ = self._memory(request, pk)
        deleted = CharacterLongTermMemory.wipe(character)
        return Response({'deleted': deleted})


def _serialise_memory_item(item):
    return {
        'short_id': item.short_id,
        'section': item.section,
        'description': item.description,
        'description_history': list(item.description_history or []),
        'created_at': item.created_at.isoformat() if item.created_at else None,
        'updated_at': item.updated_at.isoformat() if item.updated_at else None,
    }

class MessageViewSet(viewsets.ModelViewSet):
    queryset = Message.objects.none()
    permission_classes = [IsAuthenticated]

    def get_serializer_class(self):
        if self.action == 'create':
            return MessageCreateSerializer
        return MessageSerializer

    def get_queryset(self):
        queryset = Message.objects.filter(chat_session__user=self.request.user).prefetch_related('attachments').order_by('timestamp')
        chat_session_id = self.request.query_params.get('chat_session_id')
        if chat_session_id:
            queryset = queryset.filter(chat_session_id=chat_session_id)
        return queryset

    def perform_create(self, serializer):
        chat_session_id = self.request.data.get('chat_session_id')
        if not chat_session_id:
            return Response(
                {'error': 'chat_session_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            user = self.request.user

            chat_session = ChatSession.objects.get(
                id=chat_session_id,
                user=user
            )
            serializer.save(chat_session=chat_session)
        except ChatSession.DoesNotExist:
            return Response(
                {'error': 'Chat session not found or access denied'},
                status=status.HTTP_404_NOT_FOUND
            )

class ChatViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=['post'])
    def send_message(self, request):
        try:
            chat_session, character, user_message, generate_greeting = self._prepare_chat_turn(request)
            result = generate_ai_response(
                user_message.id if user_message else None,
                character.id,
                generate_greeting=generate_greeting,
                chat_session_id=chat_session.id,
            )

            if not result.get('success'):
                return Response(
                    {'error': result.get('error', 'Failed to generate AI response')},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            ai_message = Message.objects.get(id=result['message_id'])
            payload = {
                'ai_message': _message_serializer(ai_message, request).data,
                'chat_session_id': chat_session.id,
            }
            if user_message:
                payload['user_message'] = _message_serializer(user_message, request).data
            return Response(payload)
        except ValueError as exc:
            return Response({'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Character.DoesNotExist:
            return Response({'error': 'Character not found'}, status=status.HTTP_404_NOT_FOUND)
        except ChatSession.DoesNotExist:
            return Response({'error': 'Chat session not found or access denied'}, status=status.HTTP_404_NOT_FOUND)
        except ModelConfiguration.DoesNotExist:
            return Response({'error': 'Model configuration not found or access denied'}, status=status.HTTP_404_NOT_FOUND)

    @action(detail=False, methods=['post'])
    def stream_message(self, request):
        try:
            chat_session, character, user_message, generate_greeting = self._prepare_chat_turn(request)
        except ValueError as exc:
            return Response({'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Character.DoesNotExist:
            return Response({'error': 'Character not found'}, status=status.HTTP_404_NOT_FOUND)
        except ChatSession.DoesNotExist:
            return Response({'error': 'Chat session not found or access denied'}, status=status.HTTP_404_NOT_FOUND)
        except ModelConfiguration.DoesNotExist:
            return Response({'error': 'Model configuration not found or access denied'}, status=status.HTTP_404_NOT_FOUND)

        def event_stream():
            yield self._serialize_stream_event({
                'type': 'session',
                'chat_session_id': chat_session.id,
                'user_message': _message_serializer(user_message, request).data if user_message else None,
                'is_greeting': generate_greeting,
            })

            try:
                for event in stream_ai_response(
                    chat_session=chat_session,
                    character=character,
                    user_message=user_message,
                    generate_greeting=generate_greeting,
                ):
                    yield self._serialize_stream_event(event)
            except Exception as exc:
                logger.exception("Streaming response failed for session %s", chat_session.id)
                yield self._serialize_stream_event({
                    'type': 'error',
                    'error': str(exc),
                })

        return StreamingHttpResponse(event_stream(), content_type='application/x-ndjson')

    def _prepare_chat_turn(self, request):
        user = request.user
        message_content = (request.data.get('message') or '').strip()
        character_id = request.data.get('character_id')
        chat_session_id = request.data.get('chat_session_id')
        start_conversation = self._parse_bool(request.data.get('start_conversation', False))
        origin = request.data.get('origin')
        if origin not in dict(ChatSession.ORIGIN_CHOICES):
            origin = 'topic'
        attachments = list(request.FILES.getlist('attachments'))
        if not attachments:
            legacy_attachment = request.FILES.get('attachment')
            if legacy_attachment:
                attachments = [legacy_attachment]

        if not character_id:
            raise ValueError('character_id is required')

        if not start_conversation and not message_content and not attachments:
            raise ValueError('message or attachment is required')

        attachment_payloads = []
        for index, attachment in enumerate(attachments):
            attachment_kind, attachment_mime_type = guess_attachment_kind(attachment)
            validate_attachment_size(attachment, attachment_kind)
            attachment_text_content = ''
            if attachment_kind == AttachmentKind.TEXT:
                attachment_text_content = extract_text_attachment_content(attachment)
            attachment_payloads.append({
                'file': attachment,
                'attachment_name': attachment.name or '',
                'attachment_kind': attachment_kind,
                'attachment_mime_type': attachment_mime_type,
                'attachment_text_content': attachment_text_content,
                'sort_order': index,
            })

        character = Character.objects.get(id=character_id, created_by=user)
        _get_required_model_config(user)

        if chat_session_id:
            chat_session = ChatSession.objects.get(
                id=chat_session_id,
                user=user,
                character=character,
            )
        else:
            chat_session = ChatSession.objects.create(
                user=user,
                character=character,
                title=f"Chat with {character.name}",
                origin=origin,
            )

        has_existing_messages = chat_session.messages.exists()
        if start_conversation and has_existing_messages:
            raise ValueError('This conversation has already started')

        generate_greeting = start_conversation or not has_existing_messages
        user_message = None
        if not generate_greeting:
            user_message = Message.objects.create(
                chat_session=chat_session,
                role='user',
                content=message_content,
                character=character,
            )
            created_attachments = []
            for payload in attachment_payloads:
                created_attachments.append(
                    MessageAttachment.objects.create(
                        message=user_message,
                        file=payload['file'],
                        attachment_name=payload['attachment_name'],
                        attachment_mime_type=payload['attachment_mime_type'],
                        attachment_kind=payload['attachment_kind'],
                        attachment_text_content=payload['attachment_text_content'],
                        sort_order=payload['sort_order'],
                    )
                )

            if created_attachments:
                primary_attachment = created_attachments[0]
                Message.objects.filter(pk=user_message.pk).update(
                    attachment=primary_attachment.file.name,
                    attachment_name=primary_attachment.attachment_name,
                    attachment_mime_type=primary_attachment.attachment_mime_type,
                    attachment_kind=primary_attachment.attachment_kind,
                    attachment_text_content=primary_attachment.attachment_text_content,
                )
                user_message.refresh_from_db()
            chat_session.save(update_fields=['updated_at'])

        return chat_session, character, user_message, generate_greeting

    def _parse_bool(self, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).lower() in {'1', 'true', 'yes', 'on'}

    def _serialize_stream_event(self, payload):
        return json.dumps(payload, ensure_ascii=False) + '\n'
