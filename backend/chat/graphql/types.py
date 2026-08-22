import strawberry
from typing import List, Optional
import os
from asgiref.sync import sync_to_async
import strawberry_django
from chat.models import AttachmentKind, Character, ChatSession, CharacterKnowledgeAsset


@strawberry.type
class PrisMateDraft:
    name: str
    description: str
    personality: str
    appearance: str
    affiliation: str
    tags: List[str]
    visual_summary: str
    example_dialogue: str = ""

@strawberry.input
class CharacterKnowledgeAssetInput:
    uploaded_url: str
    file_name: str


@strawberry.input
class CharacterInput:
    name: str
    avatar_url: str
    description: str
    user_address: Optional[str] = ""
    personality: Optional[str] = ""
    appearance: Optional[str] = ""
    response_guidelines: Optional[str] = ""
    scenario: str
    example_dialogue: str
    affiliation: Optional[str] = ""
    system_prompt_preview: Optional[str] = ""
    tags: List[str]
    background_file_url: Optional[str] = ""
    background_file_name: Optional[str] = ""
    background_files: Optional[List[CharacterKnowledgeAssetInput]] = None


@strawberry.type
class CharacterKnowledgeAssetType:
    file_url: str
    file_name: str
    file_type: str
    file_mime_type: str


def _serialize_character_knowledge_asset(asset: CharacterKnowledgeAsset) -> CharacterKnowledgeAssetType:
    return CharacterKnowledgeAssetType(
        file_url=asset.file.url if asset.file else "",
        file_name=asset.attachment_name or os.path.basename(asset.file.name or ""),
        file_type=asset.attachment_kind or "",
        file_mime_type=asset.attachment_mime_type or "",
    )


def _primary_text_knowledge_asset(character: Character) -> Optional[CharacterKnowledgeAsset]:
    return character.knowledge_assets.filter(
        attachment_kind=AttachmentKind.TEXT,
    ).order_by('sort_order', 'id').first()

@strawberry_django.type(Character)
class CharacterType:
    id: strawberry.ID
    name: str
    avatar_url: Optional[str]
    description: str
    user_address: str
    personality: Optional[str]
    appearance: Optional[str]
    response_guidelines: Optional[str]
    scenario: str
    example_dialogue: str
    affiliation: str
    system_prompt_preview: str
    tags: List[str]

    @strawberry.field
    async def background_file_url(self) -> Optional[str]:
        asset = await sync_to_async(_primary_text_knowledge_asset)(self)
        if asset and asset.file:
            try:
                return asset.file.url
            except ValueError:
                return None

        # Legacy fallback for characters that only carry a `Character.file` row.
        if self.file:
            try:
                return self.file.url
            except ValueError:
                return None
        return None

    @strawberry.field
    async def background_file_name(self) -> Optional[str]:
        asset = await sync_to_async(_primary_text_knowledge_asset)(self)
        if asset and asset.file:
            return asset.attachment_name or os.path.basename(asset.file.name or "")

        if self.file:
            return os.path.basename(self.file.name or "")
        return None

    @strawberry.field
    async def knowledge_assets(self) -> List[CharacterKnowledgeAssetType]:
        assets = await sync_to_async(list)(self.knowledge_assets.all())
        if assets:
            return [_serialize_character_knowledge_asset(asset) for asset in assets]

        if not self.file:
            return []

        return [
            CharacterKnowledgeAssetType(
                file_url=self.file.url,
                file_name=os.path.basename(self.file.name or ""),
                file_type='text',
                file_mime_type='text/plain',
            )
        ]

@strawberry_django.type(ChatSession)
class ChatSessionType:
    id: strawberry.ID
    title: str
    last_response_latency_ms: Optional[int]
    character: CharacterType
