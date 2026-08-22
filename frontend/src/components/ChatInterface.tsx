"use client";

import { useEffect, useState } from 'react';
import { RootState, Message, ChatSession, ModelConfig, ModelRoleAssignments, MessageAttachment, UserProfile } from '@/types';
import { useDispatch, useSelector } from 'react-redux';
import { setCharacter, addMessage, setMessages, setLoading, setError, clearChat, setChatSession, upsertMessage, appendToMessage, appendToMessageThinking, appendToMessageToolCall, removeMessage, updateChatSession } from '@/store/chatSlice';
import ImmersiveChatWindow from '@/components/ImmersiveChatWindow';
import ResearchPanel from '@/components/ResearchPanel';
import SoulPanel from '@/components/SoulPanel';
import MemoryPanel from '@/components/MemoryPanel';
import { apiService, normalizeTokenUsage, SendMessageRequest, StreamMessageEvent } from '@/utils/api';
import { AttachmentKind, getAttachmentAvailability } from '@/utils/modelCapabilities';
import { FolderTree, Brain, Globe, Pencil } from 'lucide-react';
import { useI18n } from '@/i18n/provider';

interface ChatInterfaceProps {
  characterId?: string;
  initialSessionId?: string | null;
  modelConfigs: ModelConfig[];
  modelRoles?: ModelRoleAssignments | null;
  defaultModelConfigId?: string | null;
  userProfile?: UserProfile | null;
  sessionOrigin?: 'topic' | 'chat';
  onBack?: () => void;
  onSessionUpdate?: () => void;
  onSoulRefreshKeyChange?: (value: string) => void;
}

interface PendingAttachment {
  file: File;
  kind: AttachmentKind;
  previewUrl?: string;
}

function normalizeStreamMessage(apiMessage: {
  id: string | number;
  content: string;
  role: 'user' | 'assistant';
  timestamp: string;
  thinking?: string | null;
  tool_calls?: Array<{
    tool: string;
    arguments?: Record<string, unknown>;
  }>;
  file_uri?: string | null;
  file_name?: string | null;
  file_preview_url?: string | null;
  file_type?: string | null;
  file_mime_type?: string | null;
  attachments?: Array<{
    file_uri?: string | null;
    file_name?: string | null;
    file_preview_url?: string | null;
    file_type?: string | null;
    file_mime_type?: string | null;
  }>;
}): Message {
  const attachments: MessageAttachment[] = apiMessage.attachments?.length
    ? apiMessage.attachments.map((attachment) => ({
        fileUri: attachment.file_uri || undefined,
        fileName: attachment.file_name || undefined,
        filePreviewUrl: attachment.file_preview_url || undefined,
        fileType: attachment.file_type || undefined,
        fileMimeType: attachment.file_mime_type || undefined,
      }))
    : (apiMessage.file_name || apiMessage.file_uri || apiMessage.file_preview_url || apiMessage.file_type || apiMessage.file_mime_type)
      ? [{
          fileUri: apiMessage.file_uri || undefined,
          fileName: apiMessage.file_name || undefined,
          filePreviewUrl: apiMessage.file_preview_url || undefined,
          fileType: apiMessage.file_type || undefined,
          fileMimeType: apiMessage.file_mime_type || undefined,
        }]
      : [];
  const primaryAttachment = attachments[0];

  return {
    id: String(apiMessage.id),
    content: apiMessage.content || '',
    role: apiMessage.role,
    timestamp: apiMessage.timestamp,
    thinking: apiMessage.thinking || '',
    toolCalls: (apiMessage.tool_calls || [])
      .filter((call) => call?.tool)
      .map((call) => ({
        tool: call.tool,
        arguments: call.arguments || {},
      })),
    attachments,
    fileUri: primaryAttachment?.fileUri || apiMessage.file_uri || undefined,
    fileName: primaryAttachment?.fileName || apiMessage.file_name || undefined,
    filePreviewUrl: primaryAttachment?.filePreviewUrl || apiMessage.file_preview_url || undefined,
    fileType: primaryAttachment?.fileType || apiMessage.file_type || undefined,
    fileMimeType: primaryAttachment?.fileMimeType || apiMessage.file_mime_type || undefined,
  };
}

export default function ChatInterface({
  characterId,
  initialSessionId,
  modelConfigs,
  modelRoles,
  defaultModelConfigId,
  sessionOrigin,
  onSessionUpdate,
  onSoulRefreshKeyChange,
}: ChatInterfaceProps) {
  const { messages: copy } = useI18n();
  const failedToLoadCharacterMessage = copy.chat.failedToLoadCharacter;
  const failedToLoadHistoryMessage = copy.chat.failedToLoadHistory;

  const [showResearchPanel, setShowResearchPanel] = useState(false);
  const [showSoulPanel, setShowSoulPanel] = useState(false);
  const [showMemoryPanel, setShowMemoryPanel] = useState(false);
  const [hasStartedConversation, setHasStartedConversation] = useState(false);
  const [chatSessionId, setChatSessionId] = useState<string | null>(initialSessionId || null);
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState('');

  const dispatch = useDispatch();
  const character = useSelector((state: RootState) => state.chat.character);
  const chatSession = useSelector((state: RootState) => state.chat.chatSession);
  const isLoading = useSelector((state: RootState) => state.chat.isLoading);
  const messages = useSelector((state: RootState) => state.chat.messages);

  useEffect(() => {

    const loadCharacter = async () => {
      if (character && (!characterId || character.id === characterId)) {
        return;
      }

      dispatch(setLoading(true));
      try {
        let serverCharacter;

        if (characterId) {
          const response = await apiService.getCharacter(characterId);
          if (response.data) {
            serverCharacter = response.data;
          } else {
            throw new Error(failedToLoadCharacterMessage);
          }
        } else {
          const response = await apiService.getCharacters();
          if (response.data && response.data.length > 0) {
            serverCharacter = response.data[0];
          }
        }

        if (serverCharacter) {
          dispatch(setCharacter(serverCharacter));
        } else {
          console.error("Fatal Error: Character not found in database.");
        }
      } catch (error) {
        console.error("Failed to load character:", error);
        dispatch(setError(error instanceof Error ? error.message : failedToLoadCharacterMessage));
      } finally {
        dispatch(setLoading(false));
      }
    };

    loadCharacter();
  }, [dispatch, character, characterId, failedToLoadCharacterMessage]);

  useEffect(() => {
    const loadChatHistory = async () => {
      if (!initialSessionId) {
        dispatch(clearChat());
        dispatch(setChatSession(null));
        setChatSessionId(null);
        setHasStartedConversation(false);
        return;
      }

      dispatch(clearChat());
      dispatch(setLoading(true));
      setChatSessionId(initialSessionId);

      try {
        const [messagesRes, sessionRes] = await Promise.all([
          apiService.getMessages(initialSessionId),
          apiService.getChatSession(initialSessionId),
        ]);

        if (messagesRes.data && messagesRes.data.length > 0) {
          dispatch(setMessages(messagesRes.data));
          setHasStartedConversation(true);
        } else {
          setHasStartedConversation(false);
        }

        if (sessionRes.data) {
          dispatch(setChatSession(sessionRes.data));
        }
      } catch (err) {
        console.error("Failed to load chat history:", err);
        dispatch(setError(failedToLoadHistoryMessage));
      } finally {
        dispatch(setLoading(false));
      }
    };

    loadChatHistory();
  }, [initialSessionId, dispatch, failedToLoadHistoryMessage]);

  const syncSessionState = async (sessionId: string) => {
    const response = await apiService.getChatSession(sessionId);
    if (response.data) {
      dispatch(setChatSession(response.data as ChatSession));
    }
  };

  const startEditTitle = () => {
    setTitleDraft(chatSession?.title || '');
    setIsEditingTitle(true);
  };

  const handleSaveTitle = async () => {
    const trimmed = titleDraft.trim();
    setIsEditingTitle(false);

    if (!chatSessionId || !trimmed || trimmed === (chatSession?.title || '')) {
      return;
    }

    try {
      const response = await apiService.updateChatSession(chatSessionId, { title: trimmed });
      if (response.data) {
        dispatch(updateChatSession({ title: response.data.title }));
        onSessionUpdate?.();
      }
    } catch (error) {
      console.error('Failed to update session title:', error);
    }
  };

  const handleSendMessage = async (userInput: string, attachments: PendingAttachment[] = []) => {
    if (!character) return;

    const isFirstMessage = !hasStartedConversation;
    const trimmedInput = userInput.trim();
    const streamingAssistantId = `stream-${Date.now()}`;
    const optimisticUserMessageId = `local-user-${Date.now()}`;
    const previousStartedState = hasStartedConversation;
    const currentUserLabel = copy.chat.you;
    const optimisticAttachments: MessageAttachment[] = attachments.map((attachment) => ({
      fileName: attachment.file.name,
      fileType: attachment.kind,
      fileMimeType: attachment.file.type || undefined,
      filePreviewUrl:
        attachment.kind === 'image' || attachment.kind === 'video'
          ? URL.createObjectURL(attachment.file)
          : undefined,
    }));
    const primaryOptimisticAttachment = optimisticAttachments[0];

    if (!isFirstMessage && !trimmedInput && attachments.length === 0) {
      return;
    }

    if (isFirstMessage) {
      setHasStartedConversation(true);
    } else {
      dispatch(addMessage({
        id: optimisticUserMessageId,
        content: trimmedInput,
        role: 'user',
        timestamp: new Date().toISOString(),
        senderId: 'user',
        senderName: currentUserLabel,
        senderType: 'user',
        attachments: optimisticAttachments,
        fileName: primaryOptimisticAttachment?.fileName,
        fileType: primaryOptimisticAttachment?.fileType,
        fileMimeType: primaryOptimisticAttachment?.fileMimeType,
        filePreviewUrl: primaryOptimisticAttachment?.filePreviewUrl,
        fileUri: primaryOptimisticAttachment?.filePreviewUrl,
      }));
    }

    dispatch(upsertMessage({
      id: streamingAssistantId,
      content: '',
      role: 'assistant',
      timestamp: new Date().toISOString(),
      senderId: character.id,
      senderName: character.name,
      senderAvatarUrl: character.avatarUrl,
      senderType: 'character',
    }));

    dispatch(setLoading(true));
    dispatch(setError(null));

    try {
      const requestData: SendMessageRequest = {
        message: isFirstMessage ? '' : trimmedInput,
        character_id: character.id,
        chat_session_id: chatSessionId || undefined,
        start_conversation: isFirstMessage,
        origin: sessionOrigin,
        attachments: attachments.map((attachment) => attachment.file),
      };

      const response = await apiService.streamMessage(requestData, {
        onEvent: (event: StreamMessageEvent) => {
          if (event.type === 'session') {
            setChatSessionId(String(event.chat_session_id));
            if (event.user_message) {
              optimisticAttachments.forEach((attachment) => {
                if (attachment.filePreviewUrl?.startsWith('blob:')) {
                  URL.revokeObjectURL(attachment.filePreviewUrl);
                }
              });
              dispatch(removeMessage(optimisticUserMessageId));
              dispatch(addMessage({
                ...normalizeStreamMessage(event.user_message),
                senderId: 'user',
                senderName: currentUserLabel,
                senderType: 'user',
              }));
            }
            return;
          }

          if (event.type === 'delta') {
            dispatch(appendToMessage({
              id: streamingAssistantId,
              content: event.content,
            }));
            return;
          }

          if (event.type === 'thinking') {
            dispatch(appendToMessageThinking({
              id: streamingAssistantId,
              content: event.content,
            }));
            return;
          }

          if (event.type === 'tool') {
            dispatch(appendToMessageToolCall({
              id: streamingAssistantId,
              toolCall: {
                tool: event.tool,
                arguments: event.arguments || {},
              },
            }));
            return;
          }

          if (event.type === 'done') {
            dispatch(removeMessage(streamingAssistantId));
            dispatch(addMessage({
              id: String(event.message_id),
              content: event.content,
              role: 'assistant',
              timestamp: event.timestamp,
              senderId: character.id,
              senderName: character.name,
              senderAvatarUrl: character.avatarUrl,
              senderType: 'character',
              thinking: event.thinking || '',
              toolCalls: (event.tool_calls || [])
                .filter((call) => call?.tool)
                .map((call) => ({
                  tool: call.tool,
                  arguments: call.arguments || {},
                })),
              tokenUsage: normalizeTokenUsage(event.token_usage),
              researchPayload: event.research_payload ? {
                query: event.research_payload.query || '',
                provider: event.research_payload.provider || '',
                items: (event.research_payload.items || []).filter((item) => item?.url).map((item) => ({
                  title: item?.title || item?.url || copy.chat.untitled,
                  url: item?.url || '',
                  snippet: item?.snippet || '',
                  domain: item?.domain || '',
                  source: item?.source || '',
                })),
                error: event.research_payload.error || '',
              } : null,
            }));
            return;
          }

          if (event.type === 'error') {
            throw new Error(event.error);
          }
        },
      });

      if (response.error) {
        throw new Error(response.error);
      }

      if (response.data?.chat_session_id) {
        setChatSessionId(response.data.chat_session_id);
        await syncSessionState(response.data.chat_session_id);

        if (onSessionUpdate) {
          onSessionUpdate();
          setTimeout(() => {
            onSessionUpdate();
          }, 1500);
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      optimisticAttachments.forEach((attachment) => {
        if (attachment.filePreviewUrl?.startsWith('blob:')) {
          URL.revokeObjectURL(attachment.filePreviewUrl);
        }
      });
      dispatch(removeMessage(optimisticUserMessageId));
      dispatch(removeMessage(streamingAssistantId));
      if (!previousStartedState) {
        setHasStartedConversation(false);
      }

      const errorMessageContent = error instanceof Error ? error.message : copy.chat.failedToGetResponse;
      dispatch(setError(errorMessageContent));

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: copy.chat.sorryEncounteredError(errorMessageContent),
        role: 'assistant',
        timestamp: new Date().toISOString(),
        senderId: character.id,
        senderName: character.name,
        senderAvatarUrl: character.avatarUrl,
        senderType: 'character',
      };
      dispatch(addMessage(errorMessage));
    } finally {
      dispatch(setLoading(false));
    }
  };

  const activeModelConfig =
    modelRoles?.text ||
    modelConfigs.find((config) => config.id === defaultModelConfigId) ||
    null;
  const attachmentSupport = getAttachmentAvailability(modelRoles, activeModelConfig);
  const localizedMediaMode = (mode: 'analyzed' | 'native' | 'unavailable') =>
    ({
      analyzed: copy.modelApi.attachmentModes.analyzed,
      native: copy.modelApi.attachmentModes.native,
      unavailable: copy.modelApi.attachmentModes.unavailable,
    }[mode]);
  const latestResearchMessage = [...messages].reverse().find(
    (message) => message.role === 'assistant' && message.researchPayload
  ) || null;
  const soulRefreshKey = `${chatSession?.updatedAt || 'no-session'}:${latestResearchMessage?.id || 'no-research'}`;

  useEffect(() => {
    onSoulRefreshKeyChange?.(soulRefreshKey);
  }, [onSoulRefreshKeyChange, soulRefreshKey]);

  return (
    <div className="flex h-full flex-col bg-[linear-gradient(180deg,#f8fbff_0%,#eef4f8_52%,#f4efe8_100%)]">
      <header className="flex flex-shrink-0 items-center justify-between gap-3 border-b border-slate-200/70 bg-white/75 px-4 py-3 backdrop-blur-xl md:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center overflow-hidden rounded-[1rem] bg-gradient-to-br from-sky-100 via-cyan-50 to-amber-50 shadow-sm ring-1 ring-white/70">
            {character?.avatarUrl ? (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img src={character.avatarUrl} alt={character.name} className="h-full w-full object-cover" />
            ) : (
              <span className="text-base font-semibold text-sky-700">
                {character?.name?.charAt(0) || 'C'}
              </span>
            )}
          </div>
          <div className="min-w-0">
            <h2 className="truncate text-base font-semibold tracking-tight text-slate-900">
              {character?.name || copy.chat.loadingCharacter}
            </h2>
            {sessionOrigin !== 'chat' && chatSessionId && isEditingTitle && (
              <input
                type="text"
                value={titleDraft}
                onChange={(event) => setTitleDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    void handleSaveTitle();
                  } else if (event.key === 'Escape') {
                    setIsEditingTitle(false);
                  }
                }}
                onBlur={() => void handleSaveTitle()}
                autoFocus
                placeholder={copy.chat.titlePlaceholder}
                className="mt-0.5 w-56 max-w-full rounded-lg border border-sky-200 bg-white px-2 py-0.5 text-xs text-slate-700 outline-none focus:border-sky-400 focus:ring-2 focus:ring-sky-100"
              />
            )}
            {sessionOrigin !== 'chat' && chatSessionId && !isEditingTitle && (
              <button
                type="button"
                onClick={startEditTitle}
                className="group mt-0.5 flex max-w-full items-center gap-1 text-left"
                title={copy.chat.editTitle}
              >
                <span className="truncate text-xs text-slate-400 transition-colors group-hover:text-slate-600">
                  {chatSession?.title || copy.chat.untitled}
                </span>
                <Pencil size={11} className="flex-shrink-0 text-slate-400 opacity-60 transition-opacity group-hover:opacity-100" />
              </button>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2 lg:hidden">
          <button
            onClick={() => setShowSoulPanel((prev) => !prev)}
            className={`flex h-10 w-10 items-center justify-center rounded-2xl transition-colors ${
              showSoulPanel
                ? 'bg-sky-50 text-sky-700 hover:bg-sky-100'
                : 'bg-white/80 text-slate-500 ring-1 ring-slate-200 hover:bg-white hover:text-slate-900'
            }`}
            title={copy.chat.toggleSoulPanel}
          >
            <FolderTree className="h-5 w-5" />
          </button>
          <button
            onClick={() => setShowMemoryPanel((prev) => !prev)}
            className={`flex h-10 w-10 items-center justify-center rounded-2xl transition-colors ${
              showMemoryPanel
                ? 'bg-sky-50 text-sky-700 hover:bg-sky-100'
                : 'bg-white/80 text-slate-500 ring-1 ring-slate-200 hover:bg-white hover:text-slate-900'
            }`}
            title={copy.chat.toggleMemoryPanel}
          >
            <Brain className="h-5 w-5" />
          </button>
          <button
            onClick={() => setShowResearchPanel((prev) => !prev)}
            className={`flex h-10 w-10 items-center justify-center rounded-2xl transition-colors ${
              showResearchPanel
                ? 'bg-sky-50 text-sky-700 hover:bg-sky-100'
                : 'bg-white/80 text-slate-500 ring-1 ring-slate-200 hover:bg-white hover:text-slate-900'
            }`}
            title={copy.chat.toggleResearchPanel}
          >
            <Globe className="h-5 w-5" />
          </button>
        </div>
      </header>

      <div className="flex-1 overflow-hidden px-4 pb-4 pt-4 md:px-6 md:pb-6">
        <ImmersiveChatWindow
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          isFirstMessage={!hasStartedConversation}
          currentUserLabel={copy.chat.you}
          attachmentSupport={attachmentSupport}
          localizedMediaMode={localizedMediaMode}
          contextWindowTokens={activeModelConfig?.contextWindow || null}
        />
      </div>

      {showSoulPanel && character && (
        <div
          className="fixed inset-0 z-40 bg-slate-950/35 p-3 backdrop-blur-sm lg:hidden"
          onClick={() => setShowSoulPanel(false)}
        >
          <div className="ml-auto h-full w-full max-w-md" onClick={(event) => event.stopPropagation()}>
            <SoulPanel
              characterId={character.id}
              characterName={character.name}
              refreshKey={soulRefreshKey}
              isOpen
              isMobile
              onToggle={() => setShowSoulPanel(false)}
            />
          </div>
        </div>
      )}

      {showMemoryPanel && character && (
        <div
          className="fixed inset-0 z-40 bg-slate-950/35 p-3 backdrop-blur-sm lg:hidden"
          onClick={() => setShowMemoryPanel(false)}
        >
          <div className="ml-auto h-full w-full max-w-md" onClick={(event) => event.stopPropagation()}>
            <MemoryPanel
              characterId={character.id}
              chatSessionId={chatSessionId}
              refreshKey={soulRefreshKey}
              onPrivateModeChanged={(isPrivateMode) => {
                if (chatSession && isPrivateMode !== chatSession.isPrivateMode) {
                  dispatch(updateChatSession({ isPrivateMode }));
                }
              }}
            />
          </div>
        </div>
      )}

      {showResearchPanel && character && (
        <div
          className="fixed inset-0 z-40 bg-slate-950/35 p-3 backdrop-blur-sm lg:hidden"
          onClick={() => setShowResearchPanel(false)}
        >
          <div className="ml-auto h-full w-full max-w-md" onClick={(event) => event.stopPropagation()}>
            <ResearchPanel />
          </div>
        </div>
      )}
    </div>
  );
}
