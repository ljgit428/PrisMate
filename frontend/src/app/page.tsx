"use client";

import { Suspense, useState, useEffect, useCallback } from 'react';
import { useDispatch } from 'react-redux';
import { useSearchParams, useRouter } from 'next/navigation';
import {
  Home,
  Zap,
  PlusCircle,
  KeyRound,
  UserCircle2,
  ChevronRight,
  ChevronLeft,
  Menu,
  MessageSquare,
  ArrowRight,
  Trash2,
  Search,
  X,
  FolderTree,
  Database,
  Brain,
  Globe,
  LogOut,
  MessagesSquare,
  Pencil,
} from 'lucide-react';
import CharacterGallery from '@/components/CharacterGallery';
import CreateCharacterSimplifiedForm from '@/components/CreateCharacterSimplifiedForm';
import ChatInterface from '@/components/ChatInterface';
import ModelApiSettingsPanel from '@/components/ModelApiSettingsPanel';
import UserSettingsPanel from '@/components/UserSettingsPanel';
import SoulPanel from '@/components/SoulPanel';
import MemoryPanel from '@/components/MemoryPanel';
import ResearchPanel from '@/components/ResearchPanel';
import { useI18n } from '@/i18n/provider';
import { clearChat, updateChatSession } from '@/store/chatSlice';
import { apiService, getAuthToken, removeAuthToken } from '@/utils/api';

type RightPanelKind = 'soul' | 'memory' | 'research';
import { ChatSession, ModelConfig, ModelRoleAssignments, UserProfile, WebSearchConfig } from '@/types';
import { APP_NAME, APP_VERSION } from '@/constants';

type ViewState = 'home' | 'playground' | 'history_all' | 'create' | 'model_settings' | 'user_settings';

interface ChatHistoryItem {
  id: string;
  title: string;
  characterId: string;
  characterName?: string;
  created_at: string;
  updated_at: string;
}

function AIStudioLayoutContent() {
  const { setLocale, messages, formatDate, formatTime } = useI18n();
  const [currentView, setCurrentView] = useState<ViewState>('playground');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isRightPanelOpen, setIsRightPanelOpen] = useState(false);
  const [rightPanelKind, setRightPanelKind] = useState<RightPanelKind>('soul');
  const [rightPanelWidth, setRightPanelWidth] = useState(420);
  const [isResizingRightPanel, setIsResizingRightPanel] = useState(false);
  const [selectedCharacterId, setSelectedCharacterId] = useState<string | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [soulRefreshKey, setSoulRefreshKey] = useState('shell');
  const [recentChats, setRecentChats] = useState<ChatHistoryItem[]>([]);
  const [modelConfigs, setModelConfigs] = useState<ModelConfig[]>([]);
  const [modelRoles, setModelRoles] = useState<ModelRoleAssignments | null>(null);
  const [loadingModelRoles, setLoadingModelRoles] = useState(true);
  const [loadingChats, setLoadingChats] = useState(true);
  const [loadingModelConfigs, setLoadingModelConfigs] = useState(true);
  const [loadingUserProfile, setLoadingUserProfile] = useState(true);
  const [loadingWebSearchConfig, setLoadingWebSearchConfig] = useState(true);
  const [chatError, setChatError] = useState<string | null>(null);
  const [modelConfigError, setModelConfigError] = useState<string | null>(null);
  const [userProfileError, setUserProfileError] = useState<string | null>(null);
  const [webSearchConfigError, setWebSearchConfigError] = useState<string | null>(null);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [webSearchConfig, setWebSearchConfig] = useState<WebSearchConfig | null>(null);
  const [authTokenPresent, setAuthTokenPresent] = useState(false);

  const [historyPage, setHistoryPage] = useState(1);
  const [historySearchQuery, setHistorySearchQuery] = useState('');
  const [renamingChat, setRenamingChat] = useState<ChatHistoryItem | null>(null);
  const [renameDraft, setRenameDraft] = useState('');
  const [renameError, setRenameError] = useState<string | null>(null);
  const ITEMS_PER_PAGE = 10;

  const dispatch = useDispatch();

  const searchParams = useSearchParams();
  const router = useRouter();
  const autoSelectId = searchParams.get('select');

  useEffect(() => {
    if (autoSelectId) {
      setSelectedCharacterId(autoSelectId);
      setCurrentView('playground');
      setSelectedSessionId(null);
      dispatch(clearChat());

      router.replace('/');
    }
  }, [autoSelectId, dispatch, router]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      setAuthTokenPresent(Boolean(getAuthToken()));
    }
  }, []);

  const formatChatSessions = useCallback((sessions: ChatSession[]) => {
    return sessions.map((session) => {
      const charId = session.character?.id || '';

      return {
        id: session.id,
        title: session.title || `${messages.chat.untitled} #${session.id}`,
        characterId: charId,
        characterName: session.character?.name || '',
        created_at: session.createdAt,
        updated_at: session.updatedAt
      };
    }).sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime());
  }, [messages.chat.untitled]);

  const fetchChatSessions = useCallback(async () => {
    try {
      setLoadingChats(true);
      const response = await apiService.getChatSessions(undefined, 'topic');

      if (response.error) {
        setChatError(response.error);
        return;
      }

      if (response.data) {
        setChatError(null);
        setRecentChats(formatChatSessions(response.data));
      }
    } catch (err) {
      console.error('Failed to fetch chat sessions:', err);
      setChatError(messages.shell.failedToLoadConversationHistory);
    } finally {
      setLoadingChats(false);
    }
  }, [formatChatSessions, messages.shell.failedToLoadConversationHistory]);

  const fetchModelConfigs = useCallback(async () => {
    try {
      setLoadingModelConfigs(true);
      const response = await apiService.getModelConfigs();

      if (response.error) {
        setModelConfigError(response.error);
        return;
      }

      setModelConfigError(null);
      setModelConfigs(response.data || []);
    } catch (err) {
      console.error('Failed to fetch model configurations:', err);
      setModelConfigError(messages.modelApi.failedToLoadModelConfigurations);
    } finally {
      setLoadingModelConfigs(false);
    }
  }, [messages.modelApi.failedToLoadModelConfigurations]);

  const fetchModelRoles = useCallback(async () => {
    try {
      setLoadingModelRoles(true);
      const response = await apiService.getModelRoles();

      if (response.error) {
        setModelConfigError(response.error);
        return;
      }

      setModelRoles(response.data || null);
    } catch (err) {
      console.error('Failed to fetch model role assignments:', err);
      setModelConfigError(messages.modelApi.failedToLoadRoles);
    } finally {
      setLoadingModelRoles(false);
    }
  }, [messages.modelApi.failedToLoadRoles]);

  const fetchUserProfile = useCallback(async () => {
    try {
      setLoadingUserProfile(true);
      const response = await apiService.getUserProfile();

      if (response.error) {
        setUserProfileError(response.error);
        return;
      }

      setUserProfileError(null);
      setUserProfile(response.data || null);
    } catch (err) {
      console.error('Failed to fetch user profile:', err);
      setUserProfileError(messages.user.failedToLoad);
    } finally {
      setLoadingUserProfile(false);
    }
  }, [messages.user.failedToLoad]);

  const fetchWebSearchConfig = useCallback(async () => {
    try {
      setLoadingWebSearchConfig(true);
      const response = await apiService.getWebSearchConfig();

      if (response.error) {
        setWebSearchConfigError(response.error);
        return;
      }

      setWebSearchConfigError(null);
      setWebSearchConfig(response.data || null);
    } catch (err) {
      console.error('Failed to fetch web search configuration:', err);
      setWebSearchConfigError(messages.modelApi.failedToLoadWebSearchConfiguration);
    } finally {
      setLoadingWebSearchConfig(false);
    }
  }, [messages.modelApi.failedToLoadWebSearchConfiguration]);

  useEffect(() => {
    fetchChatSessions();
    fetchModelConfigs();
    fetchModelRoles();
    fetchUserProfile();
    fetchWebSearchConfig();
  }, [fetchChatSessions, fetchModelConfigs, fetchModelRoles, fetchUserProfile, fetchWebSearchConfig]);

  useEffect(() => {
    if (userProfile?.interfaceLanguage) {
      setLocale(userProfile.interfaceLanguage);
    }
  }, [setLocale, userProfile?.interfaceLanguage]);

  useEffect(() => {
    if (!isResizingRightPanel) {
      return undefined;
    }

    const handlePointerMove = (event: MouseEvent) => {
      const viewportWidth = window.innerWidth;
      const nextWidth = viewportWidth - event.clientX;
      setRightPanelWidth(Math.max(320, Math.min(680, nextWidth)));
    };

    const handlePointerUp = () => {
      setIsResizingRightPanel(false);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    window.addEventListener('mousemove', handlePointerMove);
    window.addEventListener('mouseup', handlePointerUp);

    return () => {
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', handlePointerMove);
      window.removeEventListener('mouseup', handlePointerUp);
    };
  }, [isResizingRightPanel]);

  const handleSelectCharacter = (characterId: string) => {
    dispatch(clearChat());
    setSelectedCharacterId(characterId);
    setSelectedSessionId(null);
    setCurrentView('playground');
  };

  const handleBackToGallery = () => {
    setSelectedCharacterId(null);
    setCurrentView('playground');
  };

  const handleSelectHistoryItem = (characterId: string, sessionId: string) => {
    setSelectedCharacterId(characterId);
    setSelectedSessionId(String(sessionId));
    setCurrentView('playground');
  };


  const handleDeleteSession = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    if (!window.confirm(messages.shell.confirmDeleteConversationHistory)) {
      return;
    }

    try {
      await apiService.deleteChatSession(sessionId);
      const response = await apiService.getChatSessions(undefined, 'topic');

      if (response.data) {
        setRecentChats(formatChatSessions(response.data));
      }
    } catch (error) {
      console.error("Failed to delete session:", error);
      alert(messages.shell.failedToDeleteSession);
    }
  };

  const handleRenameSession = (e: React.MouseEvent, chat: ChatHistoryItem) => {
    e.stopPropagation();
    setRenameDraft(chat.title);
    setRenameError(null);
    setRenamingChat(chat);
  };

  const submitRename = async () => {
    const chat = renamingChat;
    const trimmed = renameDraft.trim();
    if (!chat) {
      return;
    }

    if (!trimmed || trimmed === chat.title) {
      setRenamingChat(null);
      return;
    }

    try {
      const response = await apiService.updateChatSession(chat.id, { title: trimmed });
      if (response.error || !response.data) {
        throw new Error(response.error || messages.shell.renameTopicFailed);
      }

      if (selectedSessionId === chat.id) {
        dispatch(updateChatSession({ title: response.data.title }));
      }
      await fetchChatSessions();
      setRenamingChat(null);
    } catch (error) {
      console.error("Failed to rename session:", error);
      setRenameError(messages.shell.renameTopicFailed);
    }
  };

  const defaultModelConfigId = modelRoles?.text?.id || modelConfigs[0]?.id || null;
  const hasModelConfigs = Boolean(modelRoles?.text) || modelConfigs.length > 0;

  const isCharacterInPlayground = Boolean(selectedCharacterId) && currentView === 'playground';

  const handleRightPanelIconClick = (kind: RightPanelKind) => {
    if (isRightPanelOpen && rightPanelKind === kind) {
      setIsRightPanelOpen(false);
      return;
    }
    setRightPanelKind(kind);
    setIsRightPanelOpen(true);
  };

  const handleLogout = () => {
    removeAuthToken();
    if (typeof window !== 'undefined') {
      window.location.reload();
    }
  };

  const openModelSettings = () => {
    setCurrentView('model_settings');
  };

  const normalizedHistorySearchQuery = historySearchQuery.trim().toLowerCase();
  const filteredHistoryItems = recentChats.filter((chat) => {
    if (!normalizedHistorySearchQuery) {
      return true;
    }

    return [chat.title, chat.characterName, chat.characterId, chat.id]
      .filter(Boolean)
      .join(' ')
      .toLowerCase()
      .includes(normalizedHistorySearchQuery);
  });
  const totalHistoryPages = Math.max(1, Math.ceil(filteredHistoryItems.length / ITEMS_PER_PAGE));

  useEffect(() => {
    if (historyPage > totalHistoryPages) {
      setHistoryPage(totalHistoryPages);
    }
  }, [historyPage, totalHistoryPages]);

  return (
    <div className="flex h-screen w-full bg-[linear-gradient(180deg,#f5f8fc_0%,#eef3f7_50%,#f6efe8_100%)] text-slate-700">
      <aside
        className={`${isSidebarOpen ? 'w-[300px]' : 'w-0'} flex-shrink-0 overflow-hidden border-r border-white/60 bg-[linear-gradient(180deg,rgba(255,255,255,0.88),rgba(244,247,250,0.92))] backdrop-blur-xl transition-all duration-300 ease-in-out`}
      >
        <div className="flex h-20 flex-shrink-0 items-center border-b border-slate-200/70 px-5">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="rounded-full bg-slate-900 px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.16em] text-white">
                {messages.shell.studioBadge}
              </span>
              <span className="text-[11px] font-medium uppercase tracking-[0.16em] text-slate-400">
                {APP_VERSION}
              </span>
            </div>
            <div className="font-semibold text-xl tracking-tight text-slate-900">
              <span className="text-sky-700">{APP_NAME}</span>
            </div>
            <p className="text-xs text-slate-500">{messages.shell.characterWorkbenchArchive}</p>
          </div>
        </div>
        <div className="flex-1 space-y-6 overflow-y-auto px-4 py-5">
          <div className="space-y-1">
            <NavItem
              icon={<Home size={18} />}
              label={messages.shell.home}
              active={currentView === 'home'}
              onClick={() => setCurrentView('home')}
            />
          </div>
          <div className="rounded-[1.5rem] border border-white/70 bg-white/60 p-3 shadow-[0_16px_50px_rgba(15,23,42,0.06)]">
            <div className="mb-2 flex items-center justify-between px-1">
              <div>
                <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-slate-400">{messages.shell.sessions}</p>
              </div>
            </div>
            <div className="mb-1">
              <NavItem
                icon={<Zap size={18} />}
                label={messages.shell.playground}
                active={currentView === 'playground'}
                onClick={() => {
                  setCurrentView('playground');
                  setSelectedCharacterId(null);
                  setSelectedSessionId(null);
                }}
                isPrimary
              />
            </div>
            <div className="mt-3 space-y-1 border-l border-slate-200 pl-3">
              {loadingChats && (
                <div className="px-3 py-2 text-sm text-slate-500">
                  {messages.shell.loadingConversationHistory}
                </div>
              )}
              {chatError && (
                <div className="px-3 py-2 text-sm text-rose-500">
                  {chatError}
                </div>
              )}
              {!loadingChats && !chatError && recentChats.length > 0 && recentChats.slice(0, 5).map(chat => (
                <div key={chat.id} className="group relative flex items-center">
                  <button
                    onClick={() => handleSelectHistoryItem(chat.characterId, chat.id)}
                    className={`flex w-full items-center gap-2 truncate rounded-xl px-3 py-2 pr-8 text-left text-sm transition-colors ${selectedSessionId === chat.id && currentView === 'playground'
                      ? 'bg-sky-50 text-sky-700 ring-1 ring-sky-100'
                      : 'text-slate-500 hover:bg-white/80 hover:text-slate-900'
                      }`}
                    title={chat.title}
                  >
                    <MessageSquare size={14} className={`flex-shrink-0 ${selectedSessionId === chat.id ? 'text-sky-600' : 'opacity-70'}`} />
                    <span className="truncate">{chat.title}</span>
                  </button>
                  <button
                    onClick={(e) => handleRenameSession(e, chat)}
                    className="absolute right-7 rounded-lg p-1 text-slate-400 opacity-0 transition-all hover:bg-sky-50 hover:text-sky-600 group-hover:opacity-100"
                    title={messages.shell.renameTopic}
                  >
                    <Pencil size={12} />
                  </button>
                  <button
                    onClick={(e) => handleDeleteSession(e, chat.id)}
                    className="absolute right-1 rounded-lg p-1 text-slate-400 opacity-0 transition-all hover:bg-rose-50 hover:text-rose-500 group-hover:opacity-100"
                    title={messages.shell.deleteChat}
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              ))}
              <button
                onClick={() => {
                  setCurrentView('history_all');
                  setSelectedSessionId(null);
                }}
                className={`mt-2 flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-sm font-medium transition-colors group ${currentView === 'history_all'
                  ? 'bg-sky-50 text-sky-700 ring-1 ring-sky-100'
                  : 'text-slate-600 hover:bg-sky-50 hover:text-sky-700'
                  }`}
              >
                <span>{messages.shell.viewAllHistory}</span>
                <ArrowRight size={14} className={`transition-opacity ${currentView === 'history_all' ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`} />
              </button>
            </div>
          </div>
          <div className="rounded-[1.5rem] border border-white/70 bg-white/50 p-3 shadow-[0_16px_50px_rgba(15,23,42,0.04)]">
            <div className="mb-2 px-1">
              <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-slate-400">{messages.shell.workspace}</p>
            </div>
            <div className="space-y-1">
              <button
                type="button"
                onClick={() => router.push('/chat')}
                className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-slate-600 transition-all duration-200 hover:bg-white/80 hover:text-slate-900"
              >
                <span className="text-slate-400"><MessagesSquare size={18} /></span>
                <span className="min-w-0 flex-1 truncate text-left">{messages.chatPage.pageTitle}</span>
              </button>
              <button
                type="button"
                onClick={() => router.push('/memory')}
                className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-slate-600 transition-all duration-200 hover:bg-white/80 hover:text-slate-900"
              >
                <span className="text-slate-400"><Database size={18} /></span>
                <span className="min-w-0 flex-1 truncate text-left">{messages.memory.pageTitle}</span>
              </button>
              <NavItem
                icon={<PlusCircle size={18} />}
                label={messages.shell.newCharacter}
                active={currentView === 'create'}
                onClick={() => setCurrentView('create')}
              />
              <NavItem
                icon={<KeyRound size={18} />}
                label={messages.shell.modelApiSettings}
                active={currentView === 'model_settings'}
                onClick={openModelSettings}
                badge={hasModelConfigs ? messages.shell.modelApiReady : messages.shell.modelApiRequired}
                badgeTone={hasModelConfigs ? 'ready' : 'warning'}
              />
              <NavItem
                icon={<UserCircle2 size={18} />}
                label={messages.shell.userSettings}
                active={currentView === 'user_settings'}
                onClick={() => setCurrentView('user_settings')}
              />
            </div>
          </div>

        </div>
      </aside>
      <div className="flex min-w-0 flex-1">
        <main className="relative flex h-full min-w-0 flex-1 flex-col overflow-hidden bg-[linear-gradient(180deg,rgba(255,255,255,0.5),rgba(248,250,252,0.7))]">
          {!isSidebarOpen && (
            <div className="absolute top-4 left-4 z-10">
              <button
                onClick={() => setIsSidebarOpen(true)}
                className="rounded-xl bg-white/90 p-2.5 text-slate-600 shadow-sm ring-1 ring-slate-200 transition-colors hover:bg-white hover:text-slate-900"
              >
                <span className="sr-only">Open sidebar</span>
                <Menu size={20} />
              </button>
            </div>
          )}
          <header className="flex h-16 flex-shrink-0 items-center justify-between border-b border-white/70 bg-white/65 px-5 backdrop-blur-xl">
            <div className="flex items-center gap-3">
              <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="rounded-xl p-2 text-slate-500 transition-colors hover:bg-white/80 hover:text-slate-900">
                <span className="sr-only">Toggle sidebar</span>
                <Menu size={20} />
              </button>
              <div>
                <div className="text-sm font-medium text-slate-700">
                {currentView === 'home' && messages.shell.home}
                {currentView === 'playground' && messages.shell.playgroundChat}
                {currentView === 'create' && messages.shell.buildCreateNew}
                {currentView === 'history_all' && messages.shell.historyAll}
                {currentView === 'model_settings' && messages.shell.modelApiSettingsHeader}
                {currentView === 'user_settings' && messages.shell.userSettingsHeader}
                </div>
                <div className="text-xs text-slate-400">
                  {selectedCharacterId ? messages.shell.focusedOnOneActiveCharacterSession : messages.shell.browseCharactersSessionsAndSettings}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {authTokenPresent && (
                <button
                  type="button"
                  onClick={handleLogout}
                  className="hidden rounded-xl p-2 text-slate-500 transition-colors hover:bg-white/80 hover:text-slate-900 lg:flex"
                  title={messages.auth.logout}
                >
                  <LogOut size={18} />
                </button>
              )}
              <div className="rounded-full bg-white/80 px-3 py-1.5 text-xs text-slate-500 ring-1 ring-slate-200">{APP_VERSION}</div>
            </div>
          </header>
          <div className="flex-1 overflow-hidden relative">
            {renderContent(currentView)}
          </div>
        </main>

        {isCharacterInPlayground && (
          <div
            className="relative hidden h-full flex-shrink-0 border-l border-white/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.72),rgba(244,247,250,0.92))] lg:block"
            style={{ width: isRightPanelOpen ? `${rightPanelWidth}px` : '76px' }}
          >
            {isRightPanelOpen && (
              <button
                type="button"
                onMouseDown={(event) => {
                  event.preventDefault();
                  setIsResizingRightPanel(true);
                }}
                className="absolute left-0 top-1/2 z-10 hidden h-24 w-3 -translate-x-1/2 -translate-y-1/2 cursor-col-resize rounded-full border border-slate-200/80 bg-white/95 text-slate-400 shadow-sm transition-colors hover:bg-white hover:text-slate-700 xl:flex"
                title="Resize right panel"
              >
                <span className="mx-auto h-10 w-px bg-slate-300" />
              </button>
            )}
            {isRightPanelOpen ? (
              <div className="h-full p-4">
                {rightPanelKind === 'soul' ? (
                  <SoulPanel
                    characterId={selectedCharacterId as string}
                    refreshKey={soulRefreshKey}
                    isOpen
                    onToggle={() => setIsRightPanelOpen(false)}
                    className="h-full"
                  />
                ) : rightPanelKind === 'memory' ? (
                  <MemoryPanel
                    characterId={selectedCharacterId as string}
                    chatSessionId={selectedSessionId}
                    refreshKey={soulRefreshKey}
                    onClose={() => setIsRightPanelOpen(false)}
                  />
                ) : (
                  <ResearchPanel onClose={() => setIsRightPanelOpen(false)} />
                )}
              </div>
            ) : (
              (() => {
                const soulActive = rightPanelKind === 'soul';
                const memoryActive = rightPanelKind === 'memory';
                const researchActive = rightPanelKind === 'research';
                const launcherBaseClass = 'flex h-10 w-10 items-center justify-center rounded-2xl transition-colors';
                const launcherActiveClass = 'bg-slate-900 text-white hover:bg-slate-800';
                const launcherMutedClass = 'bg-white/70 text-slate-500 ring-1 ring-slate-200 hover:bg-white hover:text-slate-900';

                return (
                  <aside className="flex h-full w-full flex-shrink-0 flex-col items-center gap-3 rounded-[1.75rem] border border-slate-200/80 bg-white/75 px-2 py-3 shadow-[0_20px_60px_rgba(15,23,42,0.08)] backdrop-blur">
                    <button
                      type="button"
                      onClick={() => handleRightPanelIconClick('soul')}
                      className={`${launcherBaseClass} ${soulActive ? launcherActiveClass : launcherMutedClass}`}
                      title={messages.chat.toggleSoulPanel}
                    >
                      <FolderTree className="h-5 w-5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => handleRightPanelIconClick('memory')}
                      className={`${launcherBaseClass} ${memoryActive ? launcherActiveClass : launcherMutedClass}`}
                      title={messages.chat.toggleMemoryPanel}
                    >
                      <Brain className="h-5 w-5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => handleRightPanelIconClick('research')}
                      className={`${launcherBaseClass} ${researchActive ? launcherActiveClass : launcherMutedClass}`}
                      title={messages.chat.toggleResearchPanel}
                    >
                      <Globe className="h-5 w-5" />
                    </button>
                  </aside>
                );
              })()
            )}
          </div>
        )}
      </div>

      {renamingChat && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4 backdrop-blur-sm"
          onClick={() => setRenamingChat(null)}
          role="presentation"
        >
          <div
            className="w-full max-w-md rounded-3xl border border-slate-200 bg-white p-6 shadow-2xl"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-label={messages.shell.renameTopic}
          >
            <h3 className="text-base font-semibold text-slate-900">{messages.shell.renameTopic}</h3>
            <input
              autoFocus
              type="text"
              value={renameDraft}
              onChange={(event) => {
                setRenameDraft(event.target.value);
                setRenameError(null);
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  void submitRename();
                } else if (event.key === 'Escape') {
                  setRenamingChat(null);
                }
              }}
              placeholder={messages.shell.renameTopicPlaceholder}
              className="mt-3 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm text-slate-800 outline-none focus:border-sky-400 focus:ring-2 focus:ring-sky-100"
            />
            {renameError && (
              <p className="mt-2 text-xs text-rose-600">{renameError}</p>
            )}
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setRenamingChat(null)}
                className="rounded-xl border border-slate-200 px-4 py-2 text-sm text-slate-600 transition-colors hover:bg-slate-50"
              >
                {messages.shell.cancel}
              </button>
              <button
                type="button"
                onClick={() => void submitRename()}
                disabled={!renameDraft.trim()}
                className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {messages.shell.save}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  function renderContent(view: ViewState) {
    switch (view) {
      case 'playground':
        return selectedCharacterId ? (
          <div className="h-full flex flex-col">
            <ChatInterface
              characterId={selectedCharacterId}
              initialSessionId={selectedSessionId}
              modelConfigs={modelConfigs}
              modelRoles={modelRoles}
              defaultModelConfigId={defaultModelConfigId}
              userProfile={userProfile}
              onBack={handleBackToGallery}
              onSessionUpdate={fetchChatSessions}
              onSoulRefreshKeyChange={setSoulRefreshKey}
            />
          </div>
        ) : (
          <div className="h-full overflow-y-auto bg-[linear-gradient(180deg,rgba(255,255,255,0.28),rgba(244,247,250,0.72))]">
            <CharacterGallery onSelect={handleSelectCharacter} />
          </div>
        );

      case 'create':
        return (
          <div className="h-full overflow-y-auto bg-[linear-gradient(180deg,rgba(255,255,255,0.28),rgba(244,247,250,0.72))]">
            <div className="max-w-5xl mx-auto py-8 px-6">
              <CreateCharacterSimplifiedForm onCancel={() => setCurrentView('playground')} />
            </div>
          </div>
        );

      case 'model_settings':
        return (
          <ModelApiSettingsPanel
            modelConfigs={modelConfigs}
            modelRoles={modelRoles}
            webSearchConfig={webSearchConfig}
            loading={loadingModelConfigs}
            loadingRoles={loadingModelRoles}
            loadingWebSearchConfig={loadingWebSearchConfig}
            error={modelConfigError}
            webSearchError={webSearchConfigError}
            onRefresh={fetchModelConfigs}
            onRefreshModelRoles={fetchModelRoles}
            onRefreshWebSearchConfig={fetchWebSearchConfig}
          />
        );

      case 'user_settings':
        return (
          <UserSettingsPanel
            profile={userProfile}
            loading={loadingUserProfile}
            error={userProfileError}
            onRefresh={fetchUserProfile}
            onOpenModelSettings={openModelSettings}
          />
        );

      case 'history_all':
        const startIndex = (historyPage - 1) * ITEMS_PER_PAGE;
        const currentHistoryItems = filteredHistoryItems.slice(startIndex, startIndex + ITEMS_PER_PAGE);
        let historyContent: React.ReactNode;

        if (loadingChats) {
          historyContent = (
            <div className="flex justify-center py-20">
              <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-slate-900"></div>
            </div>
          );
        } else if (!recentChats.length) {
          historyContent = (
            <div className="rounded-[1.75rem] border border-dashed border-slate-200 bg-white/70 py-20 text-center text-slate-500 shadow-[0_18px_60px_rgba(15,23,42,0.05)]">
              <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-50">
                <MessageSquare size={32} className="text-slate-400" />
              </div>
              <p className="text-lg font-medium text-slate-900">{messages.shell.noConversationHistory}</p>
              <p className="mt-1 max-w-sm mx-auto text-sm">{messages.shell.historyEmptyDescription}</p>
              <button
                onClick={() => setCurrentView('playground')}
                className="mt-6 rounded-xl bg-slate-900 px-6 py-2.5 font-medium text-white shadow-sm transition-colors hover:bg-slate-800"
              >
                {messages.shell.goToPlayground}
              </button>
            </div>
          );
        } else if (!filteredHistoryItems.length) {
          historyContent = (
            <div className="rounded-[1.75rem] border border-dashed border-slate-200 bg-white/70 py-20 text-center text-slate-500 shadow-[0_18px_60px_rgba(15,23,42,0.05)]">
              <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-50">
                <Search size={30} className="text-slate-400" />
              </div>
              <p className="text-lg font-medium text-slate-900">{messages.shell.noSearchResults}</p>
              <p className="mt-1 max-w-sm mx-auto text-sm">{messages.shell.noSearchResultsDescription}</p>
              <button
                onClick={() => {
                  setHistorySearchQuery('');
                  setHistoryPage(1);
                }}
                className="mt-6 rounded-xl border border-slate-200 bg-white/90 px-6 py-2.5 font-medium text-slate-700 shadow-sm transition-colors hover:bg-white"
              >
                {messages.shell.clearSearch}
              </button>
            </div>
          );
        } else {
          historyContent = (
            <>
              <div className="mb-6 overflow-hidden rounded-[1.75rem] border border-white/70 bg-white/72 shadow-[0_18px_60px_rgba(15,23,42,0.08)] backdrop-blur">
                <div className="divide-y divide-slate-100">
                  {currentHistoryItems.map((chat) => (
                    <div
                      key={chat.id}
                      onClick={() => handleSelectHistoryItem(chat.characterId, chat.id)}
                      className="group flex cursor-pointer items-center justify-between p-5 transition-colors hover:bg-sky-50/50"
                    >
                      <div className="flex items-center gap-4 min-w-0 flex-1 mr-4">
                        <div className="flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-100 via-cyan-50 to-amber-50 text-sky-700">
                          <MessageSquare size={20} />
                        </div>
                        <div className="min-w-0">
                          <h3 className="truncate font-medium text-slate-900 transition-colors group-hover:text-sky-700">
                            {chat.title}
                          </h3>
                          <div className="mt-0.5 flex items-center gap-2 text-xs text-slate-500">
                            <span>{messages.shell.sessionId}: {chat.id}</span>
                            <span className="h-1 w-1 rounded-full bg-slate-300"></span>
                            <span>{messages.shell.characterId}: {chat.characterId}</span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-4 flex-shrink-0">
                        <div className="text-right hidden sm:block">
                          <div className="text-sm font-medium text-slate-600">
                            {chat.updated_at && !isNaN(new Date(chat.updated_at).getTime())
                              ? formatDate(chat.updated_at)
                              : messages.shell.unknownDate}
                          </div>
                          <div className="text-xs text-slate-400">
                            {chat.updated_at && !isNaN(new Date(chat.updated_at).getTime())
                              ? formatTime(chat.updated_at)
                              : ''}
                          </div>
                        </div>
                        <div className="flex items-center gap-2 border-l border-slate-100 pl-4">
                          <button
                            onClick={(e) => handleRenameSession(e, chat)}
                            className="rounded-xl p-2 text-slate-400 transition-colors hover:bg-sky-50 hover:text-sky-600"
                            title={messages.shell.renameTopic}
                          >
                            <Pencil size={18} />
                          </button>
                          <button
                            onClick={(e) => handleDeleteSession(e, chat.id)}
                            className="rounded-xl p-2 text-slate-400 transition-colors hover:bg-rose-50 hover:text-rose-600"
                            title={messages.shell.deleteConversation}
                          >
                            <Trash2 size={18} />
                          </button>
                          <div className="text-slate-300 transition-colors group-hover:text-sky-500">
                            <ArrowRight size={20} />
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {totalHistoryPages > 1 && (
                <div className="flex items-center justify-center gap-2">
                  <button
                    onClick={() => setHistoryPage(p => Math.max(1, p - 1))}
                    disabled={historyPage === 1}
                    className="rounded-xl border border-slate-200 bg-white/90 p-2 transition-colors hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <ChevronLeft size={20} />
                  </button>

                  <span className="px-4 text-sm font-medium text-slate-600">
                    {messages.shell.pageStatus(historyPage, totalHistoryPages)}
                  </span>

                  <button
                    onClick={() => setHistoryPage(p => Math.min(totalHistoryPages, p + 1))}
                    disabled={historyPage === totalHistoryPages}
                    className="rounded-xl border border-slate-200 bg-white/90 p-2 transition-colors hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <ChevronRight size={20} />
                  </button>
                </div>
              )}
            </>
          );
        }

        return (
          <div className="h-full overflow-y-auto bg-[linear-gradient(180deg,rgba(255,255,255,0.22),rgba(243,247,250,0.9))]">
            <div className="p-6 md:p-10 max-w-6xl mx-auto">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-slate-400">{messages.shell.archive}</p>
                  <h2 className="mt-1 text-3xl font-semibold tracking-tight text-slate-900">{messages.shell.chatHistory}</h2>
                  <p className="mt-2 text-sm text-slate-500">{messages.shell.managePastConversations}</p>
                </div>
                <button
                  onClick={() => {
                    fetchChatSessions();
                    setHistoryPage(1);
                  }}
                  className="rounded-xl border border-slate-200 bg-white/90 px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:bg-white"
                >
                  {messages.shell.refreshList}
                </button>
              </div>

              <div className="mb-6 rounded-[1.5rem] border border-white/70 bg-white/75 p-4 shadow-[0_16px_50px_rgba(15,23,42,0.06)] backdrop-blur">
                <label className="mb-2 block text-xs font-medium uppercase tracking-[0.18em] text-slate-400" htmlFor="history-search">
                  {messages.shell.searchHistory}
                </label>
                <div className="relative">
                  <Search size={16} className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
                  <input
                    id="history-search"
                    type="search"
                    value={historySearchQuery}
                    onChange={(event) => {
                      setHistorySearchQuery(event.target.value);
                      setHistoryPage(1);
                    }}
                    placeholder={messages.shell.searchHistoryPlaceholder}
                    className="w-full rounded-2xl border border-slate-200 bg-white py-3 pl-11 pr-11 text-sm text-slate-700 outline-none transition focus:border-sky-300 focus:ring-4 focus:ring-sky-100"
                  />
                  {historySearchQuery && (
                    <button
                      type="button"
                      onClick={() => {
                        setHistorySearchQuery('');
                        setHistoryPage(1);
                      }}
                      className="absolute right-3 top-1/2 -translate-y-1/2 rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-700"
                      title={messages.shell.clearSearch}
                    >
                      <X size={14} />
                    </button>
                  )}
                </div>
              </div>

              {historyContent}
            </div>
          </div >
        );

      case 'home':
        return (
          <div className="flex h-full items-center justify-center overflow-y-auto bg-[radial-gradient(circle_at_top,#ffffff_0%,#eef4f8_52%,#f6eee5_100%)] p-8">
            <div className="w-full max-w-4xl rounded-[2rem] border border-white/80 bg-white/70 px-8 py-12 text-center shadow-[0_24px_80px_rgba(15,23,42,0.08)] backdrop-blur">
              <div className="mx-auto mb-5 flex h-18 w-18 items-center justify-center rounded-[1.75rem] bg-gradient-to-br from-sky-100 via-cyan-50 to-amber-50 text-sky-700 shadow-sm">
                <Zap size={34} />
              </div>
              <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-slate-400">{messages.shell.characterStudio}</p>
              <h1 className="mt-3 text-4xl font-semibold tracking-tight text-slate-900">{messages.shell.welcomeToApp(APP_NAME)}</h1>
              <p className="mx-auto mt-4 max-w-2xl text-base leading-8 text-slate-600">
                {messages.shell.homeHeroDescription}
              </p>
              <div className="mt-8 flex flex-wrap justify-center gap-4">
                <button onClick={() => setCurrentView('create')} className="rounded-xl bg-slate-900 px-6 py-3 font-medium text-white transition-colors hover:bg-slate-800">
                  {messages.shell.createCharacter}
                </button>
                <button onClick={() => setCurrentView('playground')} className="rounded-xl border border-slate-200 bg-white/90 px-6 py-3 font-medium text-slate-700 transition-colors hover:bg-white">
                  {messages.shell.goToPlayground}
                </button>
                {!hasModelConfigs && (
                  <button onClick={openModelSettings} className="rounded-xl border border-amber-200 bg-amber-50 px-6 py-3 font-medium text-amber-800 transition-colors hover:bg-amber-100">
                    {messages.shell.configureModelApi}
                  </button>
                )}
              </div>
              <div className="mt-10 grid gap-4 text-left md:grid-cols-3">
                <div className="rounded-2xl border border-white/80 bg-white/80 p-5 shadow-sm">
                  <p className="text-sm font-medium text-slate-900">{messages.shell.immersiveChat}</p>
                  <p className="mt-2 text-sm leading-6 text-slate-500">{messages.shell.immersiveChatDescription}</p>
                </div>
                <div className="rounded-2xl border border-white/80 bg-white/80 p-5 shadow-sm">
                  <p className="text-sm font-medium text-slate-900">{messages.shell.sessionMemory}</p>
                  <p className="mt-2 text-sm leading-6 text-slate-500">{messages.shell.sessionMemoryDescription}</p>
                </div>
                <div className="rounded-2xl border border-white/80 bg-white/80 p-5 shadow-sm">
                  <p className="text-sm font-medium text-slate-900">{messages.shell.futureReadyModel}</p>
                  <p className="mt-2 text-sm leading-6 text-slate-500">{messages.shell.futureReadyModelDescription}</p>
                </div>
              </div>
            </div>
          </div>
        );

      default:
        return <CharacterGallery onSelect={handleSelectCharacter} />;
    }
  }
}

function NavItem({
  icon,
  label,
  active = false,
  onClick,
  isPrimary = false,
  badge,
  badgeTone = 'neutral',
}: {
  icon: React.ReactNode,
  label: string,
  active?: boolean,
  onClick: () => void,
  isPrimary?: boolean,
  badge?: string,
  badgeTone?: 'neutral' | 'warning' | 'ready',
}) {
  const badgeClassName = badgeTone === 'warning'
    ? 'bg-amber-100 text-amber-800'
    : badgeTone === 'ready'
      ? 'bg-emerald-100 text-emerald-700'
      : 'bg-slate-100 text-slate-600';

  return (
    <button
      onClick={onClick}
      className={`
        flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all duration-200
        ${active
          ? 'bg-sky-50 text-sky-700 ring-1 ring-sky-100'
          : 'text-slate-600 hover:bg-white/80 hover:text-slate-900'
        }
        ${isPrimary && active ? 'shadow-sm' : ''}
      `}
    >
      <span className={active ? 'text-sky-600' : 'text-slate-400'}>{icon}</span>
      <span className="min-w-0 flex-1 truncate text-left">{label}</span>
      {badge && (
        <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] ${badgeClassName}`}>
          {badge}
        </span>
      )}
    </button>
  );
}

export default function AIStudioLayout() {
  return (
    <Suspense fallback={<div className="h-screen w-full bg-[linear-gradient(180deg,#f5f8fc_0%,#eef3f7_50%,#f6efe8_100%)]" />}>
      <AIStudioLayoutContent />
    </Suspense>
  );
}
