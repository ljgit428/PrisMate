"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useDispatch } from "react-redux";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowLeft, Loader2, Menu, MessagesSquare, Search, X } from "lucide-react";
import ChatInterface from "@/components/ChatInterface";
import { useI18n } from "@/i18n/provider";
import { clearChat } from "@/store/chatSlice";
import { apiService } from "@/utils/api";
import { Character, ChatSession, ModelConfig, ModelRoleAssignments, UserProfile } from "@/types";
import { APP_NAME } from "@/constants";

interface LatestSessionInfo {
  id: string;
  updatedAt: string;
}

function DiscordChatPageContent() {
  const { messages: copy, formatDate } = useI18n();
  const dispatch = useDispatch();
  const router = useRouter();
  const searchParams = useSearchParams();
  const autoSelectId = searchParams.get("select");

  const [characters, setCharacters] = useState<Character[]>([]);
  const [loadingCharacters, setLoadingCharacters] = useState(true);
  const [characterError, setCharacterError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [modelConfigs, setModelConfigs] = useState<ModelConfig[]>([]);
  const [modelRoles, setModelRoles] = useState<ModelRoleAssignments | null>(null);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [selectedCharacterId, setSelectedCharacterId] = useState<string | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const latestSessionByCharacter = useMemo(() => {
    const map = new Map<string, LatestSessionInfo>();

    for (const session of sessions) {
      const characterId = session.character?.id;
      if (!characterId) {
        continue;
      }

      const existing = map.get(characterId);
      if (!existing || new Date(session.updatedAt) > new Date(existing.updatedAt)) {
        map.set(characterId, { id: session.id, updatedAt: session.updatedAt });
      }
    }

    return map;
  }, [sessions]);

  const loadSidebarData = useCallback(async () => {
    setLoadingCharacters(true);
    setCharacterError(null);

    try {
      const [charactersRes, sessionsRes] = await Promise.all([
        apiService.getCharacters(),
        apiService.getChatSessions(undefined, 'chat'),
      ]);

      if (charactersRes.error || !charactersRes.data) {
        setCharacterError(charactersRes.error || copy.gallery.errorLoadingCharacters);
        return;
      }

      setCharacters(charactersRes.data);
      if (sessionsRes.data) {
        setSessions(sessionsRes.data);
      }
    } catch (error) {
      console.error("Failed to load chat page data:", error);
      setCharacterError(copy.gallery.errorLoadingCharacters);
    } finally {
      setLoadingCharacters(false);
    }
  }, [copy.gallery.errorLoadingCharacters]);

  const refreshSessions = useCallback(async () => {
    try {
      const response = await apiService.getChatSessions(undefined, 'chat');
      if (response.data) {
        setSessions(response.data);
      }
    } catch (error) {
      console.error("Failed to refresh chat sessions:", error);
    }
  }, []);

  useEffect(() => {
    void loadSidebarData();
  }, [loadSidebarData]);

  useEffect(() => {
    const loadModelSettings = async () => {
      try {
        const [configsRes, rolesRes] = await Promise.all([
          apiService.getModelConfigs(),
          apiService.getModelRoles(),
        ]);

        if (configsRes.data) {
          setModelConfigs(configsRes.data);
        }
        if (rolesRes.data) {
          setModelRoles(rolesRes.data);
        }
      } catch (error) {
        console.error("Failed to load model settings:", error);
      }
    };

    const loadUserProfile = async () => {
      try {
        const response = await apiService.getUserProfile();
        if (response.data) {
          setUserProfile(response.data);
        }
      } catch (error) {
        console.error("Failed to load user profile:", error);
      }
    };

    void loadModelSettings();
    void loadUserProfile();
  }, []);

  const handleSelectCharacter = useCallback(
    (characterId: string) => {
      dispatch(clearChat());
      setSelectedCharacterId(characterId);
      setSelectedSessionId(latestSessionByCharacter.get(characterId)?.id || null);
      setIsSidebarOpen(false);
    },
    [dispatch, latestSessionByCharacter]
  );

  useEffect(() => {
    if (!autoSelectId || loadingCharacters) {
      return;
    }

    handleSelectCharacter(autoSelectId);
    router.replace("/chat");
  }, [autoSelectId, loadingCharacters, handleSelectCharacter, router]);

  const normalizedQuery = searchQuery.trim().toLowerCase();
  const filteredCharacters = useMemo(() => {
    if (!normalizedQuery) {
      return characters;
    }

    return characters.filter((character) => {
      const haystack = [character.name, character.description, character.affiliation, character.personality]
        .join(" ")
        .toLowerCase();
      return haystack.includes(normalizedQuery);
    });
  }, [characters, normalizedQuery]);

  const selectedCharacter = characters.find((character) => character.id === selectedCharacterId) || null;
  const defaultModelConfigId = modelRoles?.text?.id || modelConfigs[0]?.id || null;

  const getCharacterSubtitle = (character: Character) => {
    const latest = latestSessionByCharacter.get(character.id);
    if (latest) {
      return `${copy.chatPage.lastChatAt} · ${formatDate(latest.updatedAt)}`;
    }

    return character.affiliation?.trim() || copy.chatPage.noChatYet;
  };

  const renderCharacterList = () => {
    if (loadingCharacters) {
      return (
        <div className="flex items-center gap-2 px-2 py-6 text-sm text-slate-500">
          <Loader2 size={14} className="animate-spin" />
          <span>{copy.gallery.loadingCharacters}</span>
        </div>
      );
    }

    if (characterError) {
      return <p className="px-2 py-4 text-sm text-rose-500">{characterError}</p>;
    }

    if (characters.length === 0) {
      return (
        <div className="mx-2 rounded-2xl border border-dashed border-slate-200 bg-white/70 px-4 py-8 text-center text-sm text-slate-500">
          {copy.chatPage.noCharacters}
        </div>
      );
    }

    if (filteredCharacters.length === 0) {
      return (
        <div className="mx-2 rounded-2xl border border-dashed border-slate-200 bg-white/70 px-4 py-8 text-center">
          <p className="text-sm font-medium text-slate-600">{copy.chatPage.noResults}</p>
          <p className="mt-1 text-xs leading-5 text-slate-400">{copy.chatPage.noResultsHint}</p>
        </div>
      );
    }

    return filteredCharacters.map((character) => {
      const isActive = character.id === selectedCharacterId;

      return (
        <button
          key={character.id}
          type="button"
          onClick={() => handleSelectCharacter(character.id)}
          className={`flex w-full items-center gap-3 rounded-2xl px-2.5 py-2 text-left transition-colors ${
            isActive
              ? "bg-sky-50 text-sky-900 ring-1 ring-sky-100"
              : "text-slate-600 hover:bg-white/80 hover:text-slate-900"
          }`}
        >
          <span className="relative h-10 w-10 flex-shrink-0">
            <span className="flex h-10 w-10 items-center justify-center overflow-hidden rounded-full bg-gradient-to-br from-sky-100 via-cyan-50 to-amber-50 text-sky-700">
              {character.avatarUrl ? (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img src={character.avatarUrl} alt={character.name} className="h-full w-full object-cover" />
              ) : (
                <span className="text-sm font-semibold">{character.name.charAt(0).toUpperCase()}</span>
              )}
            </span>
            <span
              className="absolute -bottom-0.5 -right-0.5 h-3.5 w-3.5 rounded-full border-2 border-white bg-emerald-500"
              aria-hidden
            />
          </span>
          <span className="min-w-0 flex-1">
            <span className={`block truncate text-sm ${isActive ? "font-semibold" : "font-medium"}`}>
              {character.name}
            </span>
            <span className="block truncate text-xs text-slate-400">{getCharacterSubtitle(character)}</span>
          </span>
        </button>
      );
    });
  };

  return (
    <div className="flex h-screen w-full bg-[linear-gradient(180deg,#f8fafc_0%,#eef2f7_100%)] text-slate-700">
      {isSidebarOpen && selectedCharacterId && (
        <div
          className="fixed inset-0 z-20 bg-slate-950/40 backdrop-blur-sm md:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-30 flex w-72 flex-shrink-0 flex-col border-r border-slate-200/70 bg-white/85 backdrop-blur-xl transition-transform duration-200 ease-in-out md:static md:translate-x-0 ${
          selectedCharacterId && !isSidebarOpen ? "-translate-x-full" : "translate-x-0"
        }`}
      >
        <div className="flex h-[60px] flex-shrink-0 items-center gap-2 border-b border-slate-200/70 px-4">
          <button
            type="button"
            onClick={() => router.push("/")}
            className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-700"
            title={copy.chatPage.backToTopics}
          >
            <ArrowLeft size={18} />
          </button>
          <div className="min-w-0">
            <p className="text-[10px] font-medium uppercase tracking-[0.18em] text-slate-400">{APP_NAME}</p>
            <p className="flex items-center gap-1.5 text-base font-semibold tracking-tight text-slate-900">
              <MessagesSquare size={15} className="text-sky-600" />
              {copy.chatPage.pageTitle}
            </p>
          </div>
        </div>

        <div className="flex-shrink-0 px-3 py-3">
          <div className="relative">
            <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder={copy.chatPage.searchPlaceholder}
              className="w-full rounded-xl border border-slate-200 bg-slate-50/80 py-2 pl-9 pr-8 text-sm text-slate-700 placeholder:text-slate-400 outline-none transition-colors focus:border-sky-300 focus:bg-white focus:ring-2 focus:ring-sky-100"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery("")}
                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-lg p-1 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
                title={copy.chatPage.clearSearch}
              >
                <X size={13} />
              </button>
            )}
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
          <p className="mb-2 px-2 text-[11px] font-medium uppercase tracking-[0.18em] text-slate-400">
            {copy.chatPage.directMessages}
            {!loadingCharacters && !characterError && characters.length > 0 && (
              <span className="ml-1.5 normal-case tracking-normal text-slate-300">
                · {copy.chatPage.characterCount(characters.length)}
              </span>
            )}
          </p>
          <div className="space-y-1">{renderCharacterList()}</div>
        </div>

        <div className="flex flex-shrink-0 items-center gap-3 border-t border-slate-200/70 bg-slate-50/80 px-4 py-3">
          <span className="relative h-9 w-9 flex-shrink-0">
            <span className="flex h-9 w-9 items-center justify-center overflow-hidden rounded-full bg-slate-200 text-sm font-semibold text-slate-600">
              {userProfile?.avatarUrl ? (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img src={userProfile.avatarUrl} alt="" className="h-full w-full object-cover" />
              ) : (
                (userProfile?.preferredName || "You").charAt(0).toUpperCase()
              )}
            </span>
            <span
              className="absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full border-2 border-white bg-emerald-500"
              aria-hidden
            />
          </span>
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-slate-800">
              {userProfile?.preferredName || copy.chat.you}
            </p>
            <p className="flex items-center gap-1 text-xs text-emerald-600">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" aria-hidden />
              {copy.chatPage.online}
            </p>
          </div>
        </div>
      </aside>

      <main className={`min-w-0 flex-1 flex-col ${selectedCharacterId ? "flex" : "hidden md:flex"}`}>
        {selectedCharacterId ? (
          <>
            <div className="flex flex-shrink-0 items-center gap-2 border-b border-slate-200/70 bg-white/75 px-3 py-2 backdrop-blur-xl md:hidden">
              <button
                type="button"
                onClick={() => setIsSidebarOpen(true)}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-800"
                title={copy.chatPage.openSidebar}
              >
                <Menu size={17} />
              </button>
              <p className="truncate text-sm font-medium text-slate-700">
                {selectedCharacter?.name || copy.chat.loadingCharacter}
              </p>
            </div>
            <div className="min-h-0 flex-1">
              <ChatInterface
                key={`${selectedCharacterId}:${selectedSessionId || "new"}`}
                characterId={selectedCharacterId}
                initialSessionId={selectedSessionId}
                modelConfigs={modelConfigs}
                modelRoles={modelRoles}
                defaultModelConfigId={defaultModelConfigId}
                userProfile={userProfile}
                sessionOrigin="chat"
                onSessionUpdate={refreshSessions}
              />
            </div>
          </>
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center gap-4 p-10 text-center">
            <div className="flex h-20 w-20 items-center justify-center rounded-3xl bg-white shadow-[0_18px_60px_rgba(15,23,42,0.08)] ring-1 ring-slate-200/70">
              <MessagesSquare size={34} className="text-sky-500" />
            </div>
            <div>
              <h1 className="text-xl font-semibold tracking-tight text-slate-900">
                {copy.chatPage.selectCharacterTitle}
              </h1>
              <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">
                {copy.chatPage.selectCharacterHint}
              </p>
            </div>
            <button
              type="button"
              onClick={() => router.push("/")}
              className="inline-flex items-center gap-2 rounded-full bg-white/80 px-4 py-2 text-xs font-medium text-slate-600 ring-1 ring-slate-200 transition-colors hover:bg-white hover:text-slate-900"
            >
              <ArrowLeft size={14} />
              <span>{copy.chatPage.backToTopics}</span>
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

export default function ChatPage() {
  return (
    <Suspense fallback={null}>
      <DiscordChatPageContent />
    </Suspense>
  );
}
