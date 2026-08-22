export interface ResearchItem {
  title: string;
  url: string;
  snippet: string;
  domain?: string;
  source?: string;
}

export interface ResearchPayload {
  query?: string;
  provider?: string;
  items: ResearchItem[];
  error?: string;
}

export interface MemoryExplorerEntry {
  path: string;
  entryType: 'file' | 'directory';
  layer: 'schema' | 'wiki' | 'raw' | string;
  title: string;
  kind: string;
  readHint: string;
  isLocked: boolean;
  canUserEdit: boolean;
  canAutoUpdate: boolean;
  updatedAt: string;
  manageable?: boolean;
  assetId?: string;
  previewKind?: 'text' | 'image' | 'binary' | 'directory' | string;
  childCount?: number;
  sizeHint?: number;
}

export interface MemoryExplorerFile {
  path: string;
  layer: 'schema' | 'wiki' | 'raw' | string;
  title: string;
  kind: string;
  readHint: string;
  content: string;
  truncated?: boolean;
  manageable?: boolean;
  assetId?: string;
  previewKind?: 'text' | 'image' | 'binary' | string;
  fileUrl?: string;
  mimeType?: string;
  error?: string;
}

export interface KnowledgeAsset {
  id: string;
  fileUrl?: string;
  fileName: string;
  fileType: string;
  fileMimeType?: string;
  createdAt: string;
  updatedAt: string;
}

export interface ToolCallInfo {
  tool: string;
  arguments?: Record<string, unknown>;
}

export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  cachedTokens: number;
}

export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: string;
  senderId?: string;
  senderName?: string;
  senderAvatarUrl?: string;
  senderType?: 'user' | 'character' | 'system';
  researchPayload?: ResearchPayload | null;
  thinking?: string;
  toolCalls?: ToolCallInfo[];
  tokenUsage?: TokenUsage | null;
  attachments?: MessageAttachment[];
  fileUri?: string;
  fileName?: string;
  filePreviewUrl?: string;
  fileType?: string;
  fileMimeType?: string;
}

export interface MessageAttachment {
  fileUri?: string;
  fileName?: string;
  filePreviewUrl?: string;
  fileType?: string;
  fileMimeType?: string;
}

export interface Character {
  id: string;
  name: string;
  description: string;
  userAddress: string;
  scenario: string;
  exampleDialogue: string;
  personality: string;
  appearance: string;
  responseGuidelines: string;
  avatarUrl?: string;
  fileUrl?: string;
  filePreviewUrl?: string;
  affiliation: string;
  disabled: {
    name: boolean;
    description: boolean;
    personality: boolean;
    appearance: boolean;
    responseGuidelines: boolean;
    file: boolean;
  };
}

export type ModelProvider = 'gemini' | 'openai_compatible' | 'anthropic';
export type WebSearchProvider = 'tavily';

/** 模型角色槽位：text 必填，image/audio/video 可空（空=该类附件不做 AI 解读） */
export type ModelRoleKey = 'text' | 'image' | 'audio' | 'video';

export interface ModelConfig {
  id: string;
  name: string;
  provider: ModelProvider;
  modelName: string;
  apiKey: string;
  baseUrl?: string;
  contextWindow?: number | null;
  createdAt: string;
  updatedAt: string;
}

export type ModelRoleAssignments = Record<ModelRoleKey, ModelConfig | null>;

export interface WebSearchConfig {
  id?: string;
  provider: WebSearchProvider;
  apiKey: string;
  maxResults: number;
  createdAt?: string;
  updatedAt?: string;
}

export interface WebSearchTestResult {
  query?: string;
  provider?: string;
  items: ResearchItem[];
  error?: string;
}

export type LocationPrecision = 'region' | 'city' | 'exact';
export type ReplyLengthPreference = 'short' | 'medium' | 'long';
export type PreferenceLevel = 'low' | 'normal' | 'high';

export interface UserProfile {
  id: string;
  avatarUrl?: string;
  preferredName: string;
  pronouns: string;
  bio: string;
  defaultEnableWebSearch: boolean;
  timezone: string;
  interfaceLanguage: string;
  shareLocalTime: boolean;
  shareLocation: boolean;
  locationPrecision: LocationPrecision;
  locationLabel: string;
  shareWeather: boolean;
  preferredRelationshipStyle: string;
  preferredReplyLength: ReplyLengthPreference;
  preferredProactivity: PreferenceLevel;
  preferredEmotionalIntensity: PreferenceLevel;
  allowLongTermMemory: boolean;
  allowPreferenceInference: boolean;
  allowResearchProfileUpdates: boolean;
  blockedTopics: string;
  createdAt: string;
  updatedAt: string;
}

export interface ChatSession {
  id: string;
  title: string;
  lastResponseLatencyMs?: number | null;
  isPrivateMode?: boolean;
  origin?: string;
  character: Character;
  createdAt: string;
  updatedAt: string;
}

export interface MemoryHistoryEntry {
  old_desc: string;
  new_desc: string;
  old_section?: string;
  new_section?: string;
  old_time?: string;
  new_time?: string;
  reason?: string;
  merged_from?: string;
}

export interface MemoryEntry {
  shortId: string;
  section: string;
  description: string;
  descriptionHistory?: MemoryHistoryEntry[];
  createdAt: string;
  updatedAt: string;
}

export interface MemorySectionGroup {
  section: string;
  items: MemoryEntry[];
}

export interface MemorySnapshot {
  sections: MemorySectionGroup[];
  wikiMarkdown: string;
  count: number;
}

export interface ChatState {
  messages: Message[];
  character: Character | null;
  chatSession: ChatSession | null;
  isLoading: boolean;
  error: string | null;
}

export interface RootState {
  chat: ChatState;
}
