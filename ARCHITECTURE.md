# System Architecture & Data Flow

## 1. Core Chat Loop (Sending Message)
## 2. AI Character Gen
## 3. Character Persistence
## 4. Sessions History Sidebar Display
## 5. Session Management
## 6. History List
## 7. Message Restoration & Chat Initialization
## 8. Delete Conversation (REST)
## 9. Delete Character (GraphQL + Validation)

## 1. Core Chat Loop (Sending Message) (REST API)
**Goal:** Real-time messaging with latency control.
**Flow:**
1. [UI] `ChatInterface.tsx` (User Input) (`handleSendMessage` function)
   ↓
2. [Client] `api.ts` (`sendMessage` function)
   ↓ POST /api/chat/send_message/
3. [View] `views.py` (`ChatViewSet`) -> Save User Message to DB
   ├── **Step A: Session Handling**
   │   Check `chat_session_id`. If null -> `ChatSession.objects.create()`
   │   (Initialize with User Persona, World Time, etc.)
   │
   └── **Step B: User Persistence**
       `Message.objects.create(role='user', content=...)`
       **[DB Write 1]** Save User Message to PostgreSQL.
   ↓ (Synchronous Call)
4. [Engine] `tasks.py` (`generate_ai_response`)
   ↓
5. [LLM] **Google Gemini API** (Context Window Injection)
   ↓
6. [DB] `models.py` (Save Assistant Response)
   ↓
7. [Client State] `ChatInterface.tsx` (Handle Response)
   ├── **Update Redux:** `dispatch(addMessage(ai_message))`
   │
   └── **Bind Session:** `setChatSessionId(response.chat_session_id)`
       (Crucial: Transitions state from "New Chat" -> "Existing Session")
   ↓
8. [Cross-Component Sync]
   `onSessionUpdate()` callback
   ↓ Triggers `page.tsx`
   ↓ Refreshes Sidebar History List (Shows new title/timestamp)

---

## 2. AI Character Generation (GraphQL / ETL)
**Goal:** Extract structured persona from unstructured files (novels/PDFs/dialogue scripts).
**Flow:**
1. [UI] `CreateCharacterSimplifiedForm.tsx` (File Drop) -> uploads land via `/api/upload`
   ↓ GraphQL Mutation (`GENERATE_DRAFT`, fileUrls + textContext)
2. [Schema] `graphql/schema.py` (`generateCharacterDraft`)
   ↓ `_resolve_staged_uploads()` — resolve uploaded URLs to in-memory records
   (text content extracted; images carry URL/metadata only)
   ↓
   **Branch A — small batches (< 12 text files): Memory Tools ReAct loop**
   - Files are NOT inlined into the prompt. The model calls `list_memory_files` /
     `read_memory_file` (backed by `StagedUploadMemoryFilesystem`) and reads only
     what it needs. Providers: OpenAI-compatible / Anthropic (Gemini falls back
     to local text reads).
   ↓
   **Branch B — large batches (>= 12 text files): reduce pipeline**
   `chat/character_reduce.py` `run_reduce_pipeline()`:
   1. Tier files by target-character line count (rules, 0 LLM): main/mid/cameo
   2. Batch close-read: main in full, cameo as ±3-line segments
   3. Per-batch LLM -> structured notes (personality evidence / language style /
      behavior / emotion triggers / relationships) with source-file citations
   4. Merge LLM -> profile_summary + dialogue_library + behavior_samples + evolution
   5. `reduce_result_to_draft()` maps the result onto the `PrisMateDraft` schema
   ↓
3. [Parser] JSON Response -> GraphQL Object (`PrisMateDraft`)
   ↓
4. [UI] Auto-fill React Form Fields

---

## 4. Character Persistence (CRUD)
**Goal:** Create and update character definitions.
**Flow:**
1. [UI] `CreateCharacterForm.tsx` (User clicks Save)
   ↓ GraphQL Mutation (`CREATE_CHARACTER` or `UPDATE_CHARACTER`)
2. [Client] `apolloClient.ts`
   ↓ POST /api/graphql/
3. [Schema] `graphql/schema.py` (Mutation Resolvers)
   ↓
4. [ORM] `models.py` (`Character` model)
   ↓
5. [DB] PostgreSQL (Persistent Storage)

---

## 3. Sessions History Sidebar Display
**Goal:** Show All Sessions in History Sidebar.
**Flow:**
1. [Client] `api.ts` (`getChatSessions`)
   ↓ GET /api/sessions/
2. [View] `views.py` -> `serializers.py` (Serialize DB Objects)
   ↓ JSON Response (Snake_Case)
3. [Client] `api.ts` (`normalizeSession`) -> **Data Normalization**
   ↓ CamelCase Data
4. [UI] Sidebar Render

---

## 5. Session Context Management
**Goal:** Persist dynamic settings (World Time, User Persona) for the session.
**Flow:**
1. [UI] `SessionSettings.tsx` (User updates settings)
   ↓
2. [Client] `api.ts` (`updateChatSession`)
   ↓ PATCH /api/sessions/{id}/
3. [View] `views.py` (`ChatSessionViewSet`)
   ↓
4. [Serializer] `serializers.py` (Validation)
   ↓
5. [DB] Update `ChatSession` row in PostgreSQL

---

## 6. History List Interface (Main View)
**Goal:** Display paginated, detailed list of all conversations.
**Flow:**
1. [UI] `page.tsx` (Switch view to 'history_all')
   ↓
2. [Data] Reuses `recentChats` state (Fetched via `api.getChatSessions`)
   ↓
3. [Render] Map through history items
   ↓
4. [Interaction] **Delete Session**
   ↓ `api.deleteChatSession` (DELETE /api/sessions/{id}/)
   ↓ Update State (Remove item from list)

---

## 7.1 Message Restoration (Chat Detail)
**Goal:** Reload previous messages when entering an old session.
**Flow:**
1. [User] Selects a Chat ID
   ↓
2. [UI] `ChatInterface.tsx` (useEffect trigger)
   ↓ Parallel Requests
3. [Client] `api.ts`
   ├── `getMessages(id)` (GET /api/messages/?chat_session_id=X)
   └── `getChatSession(id)` (GET /api/sessions/{id}/)
   ↓
4. [Backend] `views.py` (`MessageViewSet`) filters by Session ID
   ↓
5. [State] **Redux Store** (`setMessages`, `setChatSession`)
   ↓
6. [UI] `ChatWindow.tsx` (Re-renders message bubbles)

---

## 7.2 Chat Initialization
**Goal:** Load character details and restore chat history when entering a room.
**Location:** `frontend/src/components/ChatInterface.tsx`

**Flow A: Character Context (for NEW chat)**
1. [Trigger] `useEffect` (on `characterId` change)
   ↓
2. [Function] **`loadCharacter()`** (Internal Async)
   ↓
3. [Client] `api.ts` (`getCharacter`) -> GET /api/characters/{id}/
   ↓
4. [State] `dispatch(setCharacter)` -> Update Redux

**Flow B: History Restoration (for OLD chat)**
1. [Trigger] `useEffect` (on `initialSessionId` change)
   ↓
2. [Function] **`loadChatHistory()`** (Internal Async)
   ↓
3. [Client] Parallel Fetch via `api.ts`:
   ├── `getMessages(id)` -> GET /api/messages/
   └── `getChatSession(id)` -> GET /api/sessions/{id}/
   ↓
4. [State] `dispatch(setMessages)` & `dispatch(setChatSession)`

---

## 8. Delete Conversation (REST)
**Goal:** Remove specific chat history and cleanup database.
**Flow:**
1. [UI] `page.tsx` (User clicks Trash icon in Sidebar)
   ↓
2. [Client] `api.ts` (`deleteChatSession`)
   ↓ DELETE /api/sessions/{id}/
3. [View] `views.py` (`ChatSessionViewSet`)
   ↓ (Inherited `destroy` method)
4. [DB] PostgreSQL
   **[Cascade Delete]** Removes Session AND all associated Messages automatically.
   ↓
5. [UI] State Update (Filter out item from `recentChats` list)

---

## 9. Delete Character (GraphQL + Validation)
**Goal:** Safely remove a character while preserving data integrity.
**Flow:**
1. [UI] `CharacterGallery.tsx` (User clicks Delete)
   ↓ GraphQL Mutation (`DELETE_CHARACTER`)
2. [Schema] `graphql/schema.py` (`delete_character` resolver)
   ↓
   **[Validation Check]**
   `if character.chat_sessions.exists(): return False`
   *(Prevents deleting characters that have active history)*
   **Referential Integrity**
   ↓
3. [DB] PostgreSQL (Delete Row)
   ↓
4. [UI] Refetch List / Remove Card






















