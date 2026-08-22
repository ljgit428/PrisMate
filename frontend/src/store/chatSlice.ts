import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { ChatState, Message, Character, ChatSession, ToolCallInfo } from '@/types';

const initialState: ChatState = {
  messages: [],
  character: null,
  chatSession: null,
  isLoading: false,
  error: null,
};

const chatSlice = createSlice({
  name: 'chat',
  initialState,
  reducers: {
    setMessages: (state, action: PayloadAction<Message[]>) => {
      state.messages = action.payload;
    },
    addMessage: (state, action: PayloadAction<Message>) => {
      state.messages.push(action.payload);
    },
    upsertMessage: (state, action: PayloadAction<Message>) => {
      const index = state.messages.findIndex((message) => message.id === action.payload.id);
      if (index >= 0) {
        state.messages[index] = action.payload;
        return;
      }
      state.messages.push(action.payload);
    },
    appendToMessage: (state, action: PayloadAction<{ id: string; content: string }>) => {
      const target = state.messages.find((message) => message.id === action.payload.id);
      if (target) {
        target.content += action.payload.content;
      }
    },
    appendToMessageThinking: (state, action: PayloadAction<{ id: string; content: string }>) => {
      const target = state.messages.find((message) => message.id === action.payload.id);
      if (target) {
        target.thinking = (target.thinking || '') + action.payload.content;
      }
    },
    appendToMessageToolCall: (state, action: PayloadAction<{ id: string; toolCall: ToolCallInfo }>) => {
      const target = state.messages.find((message) => message.id === action.payload.id);
      if (target) {
        target.toolCalls = [...(target.toolCalls || []), action.payload.toolCall];
      }
    },
    removeMessage: (state, action: PayloadAction<string>) => {
      state.messages = state.messages.filter((message) => message.id !== action.payload);
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setCharacter: (state, action: PayloadAction<Character | null>) => {
      state.character = action.payload;
    },
    setChatSession: (state, action: PayloadAction<ChatSession | null>) => {
      state.chatSession = action.payload;
    },
    updateChatSession: (state, action: PayloadAction<Partial<ChatSession>>) => {
      if (state.chatSession) {
        state.chatSession = { ...state.chatSession, ...action.payload };
      }
    },
    updateCharacter: (state, action: PayloadAction<Character>) => {
      if (state.character && state.character.id === action.payload.id) {
        state.character = action.payload;
      }
    },
    clearChat: (state) => {
      state.messages = [];
      state.error = null;
    },
  },
});

export const {
  setMessages,
  addMessage,
  upsertMessage,
  appendToMessage,
  appendToMessageThinking,
  appendToMessageToolCall,
  removeMessage,
  setLoading,
  setError,
  setCharacter,
  setChatSession,
  updateChatSession,
  updateCharacter,
  clearChat,
} = chatSlice.actions;

export default chatSlice.reducer;
