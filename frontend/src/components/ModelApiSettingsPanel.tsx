"use client";

import { useEffect, useMemo, useState } from 'react';
import { ModelConfig, ModelProvider, ModelRoleAssignments, ModelRoleKey, WebSearchConfig, WebSearchTestResult } from '@/types';
import { apiService } from '@/utils/api';
import { DEFAULT_PROJECT_MODEL_NAME } from '@/constants';
import { PROVIDER_PRESETS, matchProviderPreset } from '@/constants/providerPresets';
import { AlertCircle, CheckCircle2, Cpu, Globe, Pencil, PlusCircle, Save, Search, Trash2 } from 'lucide-react';
import { useI18n } from '@/i18n/provider';

interface ModelApiSettingsPanelProps {
  modelConfigs: ModelConfig[];
  modelRoles: ModelRoleAssignments | null;
  webSearchConfig: WebSearchConfig | null;
  loading: boolean;
  loadingRoles: boolean;
  loadingWebSearchConfig: boolean;
  error?: string | null;
  webSearchError?: string | null;
  onRefresh: () => Promise<void>;
  onRefreshModelRoles: () => Promise<void>;
  onRefreshWebSearchConfig: () => Promise<void>;
}

interface ModelConfigFormState {
  name: string;
  provider: ModelProvider;
  presetId: string;
  modelName: string;
  apiKey: string;
  baseUrl: string;
  contextWindow: string;
}

interface WebSearchFormState {
  provider: 'tavily';
  apiKey: string;
  maxResults: number;
}

const ROLE_KEYS: ModelRoleKey[] = ['text', 'image', 'audio', 'video'];

const EMPTY_FORM: ModelConfigFormState = {
  name: '',
  provider: 'openai_compatible',
  presetId: 'custom',
  modelName: DEFAULT_PROJECT_MODEL_NAME,
  apiKey: '',
  baseUrl: '',
  contextWindow: '',
};

const EMPTY_WEB_SEARCH_FORM: WebSearchFormState = {
  provider: 'tavily',
  apiKey: '',
  maxResults: 5,
};

export default function ModelApiSettingsPanel({
  modelConfigs,
  modelRoles,
  webSearchConfig,
  loading,
  loadingRoles,
  loadingWebSearchConfig,
  error: externalError,
  webSearchError: externalWebSearchError,
  onRefresh,
  onRefreshModelRoles,
  onRefreshWebSearchConfig,
}: ModelApiSettingsPanelProps) {
  const { messages } = useI18n();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [formState, setFormState] = useState<ModelConfigFormState>(EMPTY_FORM);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [probeState, setProbeState] = useState<{ busy: boolean; models: string[]; error: string | null }>({
    busy: false,
    models: [],
    error: null,
  });
  const [roleSelection, setRoleSelection] = useState<Record<ModelRoleKey, string>>({
    text: '',
    image: '',
    audio: '',
    video: '',
  });
  const [isSavingRoles, setIsSavingRoles] = useState(false);
  const [rolesError, setRolesError] = useState<string | null>(null);
  const [webSearchFormState, setWebSearchFormState] = useState<WebSearchFormState>(EMPTY_WEB_SEARCH_FORM);
  const [isSavingWebSearch, setIsSavingWebSearch] = useState(false);
  const [isTestingWebSearch, setIsTestingWebSearch] = useState(false);
  const [webSearchLocalError, setWebSearchLocalError] = useState<string | null>(null);
  const [testQuery, setTestQuery] = useState('');
  const [testResult, setTestResult] = useState<WebSearchTestResult | null>(null);

  const sortedConfigs = useMemo(
    () => [...modelConfigs].sort((a, b) => a.name.localeCompare(b.name)),
    [modelConfigs]
  );

  useEffect(() => {
    if (!modelRoles) return;
    setRoleSelection({
      text: modelRoles.text?.id ?? '',
      image: modelRoles.image?.id ?? '',
      audio: modelRoles.audio?.id ?? '',
      video: modelRoles.video?.id ?? '',
    });
  }, [modelRoles]);

  useEffect(() => {
    setWebSearchFormState({
      provider: webSearchConfig?.provider || 'tavily',
      apiKey: webSearchConfig?.apiKey || '',
      maxResults: webSearchConfig?.maxResults || 5,
    });
  }, [webSearchConfig]);

  const startCreate = () => {
    setEditingId(null);
    setError(null);
    setProbeState({ busy: false, models: [], error: null });
    setFormState({ ...EMPTY_FORM });
  };

  const startEdit = (config: ModelConfig) => {
    setEditingId(config.id);
    setError(null);
    setProbeState({ busy: false, models: [], error: null });
    const preset = matchProviderPreset(config.provider, config.baseUrl || '');
    setFormState({
      name: config.name,
      provider: config.provider,
      presetId: preset?.id || 'custom',
      modelName: config.modelName,
      apiKey: config.apiKey,
      baseUrl: config.baseUrl || '',
      contextWindow: config.contextWindow ? String(config.contextWindow) : '',
    });
  };

  const applyPreset = (presetId: string) => {
    const preset = PROVIDER_PRESETS.find((item) => item.id === presetId);
    if (!preset) {
      setFormState((prev) => ({ ...prev, presetId: 'custom' }));
      return;
    }
    setFormState((prev) => ({
      ...prev,
      presetId,
      provider: preset.apiType,
      ...(preset.apiType === 'gemini' ? {} : { baseUrl: preset.defaultBaseUrl }),
    }));
  };

  const handleProbeModels = async () => {
    setProbeState({ busy: true, models: [], error: null });
    try {
      const response = await apiService.probeProviderModels({
        provider: formState.provider,
        baseUrl: formState.baseUrl,
        apiKey: formState.apiKey,
      });
      if (response.error || !response.data) {
        throw new Error(response.error || messages.modelApi.fetchModelsFailed);
      }
      setProbeState({ busy: false, models: response.data.models, error: null });
      if (response.data.models.length === 0) {
        setProbeState({ busy: false, models: [], error: messages.modelApi.fetchModelsEmpty });
      }
    } catch (probeError) {
      setProbeState({
        busy: false,
        models: [],
        error: probeError instanceof Error ? probeError.message : messages.modelApi.fetchModelsFailed,
      });
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    setError(null);

    const trimmedContextWindow = formState.contextWindow.trim();
    const parsedContextWindow = trimmedContextWindow ? Number(trimmedContextWindow) : null;
    if (parsedContextWindow !== null && (!Number.isFinite(parsedContextWindow) || parsedContextWindow <= 0)) {
      setError(messages.modelApi.contextWindowInvalid);
      setIsSaving(false);
      return;
    }

    const payload = {
      name: formState.name.trim(),
      provider: formState.provider,
      model_name: formState.modelName.trim(),
      api_key: formState.apiKey.trim(),
      base_url: formState.baseUrl.trim(),
      context_window: parsedContextWindow !== null ? Math.round(parsedContextWindow) : null,
    };

    try {
      const response = editingId
        ? await apiService.updateModelConfig(editingId, payload)
        : await apiService.createModelConfig(payload);

      if (response.error) {
        throw new Error(response.error);
      }

      await onRefresh();
      if (response.data) {
        startEdit(response.data);
      } else {
        startCreate();
      }
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : messages.modelApi.saveModelConfigurationFailed);
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async (config: ModelConfig) => {
    setRolesError(null);
    if (!window.confirm(messages.modelApi.deleteModelConfiguration(config.name))) {
      return;
    }

    setIsSaving(true);
    setError(null);

    try {
      const response = await apiService.deleteModelConfig(config.id);

      if (response.error) {
        throw new Error(response.error);
      }

      await onRefresh();
      await onRefreshModelRoles();
      if (editingId === config.id) {
        startCreate();
      }
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : messages.modelApi.deleteModelConfigurationFailed);
    } finally {
      setIsSaving(false);
    }
  };

  const handleSaveRoles = async () => {
    if (!roleSelection.text) {
      setRolesError(messages.modelApi.roleTextRequired);
      return;
    }
    setIsSavingRoles(true);
    setRolesError(null);

    try {
      const response = await apiService.updateModelRoles({
        text: roleSelection.text || null,
        image: roleSelection.image || null,
        audio: roleSelection.audio || null,
        video: roleSelection.video || null,
      });
      if (response.error) {
        throw new Error(response.error);
      }
      await onRefreshModelRoles();
    } catch (saveRolesError) {
      setRolesError(saveRolesError instanceof Error ? saveRolesError.message : messages.modelApi.saveRolesFailed);
    } finally {
      setIsSavingRoles(false);
    }
  };

  const handleSaveWebSearch = async () => {
    setIsSavingWebSearch(true);
    setWebSearchLocalError(null);

    try {
      const response = await apiService.updateWebSearchConfig({
        provider: webSearchFormState.provider,
        api_key: webSearchFormState.apiKey.trim(),
        max_results: webSearchFormState.maxResults,
      });

      if (response.error) {
        throw new Error(response.error);
      }

      await onRefreshWebSearchConfig();
    } catch (saveError) {
      setWebSearchLocalError(saveError instanceof Error ? saveError.message : messages.modelApi.saveWebSearchConfigurationFailed);
    } finally {
      setIsSavingWebSearch(false);
    }
  };

  const handleTestWebSearch = async () => {
    const normalizedQuery = testQuery.trim();
    if (!normalizedQuery) {
      setWebSearchLocalError(messages.modelApi.testQueryRequired);
      return;
    }

    setIsTestingWebSearch(true);
    setWebSearchLocalError(null);
    setTestResult(null);

    try {
      const response = await apiService.testWebSearchConfig(normalizedQuery);
      if (response.error) {
        throw new Error(response.error);
      }
      setTestResult(response.data || null);
    } catch (testError) {
      setWebSearchLocalError(testError instanceof Error ? testError.message : messages.modelApi.testWebSearchFailed);
    } finally {
      setIsTestingWebSearch(false);
    }
  };

  const roleLabel = (role: ModelRoleKey) => ({
    text: messages.modelApi.roleText,
    image: messages.modelApi.roleImage,
    audio: messages.modelApi.roleAudio,
    video: messages.modelApi.roleVideo,
  }[role]);

  const roleHintFor = (role: ModelRoleKey): string | null => {
    const selectedId = roleSelection[role];
    const config = modelConfigs.find((item) => item.id === selectedId);
    if (!config) return null;
    const preset = matchProviderPreset(config.provider, config.baseUrl || '');
    return preset?.roleHints?.[role] || null;
  };

  return (
    <div className="h-full overflow-y-auto bg-slate-50">
      <div className="mx-auto max-w-6xl px-6 py-8 md:px-10">
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-slate-900">{messages.modelApi.title}</h2>
          <p className="mt-1 text-sm text-slate-500">{messages.modelApi.subtitle}</p>
        </div>

        <div className="mb-6 rounded-2xl border border-sky-100 bg-sky-50/80 px-5 py-4 text-sm text-slate-700">
          <p className="font-medium text-slate-900">{messages.modelApi.ownershipTitle}</p>
          <p className="mt-1 text-slate-600">{messages.modelApi.ownershipDescription}</p>
        </div>

        {(error || externalError) && (
          <div className="mb-6 flex items-center gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            <AlertCircle size={16} />
            <span>{error || externalError}</span>
          </div>
        )}

        <div className="space-y-8">
          <section className="rounded-[2rem] border border-slate-200 bg-white/80 p-6 shadow-sm">
            <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
              <div>
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
                  <Cpu size={16} />
                  <span>{messages.modelApi.llmSectionTitle}</span>
                </div>
                <p className="mt-2 text-sm text-slate-500">{messages.modelApi.llmSectionDescription}</p>
              </div>
              <button
                onClick={startCreate}
                className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-blue-700"
              >
                <PlusCircle size={16} />
                <span>{messages.modelApi.newModel}</span>
              </button>
            </div>

            <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
              <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
                <div className="border-b border-slate-100 px-5 py-4">
                  <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
                    <Cpu size={16} />
                    <span>{messages.modelApi.configuredModels}</span>
                  </div>
                </div>

                {loading ? (
                  <div className="px-5 py-10 text-sm text-slate-500">{messages.modelApi.loadingModelConfigurations}</div>
                ) : sortedConfigs.length === 0 ? (
                  <div className="px-5 py-10 text-sm text-slate-500">
                    {messages.modelApi.emptyModels}
                    <div className="mt-3 text-xs text-slate-400">{messages.modelApi.emptyModelsHelp}</div>
                  </div>
                ) : (
                  <div className="divide-y divide-slate-100">
                    {sortedConfigs.map((config) => {
                      const assignedRoles = ROLE_KEYS.filter((role) => modelRoles?.[role]?.id === config.id);

                      return (
                        <div key={config.id} className="flex items-center justify-between gap-4 px-5 py-4">
                          <div className="min-w-0">
                            <div className="flex flex-wrap items-center gap-2">
                              <h3 className="truncate font-medium text-slate-900">{config.name}</h3>
                              {assignedRoles.map((role) => (
                                <span
                                  key={role}
                                  className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700"
                                >
                                  {roleLabel(role)}
                                </span>
                              ))}
                            </div>
                            <div className="mt-1 text-sm text-slate-500">
                              {config.provider} / {config.modelName}
                              {config.contextWindow ? (
                                <span className="ml-2 rounded-full bg-sky-50 px-2 py-0.5 text-xs font-medium text-sky-700">
                                  {config.contextWindow.toLocaleString()} tokens
                                </span>
                              ) : null}
                            </div>
                            {config.baseUrl && (
                              <div className="mt-1 truncate text-xs text-slate-400">{config.baseUrl}</div>
                            )}
                          </div>

                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => startEdit(config)}
                              className="rounded-lg border border-slate-200 p-2 text-slate-500 transition-colors hover:bg-slate-50 hover:text-slate-800"
                              title={messages.modelApi.editModel}
                            >
                              <Pencil size={16} />
                            </button>
                            <button
                              onClick={() => handleDelete(config)}
                              className="rounded-lg border border-red-100 p-2 text-red-500 transition-colors hover:bg-red-50"
                              title={messages.modelApi.deleteModel}
                            >
                              <Trash2 size={16} />
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="mb-5">
                  <h3 className="text-lg font-semibold text-slate-900">
                    {editingId ? messages.modelApi.editModelConfiguration : messages.modelApi.createModelConfiguration}
                  </h3>
                  <p className="mt-1 text-sm text-slate-500">{messages.modelApi.configurationHelp}</p>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="mb-1.5 block text-sm font-medium text-slate-700">{messages.modelApi.displayName}</label>
                    <input
                      type="text"
                      value={formState.name}
                      onChange={(e) => setFormState((prev) => ({ ...prev, name: e.target.value }))}
                      placeholder={messages.modelApi.displayNamePlaceholder}
                      className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                    />
                  </div>

                  <div>
                    <label className="mb-1.5 block text-sm font-medium text-slate-700">{messages.modelApi.providerPreset}</label>
                    <select
                      value={formState.presetId}
                      onChange={(e) => applyPreset(e.target.value)}
                      className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                    >
                      {PROVIDER_PRESETS.map((preset) => (
                        <option key={preset.id} value={preset.id}>
                          {preset.label}
                          {preset.defaultBaseUrl ? ` · ${preset.defaultBaseUrl}` : ''}
                        </option>
                      ))}
                    </select>
                    <p className="mt-1 text-xs text-slate-500">{messages.modelApi.providerPresetHint}</p>
                  </div>

                  <div>
                    <label className="mb-1.5 block text-sm font-medium text-slate-700">{messages.modelApi.provider}</label>
                    <select
                      value={formState.provider}
                      onChange={(e) => setFormState((prev) => ({ ...prev, provider: e.target.value as ModelProvider }))}
                      className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                    >
                      <option value="openai_compatible">{messages.modelApi.providerOptions.openaiCompatible}</option>
                      <option value="gemini">{messages.modelApi.providerOptions.gemini}</option>
                      <option value="anthropic">{messages.modelApi.providerOptions.anthropic}</option>
                    </select>
                  </div>

                  <div>
                    <label className="mb-1.5 block text-sm font-medium text-slate-700">{messages.modelApi.modelName}</label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        list="probe-model-options"
                        value={formState.modelName}
                        onChange={(e) => setFormState((prev) => ({ ...prev, modelName: e.target.value }))}
                        placeholder={messages.modelApi.modelNamePlaceholder}
                        className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                      />
                      <datalist id="probe-model-options">
                        {probeState.models.map((model) => (
                          <option key={model} value={model} />
                        ))}
                      </datalist>
                      <button
                        type="button"
                        onClick={handleProbeModels}
                        disabled={probeState.busy}
                        className="inline-flex flex-shrink-0 items-center gap-1.5 rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        <Search size={14} />
                        <span>{probeState.busy ? messages.modelApi.fetchingModels : messages.modelApi.fetchModels}</span>
                      </button>
                    </div>
                    {probeState.error && (
                      <p className="mt-1 text-xs text-amber-600">{probeState.error}</p>
                    )}
                  </div>

                  <div>
                    <label className="mb-1.5 block text-sm font-medium text-slate-700">{messages.modelApi.contextWindow}</label>
                    <input
                      type="text"
                      inputMode="numeric"
                      value={formState.contextWindow}
                      onChange={(e) => setFormState((prev) => ({ ...prev, contextWindow: e.target.value.replace(/[^0-9]/g, '') }))}
                      placeholder={messages.modelApi.contextWindowPlaceholder}
                      className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                    />
                    <p className="mt-1 text-xs text-slate-500">{messages.modelApi.contextWindowHint}</p>
                  </div>

                  <div>
                    <label className="mb-1.5 block text-sm font-medium text-slate-700">{messages.modelApi.apiKey}</label>
                    <input
                      type="password"
                      value={formState.apiKey}
                      onChange={(e) => setFormState((prev) => ({ ...prev, apiKey: e.target.value }))}
                      placeholder={messages.modelApi.apiKeyPlaceholder}
                      className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                    />
                    {!formState.apiKey.trim() && (
                      <p className="mt-1 text-xs text-slate-500">{messages.modelApi.apiKeyOptionalHint}</p>
                    )}
                  </div>

                  <div>
                    <label className="mb-1.5 block text-sm font-medium text-slate-700">{messages.modelApi.baseUrl}</label>
                    <input
                      type="text"
                      value={formState.baseUrl}
                      onChange={(e) => setFormState((prev) => ({ ...prev, baseUrl: e.target.value }))}
                      placeholder={formState.provider === 'gemini' ? messages.modelApi.optional : messages.modelApi.baseUrlPlaceholderOpenAI}
                      disabled={formState.provider === 'gemini'}
                      className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 disabled:bg-slate-100"
                    />
                    <p className="mt-1 text-xs text-slate-500">
                      {formState.provider === 'openai_compatible'
                        ? messages.modelApi.baseUrlHintOpenAI
                        : formState.provider === 'anthropic'
                          ? messages.modelApi.baseUrlHintAnthropic
                          : messages.modelApi.baseUrlHintGemini}
                    </p>
                  </div>

                  <div className="flex gap-3 pt-2">
                    <button
                      onClick={handleSave}
                      disabled={isSaving || !formState.name.trim() || !formState.modelName.trim()}
                      className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <Save size={16} />
                      <span>{isSaving ? messages.modelApi.saving : editingId ? messages.modelApi.saveChanges : messages.modelApi.createModel}</span>
                    </button>
                    <button
                      onClick={startCreate}
                      className="rounded-lg border border-slate-300 px-4 py-2.5 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-50"
                    >
                      {messages.modelApi.reset}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="rounded-[2rem] border border-slate-200 bg-white/80 p-6 shadow-sm">
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-700">
              <CheckCircle2 size={16} />
              <span>{messages.modelApi.rolesSectionTitle}</span>
            </div>
            <p className="mb-5 text-sm text-slate-500">{messages.modelApi.rolesSectionDescription}</p>

            {rolesError && (
              <div className="mb-4 flex items-center gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                <AlertCircle size={16} />
                <span>{rolesError}</span>
              </div>
            )}

            {loadingRoles ? (
              <div className="rounded-xl border border-slate-200 bg-white px-4 py-6 text-sm text-slate-500">
                {messages.modelApi.loadingRoles}
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-2">
                {ROLE_KEYS.map((role) => {
                  const hint = roleHintFor(role);
                  const required = role === 'text';

                  return (
                    <div key={role}>
                      <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-slate-700">
                        {roleLabel(role)}
                        {required && (
                          <span className="rounded-full bg-blue-50 px-2 py-0.5 text-xs font-medium text-blue-600">
                            {messages.modelApi.roleRequired}
                          </span>
                        )}
                      </label>
                      <select
                        value={roleSelection[role]}
                        onChange={(e) => setRoleSelection((prev) => ({ ...prev, [role]: e.target.value }))}
                        className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                      >
                        <option value="">{messages.modelApi.roleUnassigned}</option>
                        {sortedConfigs.map((config) => (
                          <option key={config.id} value={config.id}>
                            {config.name}（{config.provider} / {config.modelName}）
                          </option>
                        ))}
                      </select>
                      {required && !roleSelection[role] && (
                        <p className="mt-1 text-xs text-amber-600">{messages.modelApi.roleTextRequired}</p>
                      )}
                      {!required && !roleSelection[role] && (
                        <p className="mt-1 text-xs text-slate-400">{messages.modelApi.roleEmptyHint}</p>
                      )}
                      {hint && (
                        <p className="mt-1 text-xs text-slate-400">{messages.modelApi.roleHint(hint)}</p>
                      )}
                    </div>
                  );
                })}
                <div className="md:col-span-2">
                  <button
                    onClick={handleSaveRoles}
                    disabled={isSavingRoles}
                    className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    <Save size={16} />
                    <span>{isSavingRoles ? messages.modelApi.saving : messages.modelApi.saveRoles}</span>
                  </button>
                </div>
              </div>
            )}
            <p className="mt-4 text-xs text-slate-400">{messages.modelApi.audioFormatNote}</p>
          </section>

          <section className="rounded-[2rem] border border-slate-200 bg-white/80 p-6 shadow-sm">
            <div className="mb-6 flex items-start gap-3">
              <div className="rounded-full bg-sky-50 p-2 text-sky-700">
                <Globe size={18} />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-slate-900">{messages.modelApi.webSearchSectionTitle}</h3>
                <p className="mt-1 text-sm text-slate-500">{messages.modelApi.webSearchSectionDescription}</p>
              </div>
            </div>

            {(webSearchLocalError || externalWebSearchError) && (
              <div className="mb-4 flex items-center gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                <AlertCircle size={16} />
                <span>{webSearchLocalError || externalWebSearchError}</span>
              </div>
            )}

            <div className="grid gap-6 lg:grid-cols-[0.78fr_1.22fr]">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="mb-4">
                  <h4 className="text-base font-semibold text-slate-900">{messages.modelApi.currentWebSearchConfiguration}</h4>
                  <p className="mt-1 text-sm text-slate-500">{messages.modelApi.currentWebSearchConfigurationHelp}</p>
                </div>

                {loadingWebSearchConfig ? (
                  <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-5 text-sm text-slate-500">
                    {messages.modelApi.loadingWebSearchConfiguration}
                  </div>
                ) : webSearchConfig?.apiKey ? (
                  <div className="space-y-3">
                    <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800">
                      {messages.modelApi.webSearchConfigured}
                    </div>
                    <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                      <p className="text-xs font-medium uppercase tracking-[0.16em] text-slate-400">{messages.modelApi.webSearchProvider}</p>
                      <p className="mt-2 text-sm font-medium text-slate-900">
                        {webSearchConfig.provider === 'tavily'
                          ? messages.modelApi.webSearchProviderOptions.tavily
                          : webSearchConfig.provider}
                      </p>
                    </div>
                    <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                      <p className="text-xs font-medium uppercase tracking-[0.16em] text-slate-400">{messages.modelApi.webSearchMaxResults}</p>
                      <p className="mt-2 text-sm font-medium text-slate-900">{webSearchConfig.maxResults}</p>
                    </div>
                    <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                      <p className="text-xs font-medium uppercase tracking-[0.16em] text-slate-400">{messages.modelApi.webSearchApiKey}</p>
                      <p className="mt-2 text-sm font-medium text-slate-900">{messages.modelApi.webSearchApiKeyMasked}</p>
                    </div>
                  </div>
                ) : (
                  <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-500">
                    {messages.modelApi.webSearchNotConfigured}
                  </div>
                )}
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="mb-5">
                  <h3 className="text-lg font-semibold text-slate-900">{messages.modelApi.webSearchTitle}</h3>
                  <p className="mt-1 text-sm text-slate-500">{messages.modelApi.webSearchDescription}</p>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="mb-1.5 block text-sm font-medium text-slate-700">{messages.modelApi.webSearchApiKey}</label>
                    <input
                      type="password"
                      value={webSearchFormState.apiKey}
                      onChange={(e) => setWebSearchFormState((prev) => ({ ...prev, apiKey: e.target.value }))}
                      placeholder={messages.modelApi.webSearchApiKeyPlaceholder}
                      className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                    />
                    <p className="mt-1 text-xs text-slate-500">{messages.modelApi.webSearchApiKeyHelp}</p>
                  </div>

                  <div>
                    <label className="mb-1.5 block text-sm font-medium text-slate-700">{messages.modelApi.webSearchMaxResults}</label>
                    <input
                      type="number"
                      min={1}
                      max={10}
                      value={webSearchFormState.maxResults}
                      onChange={(e) => {
                        const nextValue = Math.min(10, Math.max(1, Number(e.target.value || 1)));
                        setWebSearchFormState((prev) => ({ ...prev, maxResults: nextValue }));
                      }}
                      className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                    />
                    <p className="mt-1 text-xs text-slate-500">{messages.modelApi.webSearchMaxResultsHelp}</p>
                  </div>

                  <div className="flex pt-2">
                    <button
                      onClick={handleSaveWebSearch}
                      disabled={isSavingWebSearch || !webSearchFormState.apiKey.trim()}
                      className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg bg-sky-600 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-sky-700 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <Save size={16} />
                      <span>{isSavingWebSearch ? messages.modelApi.saving : messages.modelApi.saveWebSearchConfiguration}</span>
                    </button>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-800">
                      <Search size={16} />
                      <span>{messages.modelApi.testWebSearchTitle}</span>
                    </div>
                    <div className="space-y-3">
                      <input
                        type="text"
                        value={testQuery}
                        onChange={(e) => setTestQuery(e.target.value)}
                        placeholder={messages.modelApi.testQueryPlaceholder}
                        className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                      />
                      <button
                        onClick={handleTestWebSearch}
                        disabled={isTestingWebSearch || !testQuery.trim()}
                        className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        <Search size={16} />
                        <span>{isTestingWebSearch ? messages.modelApi.testingWebSearch : messages.modelApi.testWebSearchAction}</span>
                      </button>
                    </div>

                    {testResult && (
                      <div className="mt-4 space-y-3">
                        <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600">
                          <span className="font-medium text-slate-800">{messages.modelApi.testWebSearchSummary}</span>{' '}
                          {testResult.provider || 'tavily'}
                          {testResult.query ? ` · ${testResult.query}` : ''}
                        </div>
                        {testResult.error ? (
                          <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
                            {testResult.error}
                          </div>
                        ) : testResult.items.length === 0 ? (
                          <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600">
                            {messages.modelApi.testWebSearchEmpty}
                          </div>
                        ) : (
                          <div className="space-y-2">
                            {testResult.items.map((item, index) => (
                              <div key={`${item.url}-${index}`} className="rounded-lg border border-slate-200 bg-white px-3 py-3">
                                <a
                                  href={item.url}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="text-sm font-medium text-sky-700 hover:text-sky-800"
                                >
                                  {item.title}
                                </a>
                                <p className="mt-1 break-all text-xs text-slate-400">{item.url}</p>
                                {item.snippet && (
                                  <p className="mt-2 text-sm leading-6 text-slate-600">{item.snippet}</p>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
