import type {
  DocumentPage,
  DocumentSummary,
  ExplorerSettings,
  LoadedDataset,
  PreparationJob,
  PreparedDatasetInventory,
  DatasetOption,
} from "@/types"

type CreateJobRequest = {
  name: string
  source_dir: string | null
  output_root: string | null
  config: Record<string, unknown>
}

type DocumentsResponse = {
  documents: DocumentSummary[]
  page: number
  page_size: number
  total: number
  pages: number
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  })

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    try {
      const payload = (await response.json()) as { detail?: unknown }
      if (typeof payload.detail === "string") message = payload.detail
    } catch {
      // Keep the HTTP status when the server did not return JSON.
    }
    throw new Error(message)
  }

  return (await response.json()) as T
}

function segment(value: string) {
  return encodeURIComponent(value)
}

export const api = {
  settings: () => request<ExplorerSettings>("/api/settings"),
  updateSettings: (baseDir: string | null) =>
    request<ExplorerSettings>("/api/settings", {
      method: "PUT",
      body: JSON.stringify({ base_dir: baseDir }),
    }),
  inventory: (force = false) =>
    request<PreparedDatasetInventory>(`/api/prepared-datasets?force=${force}`),
  preparationOptions: () =>
    request<{ datasets: DatasetOption[] }>("/api/preparation-options"),
  jobs: () => request<{ jobs: PreparationJob[] }>("/api/jobs"),
  createJob: (job: CreateJobRequest) =>
    request<PreparationJob>("/api/jobs", {
      method: "POST",
      body: JSON.stringify(job),
    }),
  dataset: (datasetId: string) =>
    request<LoadedDataset>(`/api/datasets/${segment(datasetId)}`),
  documents: (
    datasetId: string,
    split: string,
    page: number,
    search: string,
  ) => {
    const query = new URLSearchParams({ split, page: String(page) })
    if (search) query.set("search", search)
    return request<DocumentsResponse>(
      `/api/datasets/${segment(datasetId)}/documents?${query.toString()}`,
    )
  },
  document: (datasetId: string, split: string, index: number) =>
    request<DocumentPage>(
      `/api/datasets/${segment(datasetId)}/documents/${segment(split)}/${index}`,
    ),
}
