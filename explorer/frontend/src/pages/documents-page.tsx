import { useEffect, useState } from "react"
import { ChevronLeft, ChevronRight, Loader2, Search } from "lucide-react"
import { Link, useParams } from "react-router-dom"
import { DocumentCard } from "@/components/document-card"
import { Alert } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { api } from "@/shared/api"
import type { DocumentSummary, LoadedDataset } from "@/types"

export function DocumentsPage() {
  const { datasetId = "" } = useParams()
  const [dataset, setDataset] = useState<LoadedDataset | null>(null)
  const [split, setSplit] = useState("")
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [page, setPage] = useState(1)
  const [pages, setPages] = useState(1)
  const [total, setTotal] = useState(0)
  const [search, setSearch] = useState("")
  const [query, setQuery] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  useEffect(() => {
    setError("")
    api
      .dataset(datasetId)
      .then((current) => {
        setDataset(current)
        setSplit((existing) => existing || Object.keys(current.splits)[0] || "")
      })
      .catch((nextError: Error) => setError(nextError.message))
  }, [datasetId])

  useEffect(() => {
    if (!split) return
    setLoading(true)
    setError("")
    api
      .documents(datasetId, split, page, query)
      .then((result) => {
        setDocuments(result.documents)
        setPages(result.pages)
        setTotal(result.total)
      })
      .catch((nextError: Error) => setError(nextError.message))
      .finally(() => setLoading(false))
  }, [datasetId, split, page, query])

  useEffect(() => {
    setPage(1)
  }, [split, datasetId])

  if (error) {
    return (
      <main className="mx-auto max-w-screen-2xl px-5 py-10 lg:px-8">
        <Alert className="border-destructive/40 text-destructive">{error}</Alert>
        <Link to="/" className="mt-4 inline-block">
          <Button variant="outline">Back to inventory</Button>
        </Link>
      </main>
    )
  }

  if (!dataset) {
    return (
      <div className="flex min-h-[70vh] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <main className="mx-auto max-w-screen-2xl space-y-7 px-5 py-8 lg:px-8">
      <section className="flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
        <div>
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <h1 className="text-2xl font-semibold tracking-tight">{dataset.dataset_class_name}</h1>
            <Badge>{total} documents</Badge>
            <Badge className="bg-muted text-foreground">{dataset.storage_type}</Badge>
          </div>
          <p className="max-w-3xl text-sm text-muted-foreground">
            {dataset.metadata?.description || dataset.path}
          </p>
          <p className="mt-2 font-mono text-xs text-muted-foreground">{dataset.relative_path}</p>
        </div>
        <Link to="/">
          <Button variant="outline">Back to inventory</Button>
        </Link>
      </section>
      <section className="flex flex-col gap-4 border-b pb-5 md:flex-row md:items-center md:justify-between">
        <div className="flex gap-1 rounded-lg bg-muted p-1">
          {Object.entries(dataset.splits).map(([name, count]) => (
            <button
              key={name}
              onClick={() => setSplit(name)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
                split === name ? "bg-card shadow-sm" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {name} <span className="ml-1 text-xs">{count}</span>
            </button>
          ))}
        </div>
        <form
          className="relative w-full md:w-80"
          onSubmit={(event) => {
            event.preventDefault()
            setPage(1)
            setQuery(search)
          }}
        >
          <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
          <Input
            className="pl-9"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search ID or transcription"
          />
        </form>
      </section>
      {loading ? (
        <div className="flex min-h-80 items-center justify-center">
          <Loader2 className="h-7 w-7 animate-spin text-muted-foreground" />
        </div>
      ) : documents.length ? (
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
          {documents.map((document) => (
            <DocumentCard key={`${split}-${document.index}`} datasetId={datasetId} document={document} />
          ))}
        </div>
      ) : (
        <div className="flex min-h-80 items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">
          No documents found.
        </div>
      )}
      <div className="flex items-center justify-between border-t pt-5">
        <p className="text-sm text-muted-foreground">
          Page {page} of {pages}
        </p>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={page <= 1 || loading}
            onClick={() => setPage((value) => value - 1)}
          >
            <ChevronLeft className="mr-1 h-4 w-4" />
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={page >= pages || loading}
            onClick={() => setPage((value) => value + 1)}
          >
            Next
            <ChevronRight className="ml-1 h-4 w-4" />
          </Button>
        </div>
      </div>
    </main>
  )
}
