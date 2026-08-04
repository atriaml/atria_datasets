import { useEffect, useMemo, useState } from "react"
import {
  AlertTriangle,
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
  Loader2,
} from "lucide-react"
import { Link, useNavigate, useParams } from "react-router-dom"
import { Alert } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { api } from "@/lib/api"
import type { DocumentPage as DocumentPageType, LoadedDataset } from "@/types"

const levelColors: Record<string, string> = {
  page: "bg-violet-100 text-violet-800",
  block: "bg-blue-100 text-blue-800",
  paragraph: "bg-cyan-100 text-cyan-800",
  line: "bg-green-100 text-green-800",
  word: "bg-red-100 text-red-800",
}

export function DocumentPage() {
  const { datasetId = "", split = "", index = "0" } = useParams()
  const navigate = useNavigate()
  const documentIndex = Number(index)
  const [document, setDocument] = useState<DocumentPageType | null>(null)
  const [dataset, setDataset] = useState<LoadedDataset | null>(null)
  const [overlay, setOverlay] = useState(true)
  const [levels, setLevels] = useState(new Set(["line", "word"]))
  const [error, setError] = useState("")

  useEffect(() => {
    setDocument(null)
    setError("")
    Promise.all([api.document(datasetId, split, documentIndex), api.dataset(datasetId)])
      .then(([doc, current]) => {
        setDocument(doc)
        setDataset(current)
      })
      .catch((nextError: Error) => setError(nextError.message))
  }, [datasetId, split, documentIndex])

  const imageUrl = useMemo(() => {
    const params = new URLSearchParams({
      overlay: String(overlay),
      levels: [...levels].join(","),
    })
    return `/api/datasets/${datasetId}/documents/${split}/${documentIndex}/image?${params}`
  }, [datasetId, split, documentIndex, overlay, levels])

  if (error) {
    return (
      <main className="mx-auto max-w-screen-2xl px-5 py-10 lg:px-8">
        <Alert className="border-destructive/40 text-destructive">{error}</Alert>
        <Button className="mt-4" variant="outline" onClick={() => navigate("/")}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to inventory
        </Button>
      </main>
    )
  }

  if (!document || !dataset) {
    return (
      <div className="flex min-h-[70vh] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  const count = dataset.splits[split] || 0
  const toggleLevel = (level: string) =>
    setLevels((current) => {
      const next = new Set(current)
      if (next.has(level)) next.delete(level)
      else next.add(level)
      return next
    })

  return (
    <main className="mx-auto max-w-screen-2xl space-y-6 px-5 py-7 lg:px-8">
      <div className="flex flex-col justify-between gap-4 border-b pb-5 md:flex-row md:items-center">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={() => navigate(-1)} aria-label="Back">
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              {document.dataset} / {split}
            </p>
            <h1 className="mt-1 text-xl font-semibold">{document.sample_id}</h1>
          </div>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={documentIndex <= 0}
            onClick={() => navigate(`/datasets/${datasetId}/documents/${split}/${documentIndex - 1}`)}
          >
            <ChevronLeft className="mr-1 h-4 w-4" />
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={documentIndex >= count - 1}
            onClick={() => navigate(`/datasets/${datasetId}/documents/${split}/${documentIndex + 1}`)}
          >
            Next
            <ChevronRight className="ml-1 h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Card className="overflow-hidden">
          <CardHeader className="flex-row items-center justify-between space-y-0 border-b py-4">
            <CardTitle className="text-base">Document image</CardTitle>
            <div className="flex flex-wrap items-center justify-end gap-2">
              <Button variant={overlay ? "default" : "outline"} size="sm" onClick={() => setOverlay((value) => !value)}>
                {overlay ? <Eye className="mr-1.5 h-4 w-4" /> : <EyeOff className="mr-1.5 h-4 w-4" />}
                Overlays
              </Button>
              {["page", "block", "line", "word"].map((level) => (
                <button
                  key={level}
                  disabled={!overlay}
                  onClick={() => toggleLevel(level)}
                  className={`rounded-md px-2 py-1 text-xs font-medium capitalize transition disabled:opacity-40 ${
                    levels.has(level) ? levelColors[level] : "bg-muted text-muted-foreground"
                  }`}
                >
                  {level}
                </button>
              ))}
            </div>
          </CardHeader>
          <div className="flex min-h-[60vh] items-start justify-center overflow-auto bg-muted/40 p-4">
            <img
              key={imageUrl}
              src={imageUrl}
              alt={document.sample_id}
              className="max-h-[78vh] max-w-full object-contain shadow-sm"
            />
          </div>
        </Card>

        <aside className="space-y-5">
          {document.warnings.length > 0 ? (
            <Alert className="border-amber-300 bg-amber-50 text-amber-950">
              <div className="mb-2 flex items-center gap-2 font-medium">
                <AlertTriangle className="h-4 w-4" />
                Validation warnings
              </div>
              <ul className="list-disc space-y-1 pl-5 text-xs">
                {document.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </Alert>
          ) : null}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Metadata</CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="grid grid-cols-[110px_1fr] gap-x-3 gap-y-3 text-sm">
                <dt className="text-muted-foreground">Index</dt>
                <dd>
                  {document.index + 1} of {count}
                </dd>
                <dt className="text-muted-foreground">Dimensions</dt>
                <dd>
                  {document.width} × {document.height}
                </dd>
                <dt className="text-muted-foreground">Elements</dt>
                <dd>{document.ocr_row_count}</dd>
                <dt className="text-muted-foreground">Image path</dt>
                <dd className="break-all font-mono text-xs">{document.image_path || "In-memory image"}</dd>
              </dl>
              <div className="mt-5 flex flex-wrap gap-1.5">
                {Object.entries(document.annotation_counts).map(([name, value]) => (
                  <Badge key={name}>
                    {value} {name}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Transcription</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="max-h-[42vh] whitespace-pre-wrap break-words font-sans text-sm leading-6 text-foreground">
                {document.transcription || "No transcription available."}
              </pre>
            </CardContent>
          </Card>
        </aside>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            OCR hierarchy <span className="ml-2 font-normal text-muted-foreground">{document.ocr_row_count} elements</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="px-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-20">ID</TableHead>
                <TableHead className="w-24">Parent</TableHead>
                <TableHead className="w-28">Level</TableHead>
                <TableHead>Text</TableHead>
                <TableHead className="w-64">Normalized box</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {document.ocr_rows.map((row) => (
                <TableRow key={row.id}>
                  <TableCell>{row.id}</TableCell>
                  <TableCell>{row.parent_id ?? "—"}</TableCell>
                  <TableCell>
                    <Badge className={levelColors[String(row.level)]}>{row.level}</Badge>
                  </TableCell>
                  <TableCell className="max-w-xl whitespace-pre-wrap">
                    {row.text || <span className="text-muted-foreground">Empty</span>}
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    {row.bbox ? row.bbox.join(", ") : "—"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      <div className="pb-4">
        <Link to={`/datasets/${datasetId}`} className="text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline">
          Back to all documents
        </Link>
      </div>
    </main>
  )
}
