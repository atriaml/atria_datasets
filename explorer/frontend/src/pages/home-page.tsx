import { useEffect, useState } from "react"
import { AlertTriangle, Database, FolderSearch, RefreshCw } from "lucide-react"
import { Link } from "react-router-dom"
import { Alert } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { api } from "@/lib/api"
import type { ExplorerSettings, PreparedDatasetInventory } from "@/types"

function formatDate(value: string | null) {
  if (!value) return "Unknown"
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString()
}

function normalizeOptionalPath(value: string | null | undefined) {
  if (!value) return ""
  const normalized = value.trim()
  return normalized.toLowerCase() === "none" || normalized.toLowerCase() === "null"
    ? ""
    : normalized
}

export function HomePage() {
  const [inventory, setInventory] = useState<PreparedDatasetInventory | null>(null)
  const [settings, setSettings] = useState<ExplorerSettings | null>(null)
  const [baseDirInput, setBaseDirInput] = useState("")
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState("")

  const load = async (force = false) => {
    setLoading(true)
    setError("")
    try {
      const nextSettings = await api.settings()
      setSettings(nextSettings)
      setBaseDirInput(normalizeOptionalPath(nextSettings.base_dir))

      const next = await api.inventory(force)
      setInventory(next)
      setBaseDirInput(normalizeOptionalPath(next.base_dir))
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : String(nextError))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const saveBaseDir = async () => {
    setSaving(true)
    setError("")
    try {
      const normalized = normalizeOptionalPath(baseDirInput)
      await api.updateSettings(normalized || null)
      await load(true)
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : String(nextError))
    } finally {
      setSaving(false)
    }
  }

  return (
    <main className="mx-auto max-w-screen-2xl space-y-7 px-5 py-8 lg:px-8">
      <section className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(320px,0.9fr)]">
        <Card className="border-border/70 bg-card/90">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl">
              <FolderSearch className="h-5 w-5" />
              Prepared dataset inventory
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="max-w-3xl text-sm text-muted-foreground">
              The explorer scans one output root for valid cached dataset manifests. Empty
              means the default Atria cache directory is used.
            </p>
            <div className="grid gap-2">
              <label className="text-sm font-medium">Output root</label>
              <Input
                value={baseDirInput}
                onChange={(event) => setBaseDirInput(event.target.value)}
                placeholder={
                  inventory?.default_base_dir ||
                  settings?.default_base_dir ||
                  "Use default cache directory"
                }
              />
              <p className="text-xs text-muted-foreground">
                Resolved path:{" "}
                {inventory?.resolved_base_dir || settings?.resolved_base_dir || "Loading..."}
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <Button onClick={saveBaseDir} disabled={saving}>
                {saving ? "Saving..." : "Save and rescan"}
              </Button>
              <Button variant="outline" onClick={() => load(true)} disabled={loading}>
                <RefreshCw className="mr-2 h-4 w-4" />
                Refresh
              </Button>
              <Link to="/prepare">
                <Button variant="outline">Prepare new dataset</Button>
              </Link>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/70 bg-card/80">
          <CardHeader>
            <CardTitle className="text-base">Scan summary</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3 text-sm">
            <div className="flex items-center justify-between rounded-lg border border-border/70 bg-muted/40 px-4 py-3">
              <span className="text-muted-foreground">Prepared datasets</span>
              <span className="font-semibold">{inventory?.datasets.length ?? "..."}</span>
            </div>
            <div className="flex items-center justify-between rounded-lg border border-border/70 bg-muted/40 px-4 py-3">
              <span className="text-muted-foreground">Needs attention</span>
              <span className="font-semibold">
                {inventory?.invalid_snapshots.length ?? "..."}
              </span>
            </div>
            <div className="rounded-lg border border-border/70 bg-muted/40 px-4 py-3 text-xs text-muted-foreground">
              Last scan: {inventory ? formatDate(inventory.scanned_at) : "Loading..."}
            </div>
          </CardContent>
        </Card>
      </section>

      {error && <Alert className="border-destructive/40 text-destructive">{error}</Alert>}

      {loading && !inventory ? (
        <Card>
          <CardContent className="flex min-h-56 items-center justify-center text-sm text-muted-foreground">
            Scanning prepared datasets...
          </CardContent>
        </Card>
      ) : null}

      {inventory && inventory.datasets.length > 0 ? (
        <section className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
          {inventory.datasets.map((dataset) => (
            <Link key={dataset.id} to={`/datasets/${dataset.id}`} className="group block">
              <Card className="h-full border-border/70 bg-card/90 transition hover:-translate-y-0.5 hover:shadow-md">
                <CardHeader className="space-y-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <CardTitle className="text-lg">{dataset.dataset_class_name}</CardTitle>
                      <p className="mt-1 text-sm text-muted-foreground">
                        {dataset.config_name}
                      </p>
                    </div>
                    <div className="rounded-full border border-border/70 bg-muted/60 p-2">
                      <Database className="h-4 w-4" />
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Badge>{dataset.storage_type}</Badge>
                    {Object.entries(dataset.splits).map(([split, count]) => (
                      <Badge key={split} className="bg-muted text-foreground">
                        {split} {count}
                      </Badge>
                    ))}
                  </div>
                </CardHeader>
                <CardContent className="space-y-4 text-sm">
                  <p className="line-clamp-3 min-h-16 text-muted-foreground">
                    {dataset.metadata.description || "No description available."}
                  </p>
                  <dl className="grid gap-2 text-xs">
                    <div className="flex items-start justify-between gap-3">
                      <dt className="text-muted-foreground">Created</dt>
                      <dd className="text-right">{formatDate(dataset.created_at)}</dd>
                    </div>
                    <div className="flex items-start justify-between gap-3">
                      <dt className="text-muted-foreground">Relative path</dt>
                      <dd className="text-right font-mono">{dataset.relative_path}</dd>
                    </div>
                    <div className="flex items-start justify-between gap-3">
                      <dt className="text-muted-foreground">Config hash</dt>
                      <dd className="font-mono">{dataset.config_hash}</dd>
                    </div>
                  </dl>
                </CardContent>
              </Card>
            </Link>
          ))}
        </section>
      ) : null}

      {inventory && inventory.datasets.length === 0 && !loading ? (
        <Card className="border-dashed">
          <CardContent className="flex min-h-56 flex-col items-center justify-center gap-4 text-center">
            <div className="rounded-full bg-muted p-3">
              <FolderSearch className="h-5 w-5" />
            </div>
            <div className="space-y-1">
              <p className="font-medium">No prepared datasets found</p>
              <p className="max-w-lg text-sm text-muted-foreground">
                Point the output root at a directory containing cached dataset manifests, or
                use the prepare page to generate one in the background.
              </p>
            </div>
            <Link to="/prepare">
              <Button>Open prepare page</Button>
            </Link>
          </CardContent>
        </Card>
      ) : null}

      {inventory && inventory.invalid_snapshots.length > 0 ? (
        <Card className="border-amber-300/70 bg-amber-50/70">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base text-amber-950">
              <AlertTriangle className="h-4 w-4" />
              Snapshots needing attention
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-amber-950">
            {inventory.invalid_snapshots.map((item) => (
              <div key={item.path} className="rounded-lg border border-amber-200 bg-white/70 p-3">
                <p className="break-all font-mono text-xs">{item.path}</p>
                <p className="mt-1 text-xs text-amber-900/80">{item.reason}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      ) : null}
    </main>
  )
}
