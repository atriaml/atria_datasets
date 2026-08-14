import { useEffect, useState } from "react"
import { Clock3, FolderCog, RefreshCw } from "lucide-react"
import { Alert } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { api } from "@/shared/api"
import type { DatasetOption, ExplorerSettings, PreparationJob } from "@/types"

function formatDate(value: string | null) {
  if (!value) return "—"
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

export function PreparePage() {
  const [options, setOptions] = useState<DatasetOption[]>([])
  const [settings, setSettings] = useState<ExplorerSettings | null>(null)
  const [jobs, setJobs] = useState<PreparationJob[]>([])
  const [name, setName] = useState("")
  const [sourceDir, setSourceDir] = useState("")
  const [outputRoot, setOutputRoot] = useState("")
  const [configText, setConfigText] = useState("{}")
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState("")

  const load = async () => {
    try {
      const [nextOptions, nextSettings, nextJobs] = await Promise.all([
        api.preparationOptions(),
        api.settings(),
        api.jobs(),
      ])
      setOptions(nextOptions.datasets)
      setSettings(nextSettings)
      setJobs(nextJobs.jobs)
      if (!name && nextOptions.datasets.length > 0) setName(nextOptions.datasets[0].name)
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : String(nextError))
    }
  }

  useEffect(() => {
    void (async () => {
      try {
        const [nextOptions, nextSettings, nextJobs] = await Promise.all([
          api.preparationOptions(),
          api.settings(),
          api.jobs(),
        ])
        setOptions(nextOptions.datasets)
        setSettings(nextSettings)
        setJobs(nextJobs.jobs)
        if (nextOptions.datasets.length > 0) {
          setName((current) => current || nextOptions.datasets[0].name)
        }
      } catch (nextError) {
        setError(nextError instanceof Error ? nextError.message : String(nextError))
      }
    })()
  }, [])

  useEffect(() => {
    const timer = window.setInterval(() => {
      api.jobs()
        .then((result) => setJobs(result.jobs))
        .catch(() => undefined)
    }, 3000)
    return () => window.clearInterval(timer)
  }, [])

  const submit = async () => {
    setError("")
    setSubmitting(true)
    try {
      const config = JSON.parse(configText) as Record<string, unknown>
      const normalizedSourceDir = normalizeOptionalPath(sourceDir)
      const normalizedOutputRoot = normalizeOptionalPath(outputRoot)
      await api.createJob({
        name,
        source_dir: normalizedSourceDir || null,
        output_root: normalizedOutputRoot || null,
        config,
      })
      setSourceDir("")
      setOutputRoot("")
      setConfigText("{}")
      const refreshed = await api.jobs()
      setJobs(refreshed.jobs)
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : String(nextError))
    } finally {
      setSubmitting(false)
    }
  }

  const selected = options.find((item) => item.name === name)

  return (
    <main className="mx-auto max-w-screen-2xl space-y-7 px-5 py-8 lg:px-8">
      <section className="grid gap-4 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        <Card className="border-border/70 bg-card/90">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl">
              <FolderCog className="h-5 w-5" />
              Queue a preparation job
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <p className="text-sm text-muted-foreground">
              Preparation is separate from browsing. Jobs run in the background and the
              inventory page will pick them up after the cache manifest appears.
            </p>
            <div className="grid gap-2">
              <label className="text-sm font-medium">Dataset</label>
              <select
                className="h-10 rounded-md border bg-background px-3 text-sm"
                value={name}
                onChange={(event) => setName(event.target.value)}
              >
                {options.map((item) => (
                  <option key={item.name}>{item.name}</option>
                ))}
              </select>
            </div>
            <div className="grid gap-2">
              <label className="text-sm font-medium">
                Source directory <span className="font-normal text-muted-foreground">(optional)</span>
              </label>
              <Input
                value={sourceDir}
                onChange={(event) => setSourceDir(event.target.value)}
                placeholder="Only needed when the dataset expects local source files"
              />
            </div>
            <div className="grid gap-2">
              <label className="text-sm font-medium">
                Output root <span className="font-normal text-muted-foreground">(optional)</span>
              </label>
              <Input
                value={outputRoot}
                onChange={(event) => setOutputRoot(event.target.value)}
                placeholder={settings?.resolved_base_dir || "Uses the current inventory root"}
              />
              <p className="text-xs text-muted-foreground">
                Current inventory root: {settings?.resolved_base_dir || "Loading..."}
              </p>
            </div>
            <div className="grid gap-2">
              <label className="text-sm font-medium">
                Config overrides <span className="font-normal text-muted-foreground">(JSON)</span>
              </label>
              <textarea
                className="min-h-28 rounded-md border bg-background p-3 font-mono text-sm outline-none focus:ring-2"
                value={configText}
                onChange={(event) => setConfigText(event.target.value)}
              />
              {selected && selected.config_fields.length > 0 ? (
                <p className="text-xs text-muted-foreground">
                  Fields:{" "}
                  {selected.config_fields
                    .map((field) => `${field.name}=${JSON.stringify(field.default)}`)
                    .join(", ")}
                </p>
              ) : null}
            </div>
            {error && <Alert className="border-destructive/40 text-destructive">{error}</Alert>}
            <Button disabled={!name || submitting} onClick={submit}>
              {submitting ? "Queueing..." : "Queue preparation job"}
            </Button>
          </CardContent>
        </Card>

        <Card className="border-border/70 bg-card/90">
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="text-base">Job activity</CardTitle>
            <Button variant="outline" size="sm" onClick={load}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
          </CardHeader>
          <CardContent className="space-y-3">
            {jobs.length === 0 ? (
              <div className="flex min-h-40 items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">
                No preparation jobs yet.
              </div>
            ) : (
              jobs.map((job) => (
                <div key={job.id} className="rounded-lg border border-border/70 bg-muted/30 p-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="font-medium">{job.dataset_name}</p>
                      <p className="text-xs text-muted-foreground">{job.id}</p>
                    </div>
                    <Badge>{job.status}</Badge>
                  </div>
                  <dl className="mt-3 grid gap-2 text-xs">
                    <div className="flex items-start justify-between gap-3">
                      <dt className="text-muted-foreground">Created</dt>
                      <dd>{formatDate(job.created_at)}</dd>
                    </div>
                    <div className="flex items-start justify-between gap-3">
                      <dt className="text-muted-foreground">Output root</dt>
                      <dd className="text-right font-mono">
                        {normalizeOptionalPath(job.output_root) || settings?.resolved_base_dir || "Default"}
                      </dd>
                    </div>
                    {job.result_path ? (
                      <div className="flex items-start justify-between gap-3">
                        <dt className="text-muted-foreground">Result path</dt>
                        <dd className="text-right font-mono">{job.result_path}</dd>
                      </div>
                    ) : null}
                  </dl>
                  {job.error ? (
                    <Alert className="mt-3 border-destructive/40 text-destructive">
                      {job.error}
                    </Alert>
                  ) : null}
                </div>
              ))
            )}
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Clock3 className="h-3.5 w-3.5" />
              Running and queued jobs refresh automatically every 3 seconds.
            </div>
          </CardContent>
        </Card>
      </section>
    </main>
  )
}
