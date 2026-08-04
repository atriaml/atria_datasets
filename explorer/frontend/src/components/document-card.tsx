import { AlertTriangle, FileText } from "lucide-react"
import { Link } from "react-router-dom"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import type { DocumentSummary } from "@/types"

export function DocumentCard({
  datasetId,
  document,
}: {
  datasetId: string
  document: DocumentSummary
}) {
  return (
    <Link
      to={`/datasets/${datasetId}/documents/${document.split}/${document.index}`}
      className="group block"
    >
      <Card className="h-full overflow-hidden border-border/70 transition hover:-translate-y-0.5 hover:border-foreground/20 hover:shadow-md">
        <div className="flex aspect-[4/3] items-center justify-center overflow-hidden bg-muted/60">
          <img
            src={document.image_url}
            alt=""
            loading="lazy"
            className="h-full w-full object-contain transition group-hover:scale-[1.01]"
          />
        </div>
        <CardContent className="space-y-3 p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <p className="truncate text-sm font-medium">{document.sample_id}</p>
              <p className="mt-1 flex items-center gap-1 text-xs text-muted-foreground">
                <FileText className="h-3 w-3" />
                Document {document.index + 1}
              </p>
            </div>
            {document.warning_count > 0 && (
              <AlertTriangle className="h-4 w-4 shrink-0 text-amber-600" />
            )}
          </div>
          <p className="line-clamp-2 min-h-10 text-xs leading-5 text-muted-foreground">
            {document.transcription_preview || "No transcription"}
          </p>
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(document.annotation_counts).map(([name, count]) => (
              <Badge key={name}>
                {count} {name}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>
    </Link>
  )
}
