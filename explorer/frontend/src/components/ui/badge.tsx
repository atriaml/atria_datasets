import type { HTMLAttributes } from "react"
import { cn } from "@/lib/utils"

export function Badge({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("inline-flex items-center rounded-md bg-muted px-2 py-1 text-xs font-medium text-muted-foreground", className)} {...props} />
}
