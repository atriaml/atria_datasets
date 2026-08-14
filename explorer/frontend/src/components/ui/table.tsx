import * as React from "react"
import { cn } from "@/shared/utils"

export const Table = ({ className, ...props }: React.HTMLAttributes<HTMLTableElement>) => <div className="relative w-full overflow-auto"><table className={cn("w-full caption-bottom text-sm", className)} {...props} /></div>
export const TableHeader = (props: React.HTMLAttributes<HTMLTableSectionElement>) => <thead className="border-b" {...props} />
export const TableBody = (props: React.HTMLAttributes<HTMLTableSectionElement>) => <tbody {...props} />
export const TableRow = (props: React.HTMLAttributes<HTMLTableRowElement>) => <tr className="border-b transition-colors hover:bg-muted/50" {...props} />
export const TableHead = (props: React.ThHTMLAttributes<HTMLTableCellElement>) => <th className="h-11 px-4 text-left align-middle font-medium text-muted-foreground" {...props} />
export const TableCell = (props: React.TdHTMLAttributes<HTMLTableCellElement>) => <td className="p-4 align-top" {...props} />
