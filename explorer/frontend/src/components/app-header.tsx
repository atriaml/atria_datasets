import { Database, FolderCog, LayoutGrid } from "lucide-react"
import { Link, NavLink } from "react-router-dom"

const navClass = ({ isActive }: { isActive: boolean }) =>
  `rounded-full px-3 py-2 text-sm transition ${
    isActive ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
  }`

export function AppHeader() {
  return (
    <header className="sticky top-0 z-20 border-b border-border/70 bg-background/90 backdrop-blur">
      <div className="mx-auto flex min-h-16 max-w-screen-2xl flex-col gap-3 px-5 py-4 lg:flex-row lg:items-center lg:justify-between lg:px-8">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-sm">
            <Database className="h-5 w-5" />
          </div>
          <div>
            <Link to="/" className="text-base font-semibold tracking-tight">
              Dataset Manifest
            </Link>
            <p className="text-xs text-muted-foreground">
              Scan prepared datasets, inspect documents, and queue new prep jobs.
            </p>
          </div>
        </div>
        <nav className="flex items-center gap-2">
          <NavLink to="/" end className={navClass}>
            <span className="inline-flex items-center gap-2">
              <LayoutGrid className="h-4 w-4" />
              Inventory
            </span>
          </NavLink>
          <NavLink to="/prepare" className={navClass}>
            <span className="inline-flex items-center gap-2">
              <FolderCog className="h-4 w-4" />
              Prepare
            </span>
          </NavLink>
        </nav>
      </div>
    </header>
  )
}
