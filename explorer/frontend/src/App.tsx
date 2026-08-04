import { BrowserRouter, Route, Routes } from "react-router-dom"
import { AppHeader } from "@/components/app-header"
import { DocumentPage } from "@/pages/document-page"
import { DocumentsPage } from "@/pages/documents-page"
import { HomePage } from "@/pages/home-page"
import { PreparePage } from "@/pages/prepare-page"

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(18,80,90,0.10),_transparent_32%),linear-gradient(180deg,_rgba(255,255,255,0.96),_rgba(244,247,245,0.96))]">
        <AppHeader />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/prepare" element={<PreparePage />} />
          <Route path="/datasets/:datasetId" element={<DocumentsPage />} />
          <Route
            path="/datasets/:datasetId/documents/:split/:index"
            element={<DocumentPage />}
          />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
