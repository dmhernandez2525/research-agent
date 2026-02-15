import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";

interface ReportViewerProps {
  markdown: string;
  sessionId: string;
}

interface HeadingEntry {
  id: string;
  depth: number;
  text: string;
}

const HEADING_RE = /^(#{2,4})\s+(.+)$/gm;

function slugify(value: string): string {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-");
}

function extractHeadings(markdown: string): HeadingEntry[] {
  const headings: HeadingEntry[] = [];
  let match = HEADING_RE.exec(markdown);
  while (match) {
    headings.push({
      id: slugify(match[2]),
      depth: match[1].length,
      text: match[2],
    });
    match = HEADING_RE.exec(markdown);
  }
  return headings;
}

function citationCount(markdown: string): number {
  const matches = markdown.match(/\[[0-9]+\]/g);
  return matches?.length ?? 0;
}

export function ReportViewer({ markdown, sessionId }: ReportViewerProps): JSX.Element {
  const headings = useMemo(() => extractHeadings(markdown), [markdown]);
  const [copyStatus, setCopyStatus] = useState<string>("");

  const handleCopy = async () => {
    await navigator.clipboard.writeText(markdown);
    setCopyStatus("Copied full report");
    window.setTimeout(() => setCopyStatus(""), 1600);
  };

  return (
    <section className="card p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-lg font-semibold">Report Viewer</h3>
        <div className="no-print flex items-center gap-2">
          <button
            type="button"
            onClick={handleCopy}
            className="rounded-full border border-slate-300 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-ink/70"
          >
            Copy
          </button>
          <a
            href={`/api/sessions/${sessionId}/report?format=pdf`}
            className="rounded-full bg-calm px-3 py-1 text-xs font-semibold uppercase tracking-wide text-white"
          >
            PDF
          </a>
          <button type="button" onClick={() => window.print()} className="rounded-full border border-slate-300 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-ink/70">
            Print
          </button>
        </div>
      </div>

      <div className="mb-3 text-xs text-ink/60">Citations detected: {citationCount(markdown)}</div>
      {copyStatus ? <p className="mb-3 text-xs text-emerald-700">{copyStatus}</p> : null}

      <div className="grid gap-4 lg:grid-cols-[220px_minmax(0,1fr)]">
        <aside className="rounded-xl border border-slate-200 bg-white p-3">
          <h4 className="mb-2 text-xs uppercase tracking-wide text-slate-500">Table of Contents</h4>
          {headings.length === 0 ? (
            <p className="text-sm text-ink/60">No headings</p>
          ) : (
            <ul className="space-y-1 text-sm">
              {headings.map((heading) => (
                <li key={heading.id} style={{ marginLeft: `${(heading.depth - 2) * 10}px` }}>
                  <a href={`#${heading.id}`} className="text-calm hover:underline">
                    {heading.text}
                  </a>
                </li>
              ))}
            </ul>
          )}
        </aside>

        <article className="prose prose-slate max-w-none rounded-xl border border-slate-200 bg-white p-4">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeRaw, rehypeHighlight]}
            components={{
              h2(props) {
                const text = String(props.children);
                return <h2 id={slugify(text)} {...props} />;
              },
              h3(props) {
                const text = String(props.children);
                return <h3 id={slugify(text)} {...props} />;
              },
              a(props) {
                const href = typeof props.href === "string" ? props.href : "";
                if (href.startsWith("#source-")) {
                  return <a {...props} className="font-semibold text-accent underline decoration-dotted" />;
                }
                return <a {...props} className="text-calm underline" target="_blank" rel="noreferrer" />;
              },
            }}
          >
            {markdown}
          </ReactMarkdown>
        </article>
      </div>
    </section>
  );
}
