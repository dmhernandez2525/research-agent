import { useMemo, useState } from "react";

import { useLocalStorage } from "../../hooks/useLocalStorage";

interface PlanItem {
  id: string;
  title: string;
  queryPreview: string;
}

interface PlanEditorProps {
  sessionId: string;
  initialQuery: string;
}

const STORAGE_PREFIX = "ra_plan_draft_";

function createInitialPlan(query: string): PlanItem[] {
  if (!query.trim()) {
    return [];
  }
  return [
    { id: "subtopic-1", title: "Core problem framing", queryPreview: `${query} fundamentals` },
    { id: "subtopic-2", title: "Implementation approaches", queryPreview: `${query} implementation examples` },
    { id: "subtopic-3", title: "Risks and tradeoffs", queryPreview: `${query} limitations and risks` },
  ];
}

function reorder(items: PlanItem[], fromIndex: number, toIndex: number): PlanItem[] {
  const next = [...items];
  const [moved] = next.splice(fromIndex, 1);
  next.splice(toIndex, 0, moved);
  return next;
}

export function PlanEditor({ sessionId, initialQuery }: PlanEditorProps): JSX.Element {
  const storageKey = `${STORAGE_PREFIX}${sessionId}`;
  const [items, setItems] = useLocalStorage<PlanItem[]>(storageKey, createInitialPlan(initialQuery));
  const [workflowState, setWorkflowState] = useState<"draft" | "approved" | "rejected">("draft");
  const [dragIndex, setDragIndex] = useState<number | null>(null);

  const canApprove = useMemo(() => items.length > 0, [items.length]);

  const addItem = () => {
    const nextId = `subtopic-${Date.now()}`;
    setItems([
      ...items,
      {
        id: nextId,
        title: "New subtopic",
        queryPreview: `${initialQuery} follow-up research`,
      },
    ]);
    setWorkflowState("draft");
  };

  const updateItem = (id: string, field: "title" | "queryPreview", value: string) => {
    setItems(items.map((item) => (item.id === id ? { ...item, [field]: value } : item)));
    setWorkflowState("draft");
  };

  const removeItem = (id: string) => {
    setItems(items.filter((item) => item.id !== id));
    setWorkflowState("draft");
  };

  const onDrop = (targetIndex: number) => {
    if (dragIndex === null || dragIndex === targetIndex) {
      return;
    }
    setItems(reorder(items, dragIndex, targetIndex));
    setDragIndex(null);
    setWorkflowState("draft");
  };

  return (
    <section className="card p-5">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-lg font-semibold">Interactive Plan Editor</h3>
        <p className="text-xs text-ink/60">Draft saved to localStorage</p>
      </div>

      <div className="mb-4 flex flex-wrap gap-2">
        <button type="button" onClick={addItem} className="rounded-full bg-calm px-3 py-1 text-xs font-semibold uppercase tracking-wide text-white">
          Add
        </button>
        <button
          type="button"
          onClick={() => setWorkflowState("approved")}
          disabled={!canApprove}
          className="rounded-full border border-emerald-300 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-emerald-700 disabled:opacity-40"
        >
          Approve
        </button>
        <button
          type="button"
          onClick={() => setWorkflowState("rejected")}
          className="rounded-full border border-rose-300 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-rose-700"
        >
          Reject
        </button>
        <span className={`status-pill ${workflowState === "approved" ? "bg-emerald-100 text-emerald-700" : workflowState === "rejected" ? "bg-rose-100 text-rose-700" : "bg-slate-100 text-slate-700"}`}>
          {workflowState}
        </span>
      </div>

      {items.length === 0 ? (
        <p className="text-sm text-ink/70">No subtopics defined.</p>
      ) : (
        <ol className="space-y-2">
          {items.map((item, index) => (
            <li
              key={item.id}
              draggable
              onDragStart={() => setDragIndex(index)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={() => onDrop(index)}
              className="rounded-xl border border-slate-200 bg-white p-3"
            >
              <div className="mb-2 flex items-center justify-between gap-2">
                <span className="cursor-grab text-xs uppercase tracking-wide text-ink/50">Drag</span>
                <button type="button" onClick={() => removeItem(item.id)} className="text-xs text-rose-600">
                  Remove
                </button>
              </div>
              <input
                value={item.title}
                onChange={(event) => updateItem(item.id, "title", event.target.value)}
                className="mb-2 w-full rounded-lg border border-slate-300 px-2 py-1 text-sm"
              />
              <input
                value={item.queryPreview}
                onChange={(event) => updateItem(item.id, "queryPreview", event.target.value)}
                className="w-full rounded-lg border border-slate-300 px-2 py-1 text-xs text-ink/70"
              />
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
