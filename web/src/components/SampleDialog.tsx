import { useQuery } from "@tanstack/react-query";
import { api, type SampleDetail } from "@/lib/api";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useState } from "react";

export function SampleDialog({
  runId,
  sampleId,
  open,
  onOpenChange,
}: {
  runId: string;
  sampleId: string | null;
  open: boolean;
  onOpenChange: (v: boolean) => void;
}) {
  const { data, isLoading } = useQuery({
    queryKey: ["sample", runId, sampleId],
    queryFn: () => api.getSample(runId, sampleId as string),
    enabled: !!sampleId && open,
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl">
        <DialogHeader>
          <DialogTitle className="truncate pr-8">{sampleId ?? "Sample"}</DialogTitle>
        </DialogHeader>
        <ScrollArea className="max-h-[75vh] pr-3">
          {isLoading || !data ? (
            <div className="space-y-3">
              <Skeleton className="h-6 w-3/4" />
              <Skeleton className="h-32 w-full" />
              <Skeleton className="h-24 w-full" />
            </div>
          ) : (
            <SampleBody sample={data} />
          )}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}

function SampleBody({ sample }: { sample: SampleDetail }) {
  return (
    <div className="space-y-5 text-sm">
      <div className="flex flex-wrap gap-2">
        {sample.category && <Badge variant="outline">{sample.category}</Badge>}
        {sample.model_name && <Badge variant="secondary">{sample.model_name}</Badge>}
        {typeof sample.autoreason_iterations === "number" && (
          <Badge variant="outline">iters: {sample.autoreason_iterations}</Badge>
        )}
        {sample.autoreason_converged != null && (
          <Badge variant={sample.autoreason_converged ? "success" : "warning"}>
            {sample.autoreason_converged ? "converged" : "not converged"}
          </Badge>
        )}
        {typeof sample.total_calls === "number" && (
          <Badge variant="outline">calls: {sample.total_calls}</Badge>
        )}
      </div>

      <Section title="Instruction">
        <pre className="whitespace-pre-wrap break-words rounded-md bg-muted p-3 font-mono text-xs leading-relaxed">
          {sample.instruction ?? "(empty)"}
        </pre>
      </Section>

      <Section title="Generation">
        <pre className="whitespace-pre-wrap break-words rounded-md bg-muted p-3 font-mono text-xs leading-relaxed">
          {sample.generation ?? "(empty)"}
        </pre>
      </Section>

      {Array.isArray(sample.autoreason_trace) && sample.autoreason_trace.length > 0 && (
        <Section title={`AutoReason Trace (${sample.autoreason_trace.length})`}>
          <div className="space-y-2">
            {sample.autoreason_trace.map((step, i) => (
              <AutoReasonStep key={i} step={step as Record<string, unknown>} idx={i} />
            ))}
          </div>
        </Section>
      )}

      {Array.isArray(sample.reasoning_trace) && sample.reasoning_trace.length > 0 && (
        <Section title={`Reasoning Trace (${sample.reasoning_trace.length})`}>
          <div className="space-y-2">
            {sample.reasoning_trace.map((entry, i) => (
              <ReasoningEntry key={i} entry={entry as Record<string, unknown>} idx={i} />
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">{title}</div>
      {children}
      <Separator className="mt-4" />
    </div>
  );
}

function AutoReasonStep({ step, idx }: { step: Record<string, unknown>; idx: number }) {
  const [open, setOpen] = useState(idx === 0);
  const winner = step.winner as string | undefined;
  const borda = step.borda as Record<string, number> | undefined;
  const iteration = (step.iteration as number | undefined) ?? idx;
  return (
    <div className="rounded-md border">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-3 py-2 text-left text-xs hover:bg-muted/50"
      >
        <div className="flex items-center gap-2">
          <Badge variant="outline">iter {iteration}</Badge>
          {winner && <Badge variant="secondary">winner: {winner}</Badge>}
          {borda && (
            <span className="text-muted-foreground">
              borda A={borda.A ?? 0} B={borda.B ?? 0} AB={borda.AB ?? 0}
            </span>
          )}
        </div>
        <span className="text-muted-foreground">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words border-t p-3 font-mono text-[11px] leading-relaxed">
          {JSON.stringify(step, null, 2)}
        </pre>
      )}
    </div>
  );
}

function ReasoningEntry({ entry, idx }: { entry: Record<string, unknown>; idx: number }) {
  const [open, setOpen] = useState(false);
  const role = entry.role_hint as string | undefined;
  const source = entry.source_model as string | undefined;
  const text = (entry.text as string | undefined) ?? (entry.content as string | undefined) ?? "";
  const reasoning = entry.reasoning as string | undefined;
  return (
    <div className="rounded-md border">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-3 py-2 text-left text-xs hover:bg-muted/50"
      >
        <div className="flex items-center gap-2">
          <Badge variant="outline">#{idx}</Badge>
          {role && <Badge variant="secondary">{role}</Badge>}
          {source && <span className="text-muted-foreground">{source}</span>}
        </div>
        <span className="text-muted-foreground">
          {text.length} chars {open ? "−" : "+"}
        </span>
      </button>
      {open && (
        <div className="max-h-80 overflow-auto border-t">
          {reasoning && (
            <div className="border-b p-3">
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">reasoning</div>
              <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed">{reasoning}</pre>
            </div>
          )}
          <div className="p-3">
            <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">text</div>
            <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed">{text || "(empty)"}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
