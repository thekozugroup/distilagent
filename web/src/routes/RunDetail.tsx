import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api, type SampleSummary } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { StatusBadge } from "@/components/StatusBadge";
import { StatCard } from "@/components/StatCard";
import { SampleDialog } from "@/components/SampleDialog";
import { useSse, type SseEvent } from "@/lib/sse";
import { elapsed, formatDuration, formatTime, shortId } from "@/lib/utils";
import { ArrowLeft, Download, Pause, Play, StopCircle } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

export default function RunDetail() {
  const { id = "" } = useParams();
  const qc = useQueryClient();
  const { data: run } = useQuery({
    queryKey: ["run", id],
    queryFn: () => api.getRun(id),
    refetchInterval: (q) => {
      const status = (q.state.data as { status?: string } | undefined)?.status;
      return status && ["done", "failed", "cancelled"].includes(status) ? false : 3000;
    },
  });

  const cancel = useMutation({
    mutationFn: () => api.cancelRun(id),
    onSuccess: () => {
      toast.success("Cancellation requested");
      qc.invalidateQueries({ queryKey: ["run", id] });
    },
    onError: (err: Error) => toast.error(err.message),
  });

  if (!run) {
    return (
      <div className="mx-auto max-w-6xl space-y-4 px-4 py-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-24 w-full" />
      </div>
    );
  }

  const elapsedSec = elapsed(run);
  const completed = run.completed ?? 0;
  const total = run.total_planned ?? 0;
  const failed = run.failed ?? 0;
  const avgPerSample = completed > 0 && elapsedSec ? elapsedSec / completed : null;
  const isRunning = run.status === "running" || run.status === "pending";

  return (
    <div className="mx-auto max-w-6xl space-y-4 px-4 py-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Button asChild variant="ghost" size="sm">
            <Link to="/">
              <ArrowLeft className="mr-1 h-4 w-4" />
              back
            </Link>
          </Button>
          <StatusBadge status={run.status} />
          <span className="font-mono text-sm">{shortId(run.id, 16)}</span>
          <span className="text-xs text-muted-foreground">{formatTime(run.started_at ?? run.created_at)}</span>
        </div>
        <div className="flex gap-2">
          {isRunning && (
            <Button
              variant="destructive"
              size="sm"
              onClick={() => cancel.mutate()}
              disabled={cancel.isPending}
            >
              <StopCircle className="mr-1 h-4 w-4" />
              Cancel
            </Button>
          )}
          {run.output_path && (
            <Button asChild variant="outline" size="sm">
              <a href={api.downloadUrl(run.id)} download>
                <Download className="mr-1 h-4 w-4" />
                JSONL
              </a>
            </Button>
          )}
        </div>
      </div>

      {run.error && (
        <Card>
          <CardContent className="p-4 text-sm text-destructive">{run.error}</CardContent>
        </Card>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard label="Progress" value={`${completed}/${total || "—"}`} />
        <StatCard label="Failed" value={failed} />
        <StatCard label="Elapsed" value={formatDuration(elapsedSec)} />
        <StatCard label="Avg s/sample" value={avgPerSample ? avgPerSample.toFixed(1) : "—"} />
      </div>

      <Tabs defaultValue="events">
        <TabsList>
          <TabsTrigger value="events">Events</TabsTrigger>
          <TabsTrigger value="samples">Samples</TabsTrigger>
          <TabsTrigger value="config">Config</TabsTrigger>
        </TabsList>

        <TabsContent value="events">
          <EventsTab runId={id} live={isRunning} />
        </TabsContent>
        <TabsContent value="samples">
          <SamplesTab runId={id} />
        </TabsContent>
        <TabsContent value="config">
          <Card>
            <CardContent className="p-4">
              <pre className="overflow-auto rounded-md bg-muted p-3 font-mono text-xs leading-relaxed">
                {JSON.stringify(run.config ?? {}, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function EventsTab({ runId, live }: { runId: string; live: boolean }) {
  const [paused, setPaused] = useState(false);
  const { events, connected } = useSse(live ? api.sseUrl(runId) : null, { enabled: live, paused });
  const scrollRef = useRef<HTMLDivElement>(null);
  const atBottomRef = useRef(true);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (atBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [events]);

  const onScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  };

  return (
    <Card>
      <CardContent className="p-3">
        <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
          <span>
            {live ? (connected ? "• live" : "• connecting…") : "stream closed"} — {events.length} events
          </span>
          {live && (
            <Button variant="ghost" size="sm" onClick={() => setPaused((v) => !v)}>
              {paused ? <Play className="mr-1 h-3.5 w-3.5" /> : <Pause className="mr-1 h-3.5 w-3.5" />}
              {paused ? "Resume" : "Pause"}
            </Button>
          )}
        </div>
        <div
          ref={scrollRef}
          onScroll={onScroll}
          className="h-[480px] overflow-auto rounded-md bg-muted/30 p-3 font-mono text-[11px] leading-relaxed"
        >
          {events.length === 0 && <div className="text-muted-foreground">No events yet.</div>}
          {events.map((e) => (
            <EventLine key={e.id} e={e} />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function EventLine({ e }: { e: SseEvent }) {
  const color =
    e.event === "error"
      ? "text-red-400"
      : e.event === "sample"
      ? (e.data as { status?: string })?.status === "fail"
        ? "text-amber-400"
        : "text-emerald-400"
      : e.event === "terminal" || e.event === "done"
      ? "text-blue-400"
      : "text-foreground/80";
  const summary = summarize(e);
  return (
    <div className="whitespace-pre-wrap break-all">
      <span className="text-muted-foreground">[{new Date(e.ts).toLocaleTimeString()}] </span>
      <span className={color}>{e.event.padEnd(8)}</span> {summary}
    </div>
  );
}

function summarize(e: SseEvent): string {
  const d = e.data as Record<string, unknown>;
  if (!d || typeof d !== "object") return String(e.data);
  if (e.event === "plan") return `total=${d.total} done=${d.done} remaining=${d.remaining}`;
  if (e.event === "sample") {
    const bits = [
      `${d.idx}/${d.total}`,
      d.status,
      d.teacher && `teacher=${d.teacher}`,
      d.category && `cat=${d.category}`,
      d.iterations != null && `iters=${d.iterations}`,
      d.converged != null && `conv=${d.converged}`,
      d.elapsed_seconds != null && `${Number(d.elapsed_seconds).toFixed(1)}s`,
      d.error && `err=${d.error}`,
    ].filter(Boolean);
    return bits.join(" ");
  }
  if (e.event === "info" || e.event === "error") return (d.msg as string) ?? JSON.stringify(d);
  if (e.event === "done") return `exit=${d.exit_code} status=${d.status}`;
  if (e.event === "terminal") return `status=${d.status}`;
  return JSON.stringify(d);
}

function SamplesTab({ runId }: { runId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["samples", runId],
    queryFn: () => api.listSamples(runId, 200),
    refetchInterval: 5000,
  });
  const [openSample, setOpenSample] = useState<string | null>(null);

  const rows = useMemo(() => data ?? [], [data]);

  return (
    <Card>
      <CardContent className="p-0">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>ID</TableHead>
              <TableHead>Category</TableHead>
              <TableHead>Instruction</TableHead>
              <TableHead className="text-right">Iters</TableHead>
              <TableHead className="text-right">Conv</TableHead>
              <TableHead className="text-right">Calls</TableHead>
              <TableHead className="text-right">Sec</TableHead>
              <TableHead className="text-right">Gen chars</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading && (
              <TableRow>
                <TableCell colSpan={8} className="py-8 text-center text-muted-foreground">
                  Loading…
                </TableCell>
              </TableRow>
            )}
            {!isLoading && rows.length === 0 && (
              <TableRow>
                <TableCell colSpan={8} className="py-8 text-center text-muted-foreground">
                  No samples yet.
                </TableCell>
              </TableRow>
            )}
            {rows.map((s) => (
              <SampleRow key={s.id} sample={s} onClick={() => setOpenSample(s.id)} />
            ))}
          </TableBody>
        </Table>
      </CardContent>
      <SampleDialog
        runId={runId}
        sampleId={openSample}
        open={!!openSample}
        onOpenChange={(o) => !o && setOpenSample(null)}
      />
    </Card>
  );
}

function SampleRow({ sample, onClick }: { sample: SampleSummary; onClick: () => void }) {
  return (
    <TableRow onClick={onClick} className="cursor-pointer">
      <TableCell className="max-w-[160px] truncate font-mono text-xs">{sample.id}</TableCell>
      <TableCell className="text-xs text-muted-foreground">{sample.category ?? "—"}</TableCell>
      <TableCell className="max-w-[380px] truncate text-xs">{sample.instruction}</TableCell>
      <TableCell className="text-right tabular-nums">{sample.autoreason_iterations ?? "—"}</TableCell>
      <TableCell className="text-right tabular-nums">
        {sample.autoreason_converged == null ? "—" : sample.autoreason_converged ? "✓" : "✗"}
      </TableCell>
      <TableCell className="text-right tabular-nums">{sample.total_calls ?? "—"}</TableCell>
      <TableCell className="text-right tabular-nums">
        {sample.elapsed_seconds != null ? sample.elapsed_seconds.toFixed(1) : "—"}
      </TableCell>
      <TableCell className="text-right tabular-nums">{sample.generation_chars}</TableCell>
    </TableRow>
  );
}
