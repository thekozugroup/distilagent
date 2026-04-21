import { Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api, type RunSummary } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { StatusBadge } from "@/components/StatusBadge";
import { elapsed, formatDuration, formatTime, shortId } from "@/lib/utils";
import { Download, Plus, Eye } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export default function RunsList() {
  const navigate = useNavigate();
  const { data, isLoading, error } = useQuery({
    queryKey: ["runs"],
    queryFn: api.listRuns,
    refetchInterval: 5000,
  });

  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold">Runs</h1>
        <Button onClick={() => navigate("/runs/new")}>
          <Plus className="mr-2 h-4 w-4" />
          New Run
        </Button>
      </div>

      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[110px]">Status</TableHead>
                <TableHead>ID</TableHead>
                <TableHead>Started</TableHead>
                <TableHead>Progress</TableHead>
                <TableHead>Failed</TableHead>
                <TableHead>Elapsed</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading && (
                <TableRow>
                  <TableCell colSpan={7} className="py-8 text-center text-muted-foreground">
                    Loading…
                  </TableCell>
                </TableRow>
              )}
              {error && (
                <TableRow>
                  <TableCell colSpan={7} className="py-8 text-center text-destructive">
                    {(error as Error).message}
                  </TableCell>
                </TableRow>
              )}
              {data && data.length === 0 && (
                <TableRow>
                  <TableCell colSpan={7} className="py-8 text-center text-muted-foreground">
                    No runs yet. Click <em>New Run</em> to start one.
                  </TableCell>
                </TableRow>
              )}
              {data?.map((r) => <RunRow key={r.id} run={r} />)}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}

function RunRow({ run }: { run: RunSummary }) {
  const elapsedSec = elapsed(run);
  return (
    <TableRow>
      <TableCell>
        <StatusBadge status={run.status} />
      </TableCell>
      <TableCell className="font-mono text-xs">
        <Link to={`/runs/${run.id}`} className="hover:underline">
          {shortId(run.id, 12)}
        </Link>
      </TableCell>
      <TableCell className="text-xs text-muted-foreground">{formatTime(run.started_at ?? run.created_at)}</TableCell>
      <TableCell className="tabular-nums">
        {(run.completed ?? 0)} / {run.total_planned ?? "—"}
      </TableCell>
      <TableCell className="tabular-nums">{run.failed ?? 0}</TableCell>
      <TableCell className="tabular-nums text-muted-foreground">{formatDuration(elapsedSec)}</TableCell>
      <TableCell className="text-right">
        <div className="flex justify-end gap-1">
          <Button asChild variant="ghost" size="sm">
            <Link to={`/runs/${run.id}`}>
              <Eye className="mr-1 h-3.5 w-3.5" />
              view
            </Link>
          </Button>
          {run.output_path && (
            <Button asChild variant="ghost" size="sm">
              <a href={api.downloadUrl(run.id)} download>
                <Download className="mr-1 h-3.5 w-3.5" />
                jsonl
              </a>
            </Button>
          )}
        </div>
      </TableCell>
    </TableRow>
  );
}
