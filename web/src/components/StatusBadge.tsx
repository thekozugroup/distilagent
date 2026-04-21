import { Badge } from "@/components/ui/badge";
import type { RunStatus } from "@/lib/api";

const VARIANT: Record<RunStatus, "default" | "secondary" | "destructive" | "outline" | "success" | "warning"> = {
  pending: "secondary",
  running: "warning",
  done: "success",
  failed: "destructive",
  cancelled: "outline",
};

export function StatusBadge({ status }: { status: RunStatus | string | undefined }) {
  const s = (status ?? "pending") as RunStatus;
  return <Badge variant={VARIANT[s] ?? "secondary"}>{s}</Badge>;
}
