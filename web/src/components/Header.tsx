import { Link, useLocation } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Sparkles } from "lucide-react";

export function Header() {
  const { data, isError } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 10000,
    retry: false,
  });
  const healthy = !isError && data?.status === "ok";
  const loc = useLocation();

  return (
    <header className="border-b border-border">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
        <Link to="/" className="flex items-center gap-2 font-semibold">
          <Sparkles className="h-5 w-5 text-primary" />
          <span>DistilAgent</span>
          <span className="ml-2 text-xs font-normal text-muted-foreground">AutoReason dashboard</span>
        </Link>
        <div className="flex items-center gap-4 text-sm">
          <Link
            to="/"
            className={cn(
              "text-muted-foreground hover:text-foreground",
              loc.pathname === "/" && "text-foreground",
            )}
          >
            Runs
          </Link>
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "inline-block h-2 w-2 rounded-full",
                healthy ? "bg-emerald-500" : "bg-red-500",
              )}
              aria-label={healthy ? "API healthy" : "API unreachable"}
            />
            <span className="text-xs text-muted-foreground">{healthy ? "healthy" : "offline"}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
