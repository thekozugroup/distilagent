import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { api, type RunConfig } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ChevronDown, ChevronRight, X } from "lucide-react";
import { cn } from "@/lib/utils";

export default function NewRun() {
  const navigate = useNavigate();
  const { data: pools, isLoading } = useQuery({ queryKey: ["pools"], queryFn: api.pools });
  const [cfg, setCfg] = useState<RunConfig | null>(null);

  useEffect(() => {
    if (pools && !cfg) {
      const defaults = { ...pools.defaults };
      if (!defaults.teacher_pool || defaults.teacher_pool.length === 0) {
        defaults.teacher_pool = pools.generalist_pool;
      }
      if (!defaults.judge_pool || defaults.judge_pool.length === 0) {
        defaults.judge_pool = pools.judge_pool;
      }
      setCfg(defaults);
    }
  }, [pools, cfg]);

  const createRun = useMutation({
    mutationFn: (c: RunConfig) => api.createRun(c),
    onSuccess: (res) => {
      toast.success(`Run started: ${res.run_id.slice(0, 12)}…`);
      navigate(`/runs/${res.run_id}`);
    },
    onError: (err: Error) => toast.error(err.message),
  });

  if (isLoading || !cfg || !pools) {
    return (
      <div className="mx-auto max-w-4xl space-y-4 px-4 py-6">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-40 w-full" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  const update = <K extends keyof RunConfig>(k: K, v: RunConfig[K]) => setCfg((c) => (c ? { ...c, [k]: v } : c));

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    createRun.mutate(cfg);
  };

  return (
    <form onSubmit={submit} className="mx-auto max-w-4xl space-y-4 px-4 py-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">New Run</h1>
        <div className="flex gap-2">
          <Button type="button" variant="outline" onClick={() => navigate("/")}>
            Cancel
          </Button>
          <Button type="submit" disabled={createRun.isPending}>
            {createRun.isPending ? "Starting…" : "Start Run"}
          </Button>
        </div>
      </div>

      <Section title="Provider" defaultOpen>
        <Field label="Provider">
          <select
            value={cfg.provider}
            onChange={(e) => update("provider", e.target.value)}
            className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
          >
            <option value="openrouter">openrouter</option>
            <option value="local">local</option>
          </select>
        </Field>
        <Field label="Base URL (optional)">
          <Input
            value={cfg.base_url ?? ""}
            onChange={(e) => update("base_url", e.target.value || null)}
            placeholder="http://localhost:8000/v1"
          />
        </Field>
        <Field label="Model">
          <Input value={cfg.model ?? ""} onChange={(e) => update("model", e.target.value || null)} />
        </Field>
        <Field label="OpenRouter API Key (optional override)">
          <Input
            type="password"
            value={cfg.openrouter_api_key ?? ""}
            onChange={(e) => update("openrouter_api_key", e.target.value || null)}
            placeholder="sk-or-…"
          />
        </Field>
      </Section>

      <Section title="Pools" defaultOpen>
        <SwitchField label="use_pool" value={cfg.use_pool} onChange={(v) => update("use_pool", v)} />
        <SwitchField label="no_routing" value={cfg.no_routing} onChange={(v) => update("no_routing", v)} />
        <MultiSelectField
          label="Teacher pool"
          available={pools.generalist_pool}
          selected={cfg.teacher_pool ?? []}
          onChange={(v) => update("teacher_pool", v)}
        />
        <MultiSelectField
          label="Judge pool"
          available={pools.judge_pool}
          selected={cfg.judge_pool ?? []}
          onChange={(v) => update("judge_pool", v)}
        />
      </Section>

      <Section title="Tournament">
        <NumberField label="num_judges" value={cfg.num_judges} onChange={(v) => update("num_judges", v)} />
        <NumberField label="max_iterations" value={cfg.max_iterations} onChange={(v) => update("max_iterations", v)} />
        <NumberField label="convergence_k" value={cfg.convergence_k} onChange={(v) => update("convergence_k", v)} />
        <NumberField
          label="max_concurrency"
          value={cfg.max_concurrency}
          onChange={(v) => update("max_concurrency", v)}
        />
      </Section>

      <Section title="Sampling">
        <NumberField
          label="temperature"
          step={0.05}
          value={cfg.temperature}
          onChange={(v) => update("temperature", v)}
        />
        <NumberField label="top_p" step={0.05} value={cfg.top_p} onChange={(v) => update("top_p", v)} />
        <NumberField
          label="repetition_penalty"
          step={0.05}
          value={cfg.repetition_penalty}
          onChange={(v) => update("repetition_penalty", v)}
        />
        <NumberField label="max_tokens" value={cfg.max_tokens} onChange={(v) => update("max_tokens", v)} />
        <SwitchField
          label="enable_thinking"
          value={cfg.enable_thinking}
          onChange={(v) => update("enable_thinking", v)}
        />
      </Section>

      <Section title="Rate limit">
        <NumberField label="rpm" value={cfg.rpm} onChange={(v) => update("rpm", v)} />
        <NumberField label="rpd" value={cfg.rpd} onChange={(v) => update("rpd", v)} />
        <NumberField
          label="llm_call_timeout (s)"
          value={cfg.llm_call_timeout}
          onChange={(v) => update("llm_call_timeout", v)}
        />
      </Section>

      <Section title="Dataset">
        <NumberField label="n_hf" value={cfg.n_hf} onChange={(v) => update("n_hf", v)} />
        <NumberField label="hf_seed" value={cfg.hf_seed} onChange={(v) => update("hf_seed", v)} />
        <SwitchField label="no_agentic" value={cfg.no_agentic} onChange={(v) => update("no_agentic", v)} />
        <NumberField label="rng_seed" value={cfg.rng_seed} onChange={(v) => update("rng_seed", v)} />
      </Section>
    </form>
  );
}

function Section({ title, children, defaultOpen = false }: { title: string; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <Card>
      <CardHeader className="cursor-pointer py-3" onClick={() => setOpen((v) => !v)}>
        <CardTitle className="flex items-center gap-2 text-base">
          {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          {title}
        </CardTitle>
      </CardHeader>
      {open && <CardContent className="grid gap-4 sm:grid-cols-2">{children}</CardContent>}
    </Card>
  );
}

function Field({ label, children, className }: { label: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("space-y-1.5", className)}>
      <Label className="text-xs uppercase tracking-wider text-muted-foreground">{label}</Label>
      {children}
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
  step,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
}) {
  return (
    <Field label={label}>
      <Input
        type="number"
        step={step ?? 1}
        value={Number.isFinite(value) ? value : 0}
        onChange={(e) => {
          const n = Number(e.target.value);
          onChange(Number.isFinite(n) ? n : 0);
        }}
      />
    </Field>
  );
}

function SwitchField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between rounded-md border border-input px-3 py-2">
      <Label className="text-xs uppercase tracking-wider text-muted-foreground">{label}</Label>
      <Switch checked={value} onCheckedChange={onChange} />
    </div>
  );
}

function MultiSelectField({
  label,
  available,
  selected,
  onChange,
}: {
  label: string;
  available: string[];
  selected: string[];
  onChange: (v: string[]) => void;
}) {
  const notSelected = available.filter((m) => !selected.includes(m));
  return (
    <Field label={label} className="sm:col-span-2">
      <div className="flex min-h-[40px] flex-wrap gap-1.5 rounded-md border border-input bg-background p-2">
        {selected.length === 0 && <span className="text-xs text-muted-foreground">None selected</span>}
        {selected.map((m) => (
          <Badge key={m} variant="secondary" className="gap-1">
            <span className="font-mono text-[11px]">{m}</span>
            <button
              type="button"
              onClick={() => onChange(selected.filter((x) => x !== m))}
              className="rounded-sm hover:bg-muted"
              aria-label={`remove ${m}`}
            >
              <X className="h-3 w-3" />
            </button>
          </Badge>
        ))}
      </div>
      {notSelected.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {notSelected.map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => onChange([...selected, m])}
              className="rounded-full border border-dashed border-input px-2.5 py-0.5 font-mono text-[11px] text-muted-foreground hover:border-input hover:bg-accent hover:text-foreground"
            >
              + {m}
            </button>
          ))}
        </div>
      )}
    </Field>
  );
}
