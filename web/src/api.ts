import type { RunEvent, RunListItem, Scenario } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export async function fetchScenarios(): Promise<Scenario[]> {
  const data = await json<{ scenarios: Scenario[] }>(await fetch("/api/scenarios"));
  return data.scenarios;
}

export async function fetchRuns(): Promise<RunListItem[]> {
  const data = await json<{ runs: RunListItem[] }>(await fetch("/api/runs"));
  return data.runs;
}

export interface RunPayload {
  events: RunEvent[];
  summary: Record<string, unknown>;
}

export async function runScenario(scenarioId: string): Promise<RunPayload> {
  return json<RunPayload>(
    await fetch("/api/call", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario_id: scenarioId }),
    })
  );
}

export async function fetchRun(runId: string): Promise<RunPayload> {
  return json<RunPayload>(await fetch(`/api/runs/${runId}`));
}
