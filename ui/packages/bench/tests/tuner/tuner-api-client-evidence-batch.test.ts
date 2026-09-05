// tuner-api-client-evidence-batch.test.ts — `openEvidenceStream` coalesces
// every SSE frame that arrives within one short window into a single
// `onEvents` call. This is the fix for a burst of catch-up frames (or any
// tight cluster of live events) turning into that many synchronous store
// dispatches / re-renders on the main thread -- see the comment above
// `scheduleFlush` in tuner-api-client.ts. A fake `EventSource` drives the
// same `onmessage`/event-listener surface the real one does; no network.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createTunerApiClient } from "../../src/tuner/tuner-api-client.js";
import type { EvidenceStreamHandlers } from "../../src/tuner/tuner-api-client.js";

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onerror: (() => void) | null = null;
  readyState = 0;
  private listeners = new Map<string, Set<(event: MessageEvent<string>) => void>>();

  constructor(public url: string) {
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, handler: (event: MessageEvent<string>) => void): void {
    const set = this.listeners.get(type) ?? new Set();
    set.add(handler);
    this.listeners.set(type, set);
  }

  emit(data: string): void {
    this.onmessage?.({ data } as MessageEvent<string>);
  }

  emitNamed(type: string, data = ""): void {
    for (const handler of this.listeners.get(type) ?? []) handler({ data } as MessageEvent<string>);
  }

  close(): void {
    this.readyState = 2;
  }
}

describe("openEvidenceStream — batches SSE frames arriving in one window", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeEventSource.instances = [];
    vi.stubGlobal("EventSource", FakeEventSource as unknown as typeof EventSource);
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  const openStream = (handlers: Partial<EvidenceStreamHandlers> = {}) => {
    const client = createTunerApiClient();
    const onEvents = vi.fn();
    const sub = client.openEvidenceStream("r1", 0, {
      onEvents,
      onProjectionUpdated: vi.fn(),
      onEnd: vi.fn(),
      onError: vi.fn(),
      ...handlers,
    });
    const source = FakeEventSource.instances.at(-1)!;
    return { source, onEvents, sub };
  };

  it("delivers a burst of frames as one onEvents call", () => {
    const { source, onEvents } = openStream();
    for (let seq = 1; seq <= 500; seq += 1) {
      source.emit(JSON.stringify({ sequence: seq, type: "pair_started", payload: {} }));
    }
    expect(onEvents).not.toHaveBeenCalled();

    vi.runAllTimers();

    expect(onEvents).toHaveBeenCalledTimes(1);
    const [batch] = onEvents.mock.calls[0]!;
    expect(batch).toHaveLength(500);
    expect(batch[0].sequence).toBe(1);
    expect(batch[499].sequence).toBe(500);
  });

  it("starts a fresh batch after each flush, for frames spread across windows", () => {
    const { source, onEvents } = openStream();
    source.emit(JSON.stringify({ sequence: 1, type: "pair_started", payload: {} }));
    vi.runAllTimers();
    source.emit(JSON.stringify({ sequence: 2, type: "pair_completed", payload: {} }));
    vi.runAllTimers();

    expect(onEvents).toHaveBeenCalledTimes(2);
    expect(onEvents.mock.calls[0]![0]).toHaveLength(1);
    expect(onEvents.mock.calls[1]![0]).toHaveLength(1);
  });

  it("flushes any pending batch before a projection-updated / end notification", () => {
    const { source, onEvents } = openStream();
    source.emit(JSON.stringify({ sequence: 1, type: "pair_completed", payload: {} }));
    source.emitNamed("projection-updated");
    expect(onEvents).toHaveBeenCalledTimes(1);
    expect(onEvents.mock.calls[0]![0]).toHaveLength(1);
  });
});
