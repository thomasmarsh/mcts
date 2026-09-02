import { describe, expect, it } from "vitest";
import {
  idle,
  isLoading,
  peek,
  toErr,
  toLoading,
  toOk,
  type RemoteData,
} from "../../src/tuner/remote-data.js";

describe("remote-data", () => {
  it("starts idle with no value", () => {
    const d = idle<number>();
    expect(d.status).toBe("idle");
    expect(peek(d)).toBeUndefined();
  });

  it("carries the previous value through loading and error", () => {
    const ok = toOk(42, 1000);
    const loading = toLoading(ok);
    expect(loading.status).toBe("loading");
    expect(peek(loading)).toBe(42);

    const errored = toErr("boom", loading);
    expect(errored.status).toBe("err");
    expect(peek(errored)).toBe(42);
    expect((errored as { message: string }).message).toBe("boom");
  });

  it("does not resurrect a value once it was never loaded", () => {
    const loading = toLoading(idle<string>());
    expect(peek(loading)).toBeUndefined();
    expect(peek(toErr("x", loading))).toBeUndefined();
  });

  it("toOk records the fetch time and replaces any prior value", () => {
    const d: RemoteData<string> = toOk("v2", 5000);
    expect(d).toEqual({ status: "ok", value: "v2", fetchedAt: 5000 });
  });

  it("isLoading is true only in the loading state", () => {
    expect(isLoading(toLoading(idle()))).toBe(true);
    expect(isLoading(toOk(1, 0))).toBe(false);
    expect(isLoading(toErr("e", idle()))).toBe(false);
  });
});
