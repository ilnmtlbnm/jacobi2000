// Capability detection and the backend ladder, browser side (the Rust side
// is lean-runtime's backend.rs, same rule). Detection only: no benchmark,
// no tuning. Works in a window or a worker.

export async function capabilities() {
  const caps = {
    hardwareConcurrency: navigator.hardwareConcurrency || 1,
    crossOriginIsolated: self.crossOriginIsolated === true,
    sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    adapter: "none",
    hasAdapter: false,
  };
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (adapter) {
        caps.hasAdapter = true;
        const i = adapter.info || {};
        caps.adapter = [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(" / ") || "WebGPU adapter";
      } else {
        caps.adapter = "navigator.gpu present, no adapter granted";
      }
    } catch (e) {
      caps.adapter = `requestAdapter failed: ${e && e.message ? e.message : e}`;
    }
  } else {
    caps.adapter = "no navigator.gpu (WebGPU not available in this browser)";
  }
  caps.threadsCapable = caps.crossOriginIsolated && caps.sharedArrayBuffer && caps.hardwareConcurrency > 1;
  return caps;
}

// Backends this device can run, best first ("webgpu", "threads", "single"),
// plus a reason string for each one that's missing, keyed by backend id.
export function availableBackends(caps) {
  const available = [];
  const reasons = {};
  if (caps.hasAdapter) {
    available.push("webgpu");
  } else {
    reasons.webgpu = caps.adapter;
  }
  if (caps.threadsCapable) {
    available.push("threads");
  } else {
    reasons.threads =
      !caps.crossOriginIsolated
        ? "not cross-origin isolated"
        : !caps.sharedArrayBuffer
        ? "no SharedArrayBuffer"
        : "one hardware thread";
  }
  available.push("single");
  return { available, reasons };
}
