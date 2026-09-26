export const GESTURES = [
  { id: "fist", number: "0", label: "拳头", effect: "引力核" },
  { id: "one", number: "1", label: "一根手指", effect: "光柱" },
  { id: "two", number: "2", label: "两根手指", effect: "双环" },
  { id: "three", number: "3", label: "三根手指", effect: "三叶旋" },
  { id: "four", number: "4", label: "四根手指", effect: "四象轨道" },
  { id: "five", number: "5", label: "张开手掌", effect: "星芒绽放" },
  { id: "none", number: "—", label: "无清晰手势", effect: "待机星尘" },
];
export const gestureById = (id) => GESTURES.find((g) => g.id === id);

export function decisionRequest(config, image) {
  return {
    model: config.model,
    state: {
      text: "A current unmirrored camera crop containing the hand to classify.",
      media: [{ type: "image", url: image }],
    },
    questions: {
      gesture: {
        type: "choice",
        instructions: config.instructions,
        criteria: config.criteria,
      },
    },
  };
}

export function validateDecision(output) {
  const answer = output?.answers?.gesture;
  if (!answer || answer.type !== "choice" || !gestureById(answer.choice))
    throw new Error("模型返回了未知手势。");
  const values = GESTURES.map((g) => answer.probabilities?.[g.id]);
  if (
    values.some(
      (p) => typeof p !== "number" || !Number.isFinite(p) || p < 0 || p > 1,
    ) ||
    Math.abs(values.reduce((a, b) => a + b, 0) - 1) > 0.001 ||
    typeof answer.confidence !== "number" ||
    !Number.isFinite(answer.confidence) ||
    answer.confidence < 0 ||
    answer.confidence > 1
  )
    throw new Error("模型返回的概率分布无效。");
  return answer;
}

export function appliedDecision(answer, ageMs, config) {
  if (ageMs > config.stale_ms)
    return { mode: "none", applied: false, reason: "响应已过期，未应用" };
  if (answer.confidence < config.confidence_threshold)
    return { mode: "none", applied: false, reason: "低置信度，未应用" };
  return {
    mode: answer.choice,
    applied: true,
    reason: `已应用 · ${gestureById(answer.choice).effect}`,
  };
}

export function numberHeader(headers, name) {
  const raw = headers.get(name);
  if (raw === null || raw.trim() === "") return null;
  const value = Number(raw);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

export class DecisionLoop {
  constructor({
    capture,
    request,
    onResult,
    onError,
    onSkip = () => {},
    eligible = () => true,
    intervalMs = 1000,
    clock = () => performance.now(),
    setTimer = (callback, delay) => globalThis.setInterval(callback, delay),
    clearTimer = (timer) => globalThis.clearInterval(timer),
  }) {
    Object.assign(this, {
      capture,
      request,
      onResult,
      onError,
      onSkip,
      eligible,
      intervalMs,
      clock,
      setTimer,
      clearTimer,
    });
    this.active = false;
    this.busy = false;
    this.epoch = 0;
    this.controller = null;
    this.timer = null;
    this.lastSampleAt = -Infinity;
  }
  start() {
    if (this.active) return;
    this.active = true;
    this.epoch++;
    this.lastSampleAt = -Infinity;
    this.timer = this.setTimer(() => this.tick(), this.intervalMs);
    this.tick();
  }
  stop() {
    this.active = false;
    this.epoch++;
    if (this.timer !== null) this.clearTimer(this.timer);
    this.timer = null;
    this.controller?.abort();
  }
  async tick() {
    if (!this.active || !this.eligible()) return;
    if (this.busy) {
      this.onSkip();
      return;
    }
    // A delayed browser interval must not cause catch-up bursts of camera captures.
    if (this.clock() - this.lastSampleAt < this.intervalMs - 5) return;
    this.lastSampleAt = this.clock();
    this.busy = true;
    const epoch = this.epoch;
    const controller = new AbortController();
    this.controller = controller;
    try {
      const snapshot = this.capture();
      const started = this.clock();
      const result = await this.request(snapshot, controller.signal);
      if (!this.active || epoch !== this.epoch) return;
      this.onResult({
        ...result,
        snapshot,
        wallMs: this.clock() - started,
        ageMs: this.clock() - snapshot.capturedAt,
      });
    } catch (error) {
      if (epoch === this.epoch && this.active && error.name !== "AbortError") {
        this.stop();
        this.onError(error);
      }
    } finally {
      this.busy = false;
      if (this.controller === controller) this.controller = null;
    }
  }
}
