import test from "node:test";
import assert from "node:assert/strict";
import {
  DecisionLoop,
  decisionRequest,
  appliedDecision,
  validateDecision,
  numberHeader,
  GESTURES,
} from "../raya_maas/demo_web/core.js";
const answer = (choice = "fist") => ({
  type: "choice",
  choice,
  confidence: 0.97,
  probabilities: Object.fromEntries(
    GESTURES.map((g) => [g.id, g.id === choice ? 0.97 : 0.005]),
  ),
});

test("request is image + instructions + seven explicit options, never hand coordinates", () => {
  const config = {
    model: "raya-decision-v1",
    instructions: "Count fingers",
    criteria: Object.fromEntries(GESTURES.map((g) => [g.id, g.label])),
  };
  const result = decisionRequest(config, "data:image/jpeg;base64,AA==");
  assert.deepEqual(Object.keys(result), ["model", "state", "questions"]);
  assert.equal(result.state.media[0].type, "image");
  assert.equal(result.questions.gesture.instructions, "Count fingers");
  assert.equal(Object.keys(result.questions.gesture.criteria).length, 7);
  assert(!JSON.stringify(result).includes("coordinates"));
});

test("low confidence and stale responses remain visible but do not control the effect", () => {
  const config = { stale_ms: 2500, confidence_threshold: 0.55 };
  assert.equal(appliedDecision(answer("five"), 100, config).mode, "five");
  assert.equal(appliedDecision(answer("five"), 2501, config).applied, false);
  assert.equal(
    appliedDecision({ ...answer("five"), confidence: 0.54 }, 100, config).mode,
    "none",
  );
  assert.equal(
    validateDecision({ answers: { gesture: answer("two") } }).choice,
    "two",
  );
  assert.throws(() =>
    validateDecision({
      answers: { gesture: { ...answer(), choice: "__proto__" } },
    }),
  );
  assert.throws(() =>
    validateDecision({
      answers: { gesture: { ...answer(), probabilities: {} } },
    }),
  );
});

test("one-second scheduler skips busy ticks without capturing or queuing old frames", async () => {
  let timer,
    configuredInterval,
    resolve,
    captured = 0,
    skipped = 0,
    calls = 0,
    now = 0;
  const outputs = [];
  const loop = new DecisionLoop({
    capture: () => ({ image: `frame-${++captured}`, capturedAt: now }),
    request: () => {
      calls++;
      return new Promise((r) => {
        resolve = r;
      });
    },
    onResult: (r) => outputs.push(r),
    onError: assert.fail,
    onSkip: () => skipped++,
    clock: () => now,
    setTimer: (fn, interval) => {
      timer = fn;
      configuredInterval = interval;
      return 1;
    },
    clearTimer: () => {},
  });
  loop.start();
  assert.equal(configuredInterval, 1000);
  assert.equal(calls, 1);
  now = 1000;
  await timer();
  assert.equal(captured, 1);
  assert.equal(skipped, 1);
  now = 1500;
  resolve({ answer: answer() });
  await new Promise((r) => setImmediate(r));
  assert.equal(outputs[0].wallMs, 1500);
  now = 2000;
  const next = timer();
  assert.equal(captured, 2);
  assert.equal(calls, 2);
  resolve({ answer: answer("one") });
  await next;
  loop.stop();
});

test("stopping invalidates and aborts the in-flight response", async () => {
  let resolve,
    signal,
    calls = 0,
    cleared = 0;
  const loop = new DecisionLoop({
    capture: () => ({ capturedAt: 0 }),
    request: (_, s) => {
      signal = s;
      return new Promise((r) => {
        resolve = r;
      });
    },
    onResult: () => calls++,
    onError: assert.fail,
    setTimer: () => 1,
    clearTimer: () => cleared++,
  });
  loop.start();
  loop.stop();
  assert(signal.aborted);
  assert.equal(cleared, 1);
  resolve({ answer: answer() });
  await new Promise((r) => setImmediate(r));
  assert.equal(calls, 0);
  assert.equal(loop.busy, false);
});

test("missing timing headers are unavailable, not zero", () => {
  assert.equal(numberHeader(new Headers(), "x-raya-forward-ms"), null);
  assert.equal(
    numberHeader(
      new Headers({ "x-raya-forward-ms": "238.25" }),
      "x-raya-forward-ms",
    ),
    238.25,
  );
});
