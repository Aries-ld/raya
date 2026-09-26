import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { chromium } from "playwright";
import { GESTURES } from "../raya_maas/demo_web/core.js";

const base = process.env.RAYA_DEMO_URL || "http://127.0.0.1:8765";
await mkdir("artifacts", { recursive: true });
const browser = await chromium.launch({
  channel: process.env.BROWSER_CHANNEL || "chrome",
  headless: true,
  args: [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
  ],
});
const checks = [];
const output = (choice) => ({
  model: "raya-decision-v1",
  answers: {
    gesture: {
      type: "choice",
      choice,
      confidence: 0.97,
      probabilities: Object.fromEntries(
        GESTURES.map((g) => [g.id, g.id === choice ? 0.97 : 0.005]),
      ),
    },
  },
  usage: { input_tokens: 300, output_tokens: 0 },
});
const reply = (route, choice) =>
  route.fulfill({
    status: 200,
    contentType: "application/json",
    headers: {
      "X-Raya-Forward-Ms": "321.5",
      "X-Raya-Processing-Ms": "350.1",
      "X-Raya-Queue-Ms": "0.3",
    },
    body: JSON.stringify(output(choice)),
  });

async function pageFor(options = {}) {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1050 },
    permissions: ["camera"],
    ...options,
  });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/api/health", (route) =>
    route.fulfill({ json: { ready: true, model: "raya-decision-v1" } }),
  );
  return { context, page, errors };
}

try {
  const { context, page, errors } = await pageFor();
  const calls = [],
    choices = ["fist", "one", "two", "three", "four", "five"];
  await page.route("**/api/systemone", async (route) => {
    const payload = route.request().postDataJSON();
    calls.push({
      at: Date.now(),
      captureAt: await page.evaluate(
        () => window.rayaGestureDebug.lastSampleAt,
      ),
      payload,
    });
    await reply(route, choices[Math.min(calls.length - 1, 5)]);
  });
  await page.goto(base);
  await page.waitForFunction(() => window.rayaGestureDebug);
  assert.equal(
    await page.evaluate(() => window.rayaGestureDebug.camera),
    false,
  );
  await page.screenshot({
    path: "artifacts/gesture-demo-empty.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: /开启摄像头/ }).click();
  await page.waitForFunction(() => window.rayaGestureDebug.camera);
  await page.waitForTimeout(1150);
  assert.equal(calls.length, 0, "Preview must not invoke the model");
  await page.getByRole("button", { name: "开始识别" }).click();
  for (const mode of choices) {
    await page.waitForFunction(
      (expected) => window.rayaGestureDebug.mode === expected,
      mode,
      { timeout: 10000 },
    );
  }
  await page.waitForTimeout(400);
  await page.screenshot({
    path: "artifacts/gesture-demo-mocked-five.png",
    fullPage: true,
  });
  await page.waitForFunction(
    () => window.rayaGestureDebug.samples >= 13,
    null,
    { timeout: 18000 },
  );
  assert.equal(await page.locator(".record").count(), 12);
  assert.equal(
    await page
      .locator(".record")
      .first()
      .locator(".record-image")
      .evaluate((img) => img.naturalWidth),
    384,
  );
  assert(
    (await page.locator(".record").first().textContent()).includes("322 ms"),
  );
  assert(
    (await page.locator(".record").first().textContent()).includes("350 ms"),
  );
  assert(
    (
      await page.locator(".record").first().locator("pre").textContent()
    ).includes('"answers"'),
  );
  for (const call of calls) {
    assert.equal(call.payload.state.media.length, 1);
    assert.equal(call.payload.state.media[0].type, "image");
    assert(
      call.payload.state.media[0].url.startsWith("data:image/jpeg;base64,"),
    );
    assert.deepEqual(Object.keys(call.payload.questions.gesture.criteria), [
      "fist",
      "one",
      "two",
      "three",
      "four",
      "five",
      "none",
    ]);
  }
  for (let i = 1; i < calls.length; i++)
    assert(
      calls[i].captureAt - calls[i - 1].captureAt >= 990,
      "Requests must not burst faster than 1Hz",
    );
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出响应和耗时" }).click();
  const download = await downloadPromise;
  const exported = await readFile(await download.path(), "utf8");
  assert(!exported.includes("data:image"));
  assert.equal(JSON.parse(exported).length, 12);
  await page.getByRole("button", { name: "暂停识别" }).click();
  const beforePause = calls.length;
  await page.waitForTimeout(1100);
  assert.equal(calls.length, beforePause);
  assert.equal(await page.evaluate(() => window.rayaGestureDebug.camera), true);
  await page.evaluate(() => {
    window.testTracks = document.querySelector("#camera").srcObject.getTracks();
  });
  await page.getByRole("button", { name: "关闭摄像头" }).click();
  assert(
    await page.evaluate(() =>
      window.testTracks.every((t) => t.readyState === "ended"),
    ),
  );
  assert.equal(await page.locator(".record").count(), 0);
  assert.equal(await page.evaluate(() => window.rayaGestureDebug.samples), 0);
  assert.deepEqual(errors, []);
  checks.push(
    "camera opt-in; preview only; all six effects; 1Hz cadence; per-response JSON/timings; history bound; export; pause; track cleanup",
  );
  await context.close();

  const slow = await pageFor();
  let slowCalls = 0;
  await slow.page.route("**/api/systemone", async (route) => {
    slowCalls++;
    await new Promise((r) => setTimeout(r, 2700));
    await reply(route, "five").catch(() => {});
  });
  await slow.page.goto(base);
  await slow.page.getByRole("button", { name: /开启摄像头/ }).click();
  await slow.page.waitForFunction(() => window.rayaGestureDebug.camera);
  await slow.page.getByRole("button", { name: "开始识别" }).click();
  const animationBefore = await slow.page.evaluate(
    () => window.rayaGestureDebug.frames,
  );
  await slow.page.waitForTimeout(1300);
  assert.equal(slowCalls, 1);
  assert(
    (await slow.page.evaluate(() => window.rayaGestureDebug.frames)) >
      animationBefore + 10,
  );
  assert(
    (await slow.page.evaluate(() => window.rayaGestureDebug.skipped)) >= 1,
  );
  await slow.page.waitForFunction(() => window.rayaGestureDebug.samples >= 1);
  assert.equal(
    await slow.page.evaluate(() => window.rayaGestureDebug.mode),
    "none",
  );
  assert(
    (await slow.page.locator(".record").first().textContent()).includes(
      "响应已过期",
    ),
  );
  await slow.page.getByRole("button", { name: "关闭摄像头" }).click();
  await slow.page.waitForTimeout(1000);
  assert.equal(await slow.page.locator(".record").count(), 0);
  assert.deepEqual(slow.errors, []);
  checks.push(
    "busy ticks skipped; animation independent; stale results not applied; late results ignored after close",
  );
  await slow.context.close();

  const denied = await pageFor();
  await denied.page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => {
      throw new DOMException("Denied", "NotAllowedError");
    };
  });
  await denied.page.goto(base);
  await denied.page.getByRole("button", { name: /开启摄像头/ }).click();
  await denied.page.waitForFunction(() =>
    document.querySelector("#status").textContent.includes("权限被拒绝"),
  );
  assert.equal(
    await denied.page.evaluate(() => window.rayaGestureDebug.camera),
    false,
  );
  assert.equal(
    await denied.page.getByRole("button", { name: "开始识别" }).isEnabled(),
    false,
  );
  checks.push("camera permission denial");
  await denied.context.close();

  const mobile = await pageFor({
    viewport: { width: 390, height: 844 },
    isMobile: true,
  });
  await mobile.page.goto(base);
  await mobile.page.waitForFunction(() => window.rayaGestureDebug);
  await mobile.page.waitForTimeout(700);
  assert(
    await mobile.page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  );
  await mobile.page.screenshot({
    path: "artifacts/gesture-demo-mobile.png",
    fullPage: true,
  });
  assert.deepEqual(mobile.errors, []);
  checks.push("390px mobile layout");
  await mobile.context.close();

  await writeFile(
    "artifacts/gesture-browser-checks.json",
    JSON.stringify(
      {
        passed: true,
        checks,
        limitation:
          "Synthetic camera + mocked decisions: validates UI behavior, not real-hand model accuracy.",
      },
      null,
      2,
    ),
  );
  console.log("PASS:", checks.join("\n"));
} finally {
  await browser.close();
}
