// main.ts — Entrypoint for token-radar-01 (Node 22, standard library only).
//
// Reads the sealed DefiLlama protocol snapshot from /input/snapshot.json (or --input <path>),
// fits the Empirical-Bayes peer-multiple and token-odds model in lib/model.ts, validates all 7
// gate invariants on board.json, writes /output/board.json and /output/data.js, copies dashboard/
// and screenshots/ into /output, and builds /output/dossier.json via vendor/makefun-board.ts.

import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { boardIssue, buildRadar, dataScript, parseSnapshot } from './lib/model.ts';
import { buildDossier } from './vendor/makefun-board.ts';

const here = fileURLToPath(new URL('.', import.meta.url));

const TEXT_MAX_BYTES = 65_536;
const SCREENSHOT_CAPS = { max: 4, max_bytes: 524_288 };
const DASHBOARD_CAPS = { max_files: 400, max_bytes: 4_194_304 };
const IMAGE_NAME = /^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$/;

function readArg(name: string, fallback: string): string {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx >= 0 && idx + 1 < process.argv.length) return process.argv[idx + 1]!;
  const eq = process.argv.find((a) => a.startsWith(`--${name}=`));
  return eq === undefined ? fallback : eq.slice(name.length + 3);
}

const inputArg = readArg('input', '/input');
const inputFile = inputArg.endsWith('.json') ? inputArg : join(inputArg, 'snapshot.json');
if (!existsSync(inputFile)) {
  process.stderr.write(`token-radar: missing snapshot at ${inputFile}; pass --input <snapshot.json|dir>\n`);
  process.exit(2);
}

const outDir = readArg('output', process.env.MAKEFUN_OUTPUT_DIR ?? '/output');

const snapshot = parseSnapshot(readFileSync(inputFile, 'utf8'));
const { board, radar } = buildRadar(snapshot);
const issue = boardIssue(board, snapshot);
if (issue !== null) {
  process.stderr.write(`token-radar: board validation failed: ${issue}\n`);
  process.exit(1);
}

mkdirSync(outDir, { recursive: true, mode: 0o755 });
writeFileSync(join(outDir, 'board.json'), `${JSON.stringify({ rows: board })}\n`, { mode: 0o644 });
writeFileSync(join(outDir, 'data.js'), dataScript(radar), { mode: 0o644 });

function copyPlainDirectory(srcDir: string, dstDir: string): string[] {
  if (!existsSync(srcDir)) return [];
  mkdirSync(dstDir, { recursive: true, mode: 0o755 });
  const names = readdirSync(srcDir)
    .filter((n) => !n.startsWith('.'))
    .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  const copied: string[] = [];
  for (const name of names) {
    const srcPath = join(srcDir, name);
    if (!statSync(srcPath).isFile()) continue;
    const dstPath = join(dstDir, name);
    copyFileSync(srcPath, dstPath);
    copied.push(name);
  }
  return copied;
}

const dashFiles = copyPlainDirectory(join(here, 'dashboard'), join(outDir, 'dashboard'));
const shotFiles = copyPlainDirectory(join(here, 'screenshots'), join(outDir, 'screenshots'));

if (shotFiles.length < 3 || shotFiles.length > 4) {
  process.stderr.write(`token-radar: screenshots/ holds ${shotFiles.length} images (requires 3..4)\n`);
  process.exit(1);
}
for (const name of shotFiles) {
  if (!IMAGE_NAME.test(name)) {
    process.stderr.write(`token-radar: invalid screenshot filename ${name}\n`);
    process.exit(1);
  }
}

const dossierBytes = buildDossier(outDir, {
  text_max_bytes: TEXT_MAX_BYTES,
  screenshots: SCREENSHOT_CAPS,
  dashboard: DASHBOARD_CAPS,
});
writeFileSync(join(outDir, 'dossier.json'), dossierBytes, { mode: 0o644 });

process.stdout.write(
  `token-radar (${radar.model_version}): ${snapshot.protocols.length} protocols (${snapshot.watchlist.length} watchlist, ${board.length} ranked, ${radar.early_size} early, ${radar.live_size} live benchmarks)\n` +
    `backtest: LOO x${radar.validation.loo_median_error_x} (vs starter x${radar.validation.starter_median_error_x}), Walk-Forward x${radar.validation.walk_forward_error_x}, Brier ${radar.validation.brier_loo}\n` +
    `wrote ${dashFiles.length} dashboard files, ${shotFiles.length} screenshots, and dossier.json (${dossierBytes.length} bytes)\n`
);
