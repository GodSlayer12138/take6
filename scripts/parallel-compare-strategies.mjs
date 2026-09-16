import { readFileSync } from 'node:fs'
import { availableParallelism } from 'node:os'
import { dirname, resolve } from 'node:path'
import { isMainThread, parentPort, workerData, Worker } from 'node:worker_threads'

import { playArenaGame, setCounterfactualPolicyModelsForEvaluation, setPolicyV2ModelForEvaluation } from '../src/game/ai-core.js'

const opponents = ['champion', 'external_mcs', 'cautious', 'random']
const metricKeys = ['win', 'score', 'rank']

function loadCounterfactualModel(path) {
  if (!path) return null
  const manifest = JSON.parse(readFileSync(path, 'utf8'))
  if ((manifest.format === 'ntw-counterfactual-ensemble' || manifest.format === 'ntw-counterfactual-gated') && manifest.modelPaths?.length) {
    manifest.models = manifest.modelPaths.map((member) => JSON.parse(readFileSync(resolve(dirname(path), member), 'utf8')))
  }
  return manifest
}

function play(strategy, gameIndex, options) {
  const focalSeat = (5 - (gameIndex % 5)) % 5
  const strategies = Array(5)
  strategies[focalSeat] = strategy
  opponents.forEach((opponent, offset) => {
    strategies[(focalSeat + offset + 1) % 5] = opponent
  })
  const result = playArenaGame({
    playerCount: 5,
    deckMode: 'adaptive',
    strategies,
    seed: options.seed + gameIndex * 65537,
    championSamples: options.samples,
    legacySamples: Math.max(8, Math.round(options.samples / 2)),
  })
  const score = result.scores[focalSeat]
  const minimum = Math.min(...result.scores)
  const winnerCount = result.scores.filter((value) => value === minimum).length
  const win = score === minimum ? 1 / winnerCount : 0
  const lower = result.scores.filter((value) => value < score).length
  const tied = result.scores.filter((value) => value === score).length
  return { win, score, rank: 1 + lower + (tied - 1) / 2 }
}

function emptyAccumulator() {
  return Object.fromEntries(metricKeys.map((key) => [key, { sum: 0, sumSquares: 0 }]))
}

function add(accumulator, key, value) {
  accumulator[key].sum += value
  accumulator[key].sumSquares += value * value
}

function workerRun(options) {
  const referenceModel = JSON.parse(readFileSync(options.referenceModelPath, 'utf8'))
  const candidateModel = JSON.parse(readFileSync(options.candidateModelPath, 'utf8'))
  const referenceCounterfactualModel = loadCounterfactualModel(options.referenceCounterfactualModelPath)
  const candidateCounterfactualModel = loadCounterfactualModel(options.candidateCounterfactualModelPath)
  const referenceRootCounterfactualModel = loadCounterfactualModel(options.referenceRootCounterfactualModelPath) || referenceCounterfactualModel
  const candidateRootCounterfactualModel = loadCounterfactualModel(options.candidateRootCounterfactualModelPath) || candidateCounterfactualModel
  const reference = emptyAccumulator()
  const candidate = emptyAccumulator()
  const delta = emptyAccumulator()
  for (let gameIndex = options.start; gameIndex < options.end; gameIndex += 1) {
    setPolicyV2ModelForEvaluation(referenceModel)
    setCounterfactualPolicyModelsForEvaluation({ root: referenceRootCounterfactualModel, rollout: referenceCounterfactualModel })
    const referenceResult = play(options.referenceStrategy, gameIndex, options)
    setPolicyV2ModelForEvaluation(candidateModel)
    setCounterfactualPolicyModelsForEvaluation({ root: candidateRootCounterfactualModel, rollout: candidateCounterfactualModel })
    const candidateResult = play(options.candidateStrategy, gameIndex, options)
    for (const key of metricKeys) {
      add(reference, key, referenceResult[key])
      add(candidate, key, candidateResult[key])
      add(delta, key, candidateResult[key] - referenceResult[key])
    }
  }
  return { count: options.end - options.start, reference, candidate, delta }
}

function merge(target, source) {
  target.count += source.count
  for (const group of ['reference', 'candidate', 'delta']) {
    for (const key of metricKeys) {
      target[group][key].sum += source[group][key].sum
      target[group][key].sumSquares += source[group][key].sumSquares
    }
  }
}

function summarize(entry, count) {
  const mean = entry.sum / count
  const variance = count > 1 ? Math.max(0, (entry.sumSquares - entry.sum * entry.sum / count) / (count - 1)) : 0
  const standardError = Math.sqrt(variance / count)
  return { mean, standardError, ci95: [mean - 1.96 * standardError, mean + 1.96 * standardError] }
}

if (!isMainThread) {
  parentPort.postMessage(workerRun(workerData))
} else {
  const modelPath = resolve(process.argv[2])
  const games = Number(process.argv[3] || 1000)
  const referenceStrategy = process.argv[4]
  const candidateStrategy = process.argv[5]
  const seed = Number(process.argv[6] || 370000000)
  const samples = Number(process.argv[7] || 18)
  const requestedWorkers = Number(process.argv[8] || Math.min(8, availableParallelism()))
  const referenceModelPath = resolve(process.argv[9] || modelPath)
  const candidateModelPath = resolve(process.argv[10] || modelPath)
  const referenceCounterfactualModelPath = process.argv[11] ? resolve(process.argv[11]) : null
  const candidateCounterfactualModelPath = process.argv[12] ? resolve(process.argv[12]) : null
  const referenceRootCounterfactualModelPath = process.argv[13] ? resolve(process.argv[13]) : null
  const candidateRootCounterfactualModelPath = process.argv[14] ? resolve(process.argv[14]) : null
  if (!referenceStrategy || !candidateStrategy) throw new Error('reference and candidate strategies are required')
  const workerCount = Math.max(1, Math.min(games, requestedWorkers))
  const started = performance.now()
  const jobs = Array.from({ length: workerCount }, (_, index) => {
    const start = Math.floor(index * games / workerCount)
    const end = Math.floor((index + 1) * games / workerCount)
    return new Promise((resolveJob, rejectJob) => {
      const worker = new Worker(new URL(import.meta.url), {
        workerData: { start, end, seed, samples, referenceStrategy, candidateStrategy, referenceModelPath, candidateModelPath, referenceCounterfactualModelPath, candidateCounterfactualModelPath, referenceRootCounterfactualModelPath, candidateRootCounterfactualModelPath },
      })
      worker.once('message', resolveJob)
      worker.once('error', rejectJob)
      worker.once('exit', (code) => { if (code !== 0) rejectJob(new Error(`worker exited with code ${code}`)) })
    })
  })
  const combined = { count: 0, reference: emptyAccumulator(), candidate: emptyAccumulator(), delta: emptyAccumulator() }
  for (const result of await Promise.all(jobs)) merge(combined, result)
  const reference = Object.fromEntries(metricKeys.map((key) => [key, combined.reference[key].sum / games]))
  const candidate = Object.fromEntries(metricKeys.map((key) => [key, combined.candidate[key].sum / games]))
  const delta = Object.fromEntries(metricKeys.map((key) => [key, summarize(combined.delta[key], games)]))
  delta.win.meanPP = delta.win.mean * 100
  delta.win.ci95PP = delta.win.ci95.map((value) => value * 100)
  console.log(JSON.stringify({
    modelPath,
    referenceModelPath,
    candidateModelPath,
    referenceCounterfactualModelPath,
    candidateCounterfactualModelPath,
    referenceRootCounterfactualModelPath,
    candidateRootCounterfactualModelPath,
    games,
    seed,
    samples,
    workers: workerCount,
    referenceStrategy,
    candidateStrategy,
    reference,
    candidate,
    delta,
    seconds: (performance.now() - started) / 1000,
  }))
}
